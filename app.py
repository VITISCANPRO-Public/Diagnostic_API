"""
====================================================================================================
                                    DIAGNOSTIC API - Vitiscan
         REST API for grape leaf disease classification using a fine-tuned CNN model
====================================================================================================
"""

#                                         LIBRARIES IMPORT
# ================================================================================================

import os
import json
import tempfile
import logging
import boto3
import mlflow
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import uvicorn

from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from schemas import PredictionResponse, DiseasePrediction, DiseasesResponse


#                                         CONFIGURATION
# ================================================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

DEVICE = 'cpu'
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_MODEL_URI = os.getenv("MLFLOW_MODEL_URI")
DATASET_NAME = os.getenv("DATASET_NAME", "inrae")
MODEL_ARTIFACT_ROOT = os.getenv("MODEL_ARTIFACT_ROOT")
if not MLFLOW_MODEL_URI and not os.getenv("TESTING"):
    logger.error("MLFLOW_MODEL_URI is not set — API will crash on /diagno calls")

# ── The 7 classes (alphabetical order, matches ImageFolder) ──────
CLASS_NAMES = sorted([
    "colomerus_vitis",
    "elsinoe_ampelina",
    "erysiphe_necator",
    "guignardia_bidwellii",
    "healthy",
    "phaeomoniella_chlamydospora",
    "plasmopara_viticola",
])


#                                         MOCK MODEL (CI / TESTING)
# ================================================================================================

class MockModel(nn.Module):
    """
    Lightweight mock model used exclusively in CI (GitHub Actions).

    Returns uniform logits across all 7 INRAE classes so that softmax
    produces equal probabilities (~0.143 per class). This is enough to
    validate that all API endpoints work correctly without needing a GPU
    or access to MLflow / S3 secrets.

    Activated when the environment variable TESTING=true is set.
    Never used in production.
    """
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        # Uniform logits → softmax gives 1/7 per class
        return torch.ones(batch_size, self.num_classes)


#                                         HELPER FUNCTIONS
# ================================================================================================

def load_diseases(bucket_name: str, dataset_name: str) -> dict:
    """
    Loads the disease label dictionary from S3.

    Retrieves a disease-{dataset_name}.json file and returns it as a dictionary:
    { 'disease_scientific_name': 'disease_common_name', ... }

    Args:
        bucket_name: S3 bucket name
        dataset_name: Dataset identifier ('inrae' or 'kaggle')

    Returns:
        Dictionary mapping disease keys to human-readable names
    """
    diseases = {'N/A': 'N/A'}
    disease_filename = f'disease-{dataset_name}.json'

    # Primary path: retrieve from model artifact directory
    primary_path = f'{MODEL_ARTIFACT_ROOT}/extra_files/{disease_filename}'
    fallback_path = f'vitiscan-data/{disease_filename}'

    s3 = boto3.client('s3')
    for s3_path in [primary_path, fallback_path]:
        try:
            response = s3.get_object(Bucket=bucket_name, Key=s3_path)
            data = response['Body'].read().decode('utf-8')
            diseases = json.loads(data)
            logger.info(f"Diseases loaded from s3://{bucket_name}/{s3_path}")
            logger.info(json.dumps(diseases, indent=4, ensure_ascii=True))
            return diseases
        except Exception as e:
            logger.warning(f"Could not load diseases from s3://{bucket_name}/{s3_path}: {e}")

    logger.error("Failed to load diseases from all S3 paths. Using default N/A.")
    return diseases


def predict_image(model, input_tensor: torch.Tensor, class_names: list) -> list:
    """
    Runs inference on a preprocessed image tensor.

    Args:
        model: Loaded PyTorch model
        input_tensor: Preprocessed image tensor of shape (1, 3, H, W)
        class_names: Ordered list from ImageFolder (alphabetical order)

    Returns:
        List of (disease_key, confidence_score) tuples sorted by confidence descending
    """
    with torch.no_grad():
        model.eval()
        output = model(input_tensor)
        probs  = torch.nn.functional.softmax(output, dim=1)[0]
        predictions = [
            (class_names[i], float(probs[i]))
            for i in range(len(class_names))
        ]
        predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions


#                                         STARTUP / SHUTDOWN
# ================================================================================================

async def startup():
    """
    API startup handler.

    Two modes depending on the TESTING environment variable:

    ── Production mode (TESTING unset) ──────────────────────────────────────────
    - Loads the real ResNet18 fine tuned model from MLflow model registry
    - Loads the disease label dictionary from S3
    - Initializes the image preprocessing pipeline

    ── Test mode (TESTING=true, used by GitHub Actions) ─────────────────────────
    - Loads a lightweight MockModel (no GPU, no cloud credentials needed)
    - Uses a hardcoded dictionary with the 7 classes
    - Skips all S3 and MLflow calls
    """
    logger.info("Starting Vitiscan Diagnostic API...")

    TESTING = os.getenv("TESTING", "false").lower() == "true"

    # ── Image preprocessing pipeline (identical in both modes) ────────────────
    app.state.transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    if TESTING:
        # ── Test mode ─────────────────────────────────────────────────────────
        logger.info("TESTING=true detected — loading mock model for CI environment")

        app.state.model = MockModel(num_classes=len(CLASS_NAMES))
        app.state.model_name = "mock-model-ci"

        # Hardcoded disease dictionary with the 7 classes
        app.state.diseases = {cls: cls.replace("_", " ").title() for cls in CLASS_NAMES}

        logger.info(f"Mock model loaded. Classes: {CLASS_NAMES}")
        logger.info("API startup complete (test mode).")

    else:
        # ── Production mode ───────────────────────────────────────────────────
        try:
            # Connect to MLflow
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")

            # Load model directly from MLflow registry
            # MLFLOW_MODEL_URI format: "models:/model_name/version"
            app.state.model = mlflow.pytorch.load_model(MLFLOW_MODEL_URI, map_location=DEVICE)
            app.state.model_name = MLFLOW_MODEL_URI.split("/")[1] if MLFLOW_MODEL_URI.startswith("models:/") \
                else "vitiscan-resnet18"
            logger.info(f"Model '{app.state.model_name}' loaded from MLflow: {MLFLOW_MODEL_URI}")

            # Load disease labels from S3
            S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
            app.state.diseases = load_diseases(bucket_name=S3_BUCKET_NAME, dataset_name=DATASET_NAME)

            logger.info("API startup complete (production mode).")

        except Exception as e:
            logger.error(f"Error during API startup: {e}")
            raise


async def shutdown():
    """API shutdown handler."""
    logger.info("Shutting down Vitiscan Diagnostic API.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup()
    yield
    await shutdown()


#                                         API ENDPOINTS
# ================================================================================================

app = FastAPI(
    lifespan=lifespan,
    title="Vitiscan Diagnostic API",
    description="Grape leaf disease classification using fine-tuned CNN models.",
    version="1.0.0"
)

# ── CORS — allows cross-origin requests from the Streamlit WebUI ─────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://mouniat-vitiscan-streamlit.hf.space",
        "http://localhost:8501",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    """Health check endpoint — verifies that model and diseases are loaded."""
    model_loaded = hasattr(app.state, "model") and app.state.model is not None
    diseases_loaded = hasattr(app.state, "diseases") and app.state.diseases is not None

    if model_loaded and diseases_loaded:
        return {
            "message": "Vitiscan Diagnostic API is running",
            "status": "ok",
            "model": app.state.model_name,
            "num_classes": len(CLASS_NAMES)
        }

    return JSONResponse(
        status_code=503,
        content={
            "status": "unavailable",
            "model_loaded": model_loaded,
            "diseases_loaded": diseases_loaded
        }
    )

@app.get("/diseases", response_model=DiseasesResponse)
async def get_diseases():
    """
    Returns the list of detectable diseases and their labels.
    """
    return DiseasesResponse(
        diseases=app.state.diseases,
        dataset_name=DATASET_NAME
    )


@app.post("/diagno", response_model=PredictionResponse)
async def diagno(file: UploadFile = File(...)):
    """
    Runs disease diagnosis on an uploaded grape leaf image.

    Accepts a JPEG or PNG image file and returns a ranked list of
    disease predictions with confidence scores.

    Args:
        file: Uploaded image file (JPEG or PNG)

    Returns:
        PredictionResponse with ranked disease predictions and model version
    """
    # ── Validate file type ────────────────────────────────────────────────
    ALLOWED_TYPES = {"image/jpeg", "image/png"}
    if file.content_type not in ALLOWED_TYPES:
        return JSONResponse(
            status_code=400,
            content={
                "message": f"Invalid file type: '{file.content_type}'. "
                           f"Accepted types: JPEG, PNG."
            }
        )

    # ── Read and validate file size ────────────────────────────────────────
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        return JSONResponse(
            status_code=413,
            content={
                "message": f"File too large: {len(contents) / (1024 * 1024):.1f} MB. "
                           f"Maximum allowed: {MAX_FILE_SIZE / (1024 * 1024):.0f} MB."
            }
        )

    # ── Save to temporary file ────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(contents)
        tmp_file_path = tmp_file.name
    logger.info(f"Uploaded image saved to temporary file: {tmp_file_path}")

    try:
        # Load and preprocess image
        pil_image = Image.open(tmp_file_path).convert("RGB")
        tensor = app.state.transform(pil_image).unsqueeze(0)
        logger.info("Image converted to RGB tensor successfully")

        # Move model and tensor to target device
        app.state.model.to(DEVICE)
        tensor = tensor.to(DEVICE)
        logger.info(f"Running inference with model '{app.state.model_name}' on device: {DEVICE}")

        # Run inference
        raw_predictions = predict_image(app.state.model, tensor, CLASS_NAMES)
        predictions = [
            DiseasePrediction(disease=d, confidence=c)
            for d, c in raw_predictions
        ]

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": "Prediction failed. See server logs for details."}
        )
    finally:
        os.unlink(tmp_file_path)
        logger.info(f"Temporary file deleted: {tmp_file_path}")

    return PredictionResponse(
        predictions=predictions,
        model_version=app.state.model_name
    )


#                                         ENTRY POINT
# ================================================================================================

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=4000, reload=True)