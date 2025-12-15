import mlflow
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
import os
from dotenv import load_dotenv

load_dotenv()


# ---------------------  LOADING THE MODEL FROM MLFLOW ----------------------


def load_model():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI","https://mlflow.monserveur.com")
    model_uri = os.getenv("MLFLOW_MODEL_URI")

    mlflow.set_tracking_uri(tracking_uri)

    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    return model


# -----------------------  PREPROCESSING OF THE IMAGE ------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ]) 

def preprocess_image(file_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    return tensor


# ---------------------------------  PREDICTION -------------------------------

# Classes names
disease = [
    "anthracnose",
    "brown_spot",
    "downy_mildew",
    "mites",
    "normal",
    "powdery_mildew",
    "shot_hole"
]

def predict_image(model, image: Image.Image):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0] 
        predictions = [(disease[i], float(probs[i])) for i in range(len(disease))] # Création de la liste DiseasePrediction
        predictions.sort(key=lambda x: x[1], reverse=True) # Tri décroissante de la confiance
     
    return predictions