from fastapi import FastAPI, UploadFile, File
from app.schemas import PredictionResponse
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
API_URI = os.getenv("API_URI")

app = FastAPI(title="VitiScan Diagno API")

@app.get("/")
def root():
    return {"message": "Vitiscan Diagno API is running"}


# Réponse factice : mock response

@app.post("/diagno", response_model=PredictionResponse)
async def diagno(file: UploadFile = File(...)):
    """
    Pour l'instant : réponse factice (mock)
    """
    fake_prediction = "mildiou"
    fake_confidence = 0.87

    return PredictionResponse(
        maladie=fake_prediction,
        confidence=fake_confidence,
        mlflow_uri=MLFLOW_TRACKING_URI,
        api_uri=API_URI
    )


'''
# from app.model import load_model, predict_image
# model = load_model(mlflow_uri=MLFLOW_TRACKING_URI)
#
# @app.post("/predict", response_model=PredictionResponse)
# async def predict(file: UploadFile = File(...)):
#     image = Image.open(file.file).convert("RGB")
#     disease, confidence = predict_image(model, image)
#     return PredictionResponse(
#         disease=disease,
#         confidence=confidence,
#         mlflow_uri=MLFLOW_TRACKING_URI,
#         api_uri=API_URI
#     )
'''

