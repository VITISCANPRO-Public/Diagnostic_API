from fastapi import FastAPI, UploadFile, File
from app.schemas import PredictionResponse
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.monserveur.com")
API_URI = os.getenv("API_URI", "http://localhost:7860")

app = FastAPI(title="VitiScan Diagno API")

@app.get("/")
def root():
    return {"message": "Vitiscan Diagno API is running"}


# Réponse factice : mock response

@app.post("/diagno", response_model=PredictionResponse)
async def diagno(file: UploadFile = File(...)):
    fake_prediction = "mildiou"
    fake_confidence = 0.87

    return PredictionResponse(
        disease=fake_prediction,
        confidence=fake_confidence,
        mlflow_uri=MLFLOW_TRACKING_URI,
        api_uri=API_URI
    )


'''
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

