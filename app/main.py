from fastapi import FastAPI, UploadFile, File
from app.schemas import PredictionResponse
from app.model import load_model, predict_image
from PIL import Image

app = FastAPI(title="VitiScan Diagno API")
model = load_model()

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)
    disease, confidence = predict_image(image)
    return PredictionResponse(
        disease=disease,
        confidence=confidence,
        model_version="v1.0"
    )
