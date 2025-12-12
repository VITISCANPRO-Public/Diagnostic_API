from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from schemas import PredictionResponse, DiseasePrediction
from PIL import Image
from model import load_model, predict_image
import os
from dotenv import load_dotenv
import uvicorn

load_dotenv()

PORT = os.getenv("PORT", "4000")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.monserveur.com")
API_URI = os.getenv("API_URI", "http://localhost:7860")

app = FastAPI(title="VitiScan Diagno API")

model = load_model(mlflow_uri=MLFLOW_TRACKING_URI) # A revoir 

@app.get("/")
def root():
    return {"message": "Vitiscan Diagno API is running"}

'''
@app.post("/diagno", response_model=PredictionResponse)
async def diagno(file: UploadFile = File(...)):
    if file is None:
        return JSONResponse(status_code=400, content={"message": "Aucun fichier reçu"})
    
    # Réponse factice : mock response
    fake_prediction = "mildiou"
    fake_confidence = 0.87
    fake_model_version = "Resnet18_50ep_v2"

    return PredictionResponse(
        disease=fake_prediction,
        confidence=fake_confidence,
        model_version=fake_model_version
    )
'''

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file is None:
        return JSONResponse(status_code=400, content={"message": "Aucun fichier reçu"})
    image = Image.open(file.file).convert("RGB") # Lecture de l'image
    #image_bytes = await file.read()
    #image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    raw_predictions = predict_image(model, image) # Prédictions
    predictions = [DiseasePrediction(disease=d, confidence=c) for d, c in raw_predictions]
    return PredictionResponse(
        predictions=predictions,
        model_version="Resnet18_50ep_v2"
        #disease=disease,
        #confidence=confidence,
        #mlflow_uri=MLFLOW_TRACKING_URI,
        #api_uri=API_URI
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=4000, reload=True)
