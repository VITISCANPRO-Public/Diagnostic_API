from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from schemas import PredictionResponse, DiseasePrediction
from PIL import Image
from model import load_model, preprocess_image,predict_image
import os
from dotenv import load_dotenv
import uvicorn

load_dotenv()

app = FastAPI(title="VitiScan Diagno API")

model = load_model()
# model=None # For testing the API without depending of Mlflow

@app.get("/")
def root():
    return {"message": "Vitiscan Diagno API is running"}

'''
# Réponse factice
@app.post("/diagno", response_model=PredictionResponse)
async def diagno(file: UploadFile = File(...)):
    if file is None:
        return JSONResponse(status_code=400, content={"message": "Aucun fichier reçu"})
    
    # Réponse factice : mock response
    # fake_prediction = "mildiou"
    # fake_confidence = 0.87
    # fake_model_version = "Resnet18_50ep_v2"

    predictions = [DiseasePrediction(disease="mildiou", confidence=0.87)]
    return PredictionResponse(
        predictions=predictions,
        model_version="Resnet18_50ep_v2"
    )

    # return PredictionResponse(
    #     disease=fake_prediction,
    #     confidence=fake_confidence,
    #     model_version=fake_model_version
    # )
'''

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file is None:
        return JSONResponse(status_code=400, content={"message": "Aucun fichier reçu"})
    
    contents = await file.read()
    image = preprocess_image(contents)
    raw_predictions = predict_image(model, image) # Prédictions
    predictions = [DiseasePrediction(disease=d, confidence=c) for d, c in raw_predictions]
    
    return PredictionResponse(
        predictions=predictions,
        model_version="vitiscan_resnet@production"
        )

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=4000, reload=True)
