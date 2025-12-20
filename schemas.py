from pydantic import BaseModel
from typing import List

class DiseasesResponse(BaseModel):
    diseases: dict
    dataset_name: str

class DiseasePrediction(BaseModel):
    disease: str
    confidence: float

class PredictionResponse(BaseModel):
    predictions: List[DiseasePrediction]
    model_version: str