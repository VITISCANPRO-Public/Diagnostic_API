from pydantic import BaseModel
from typing import List

class DiseasePrediction(BaseModel):
    disease: str
    confidence: float

class PredictionResponse(BaseModel):
    predictions: List[DiseasePrediction]
    model_version: str



#class PredictionResponse(BaseModel):
#    disease: str
#    confidence: float
#    model_version: str