from pydantic import BaseModel
from typing import List


class DiseasePrediction(BaseModel):
    """One disease prediction: a class name and its confidence score."""
    disease:str
    confidence: float


class PredictionResponse(BaseModel):
    """Response returned by POST /diagno."""
    predictions: List[DiseasePrediction]
    model_version: str


class DiseasesResponse(BaseModel):
    """Response returned by GET /diseases."""
    diseases: dict
    dataset_name: str