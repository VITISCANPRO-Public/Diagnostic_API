from pydantic import BaseModel

class PredictionResponse(BaseModel):
    disease: str
    confidence: float
    mlflow_uri: str
    api_uri: str