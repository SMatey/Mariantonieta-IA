from pydantic import BaseModel
from typing import Dict

class PredictionRequest(BaseModel):
    dates: list[str]  # lista de fechas a predecir (YYYY-MM-DD)

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float
    model_info: Dict
    interpretation: str