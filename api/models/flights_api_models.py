from pydantic import BaseModel
from typing import Dict, Optional

class FlightPredictionRequest(BaseModel):
    query: str
    # Parámetros específicos del vuelo
    date: Optional[str] = None                    # "2025-10-25"
    departure_time: Optional[str] = None          # "07:00"
    origin: Optional[str] = None                  # "SFO"
    destination: Optional[str] = None             # "JFK"
    airline: Optional[str] = None                 # "UA"
    distance: Optional[float] = None              # 2586
    delay_at_departure: Optional[float] = 0       # 0

class FlightPredictionResponse(BaseModel):
    query: str
    prediction: float
    confidence: float
    flight_info: Dict
    model_info: Dict
    interpretation: str