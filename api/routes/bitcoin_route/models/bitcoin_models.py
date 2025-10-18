from pydantic import BaseModel
import pandas as pd
from datetime import datetime
from typing import Dict, Optional


class PredictionRequest(BaseModel):
    query: str
    # Datos REALES del usuario - REQUERIDOS para predicción precisa
    open_price: Optional[float] = None
    high_price: Optional[float] = None
    low_price: Optional[float] = None
    volume: Optional[float] = None
    market_cap: Optional[float] = None
    ma_5: Optional[float] = None
    ma_10: Optional[float] = None
    ma_20: Optional[float] = None
    close_lag_1: Optional[float] = None
    close_lag_2: Optional[float] = None
    close_lag_3: Optional[float] = None
    close_lag_5: Optional[float] = None
    rsi_14: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float
    model_info: Dict
    interpretation: str