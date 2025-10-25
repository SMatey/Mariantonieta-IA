from pydantic import BaseModel
from typing import Dict, Optional

class PropertyPredictionRequest(BaseModel):
    query: str
    # Características de la propiedad
    bathroomcnt: Optional[float] = None
    bedroomcnt: Optional[float] = None
    finishedsquarefeet: Optional[float] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    lotsizesquarefeet: Optional[float] = None
    yearbuilt: Optional[float] = None
    taxamount: Optional[float] = None
    assessmentyear: Optional[float] = None
    landtaxvaluedollarcnt: Optional[float] = None
    structuretaxvaluedollarcnt: Optional[float] = None
    censustractandblock: Optional[float] = None

class PropertyPredictionResponse(BaseModel):
    prediction: float
    confidence: float
    model_info: Dict
    interpretation: str