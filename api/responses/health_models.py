from pydantic import BaseModel
from typing import List, Dict, Any

class HealthResponse(BaseModel):
    status: str
    available_models: List[str]
    message: str