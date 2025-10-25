from pydantic import BaseModel
from typing import Dict, Optional, List

class MovieRecommendationRequest(BaseModel):
    query: str
    # Parámetros para recomendaciones
    movie_title: Optional[str] = None
    movie_id: Optional[int] = None
    user_id: Optional[int] = None
    genre: Optional[str] = None
    num_recommendations: Optional[int] = 5

class MovieRecommendationResponse(BaseModel):
    recommendations: List[Dict]
    model_info: Dict
    interpretation: str

class MovieRatingRequest(BaseModel):
    query: str
    user_id: int
    movie_id: int

class MovieRatingResponse(BaseModel):
    predicted_rating: float
    confidence: float
    model_info: Dict
    interpretation: str