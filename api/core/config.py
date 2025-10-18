from typing import List

APP_CONFIG = {
    "title": "AI Models API Hub",
    "description": "API centralizada para múltiples modelos de ML",
    "version": "1.0.0",
}

LOG_LEVEL = "info"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000

CORS_ORIGINS: List[str] = ["*"]

MODEL_ENDPOINTS = {
    "bitcoin": "/bitcoin/models/bitcoin/predict",
    "properties": "/properties/models/properties/predict",
    "movies_recommend": "/movies/models/movies/recommend",
    "movies_predict": "/movies/models/movies/predict-rating",
}