# Configuración de modelos disponibles
MODELS_CONFIG = {
    "bitcoin": {
        "endpoint": "http://localhost:8000/bitcoin/models/bitcoin/predict",
        "description": "Para predicciones de precios de Bitcoin usando series de tiempo (Prophet), análisis temporal y tendencias futuras",
        "available": True,
        "response_type": "time_series_prediction"
    },
    "properties": {
        "endpoint": "http://localhost:8000/properties/models/properties/predict",
        "description": "Para predicción de precios de propiedades inmobiliarias, casas, apartamentos",
        "available": True,
        "response_type": "prediction"
    },
    "movies": {
        "endpoint": "http://localhost:8000/movies/models/movies/recommend",
        "description": "Para recomendaciones de películas personalizadas basadas en preferencias",
        "available": True,
        "response_type": "recommendation"
    },
    "flights": {
        "endpoint": "http://localhost:8000/flights/models/flights/predict",
        "description": "Para predicciones de retrasos de vuelos, análisis de puntualidad y planificación de viajes",
        "available": True,
        "response_type": "flight_prediction"
    },
    "acv": {
        "endpoint": "http://localhost:8000/acv/predict",
        "description": "Para evaluación de riesgo de accidente cerebrovascular (ACV/stroke) basado en características médicas y demográficas",
        "available": True,
        "response_type": "medical_classification"
    },
    "wine": {
        "endpoint": "http://localhost:8000/wine/classify",
        "description": "Para clasificación de vinos basada en características químicas",
        "available": False,
        "response_type": "classification"
    },
    "churn": {
        "endpoint": "http://localhost:8000/churn/predict",
        "description": "Para predicción de abandono de clientes",
        "available": False,
        "response_type": "prediction"
    },
    "emotions": {
        "endpoint": "http://localhost:8000/emotions/analyze",
        "description": "Para análisis de emociones en texto",
        "available": False,
        "response_type": "classification"
    }
}