from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import sys
import os
from typing import Dict, Any

# Agregar el directorio padre al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.coordinator import interpretar_y_ejecutar

# Importar las APIs de los modelos
try:
    from api.routes.bitcoin_api import app as bitcoin_app, load_bitcoin_model
    BITCOIN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Bitcoin API no disponible: {e}")
    BITCOIN_AVAILABLE = False

try:
    from api.routes.properties_api import app as properties_app, load_properties_model
    PROPERTIES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Properties API no disponible: {e}")
    PROPERTIES_AVAILABLE = False

try:
    from api.routes.movies_api import app as movies_app, load_movies_model_and_data
    MOVIES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Movies API no disponible: {e}")
    MOVIES_AVAILABLE = False

app = FastAPI(
    title="AI Models API Hub",
    description="API centralizada para múltiples modelos de Machine Learning",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    respuesta: str

class HealthResponse(BaseModel):
    status: str
    available_models: list
    message: str

# Montar las sub-aplicaciones de los modelos
if BITCOIN_AVAILABLE:
    app.mount("/bitcoin", bitcoin_app)

if PROPERTIES_AVAILABLE:
    app.mount("/properties", properties_app)

if MOVIES_AVAILABLE:
    app.mount("/movies", movies_app)

@app.get("/", response_model=HealthResponse)
def root():
    """Endpoint principal con información de la API"""
    available_models = ["coordinator"]
    if BITCOIN_AVAILABLE:
        available_models.append("bitcoin")
    if PROPERTIES_AVAILABLE:
        available_models.append("properties")
    if MOVIES_AVAILABLE:
        available_models.append("movies")
    
    return HealthResponse(
        status="active",
        available_models=available_models,
        message="Bienvenido al AI Models API Hub"
    )

@app.post("/ask", response_model=QueryResponse)
def ask_user(request: QueryRequest):
    """Endpoint original del coordinador LLM"""
    try:
        query = request.query
        respuesta = interpretar_y_ejecutar(query)
        return QueryResponse(respuesta=respuesta)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en coordinador: {str(e)}")

@app.get("/health")
def health_check():
    """Verifica el estado de todos los servicios"""
    services_status = {}
    
    # Verificar coordinador LLM
    try:
        # Test básico del coordinador
        test_response = interpretar_y_ejecutar("test")
        services_status["coordinator"] = {
            "status": "healthy",
            "description": "LLM Coordinator activo"
        }
    except Exception as e:
        services_status["coordinator"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # Verificar Bitcoin API si está disponible
    if BITCOIN_AVAILABLE:
        try:
            bitcoin_model = load_bitcoin_model()
            services_status["bitcoin"] = {
                "status": "healthy",
                "model_r2": bitcoin_model['training_info']['r2_score'],
                "description": "Bitcoin Price Prediction Model"
            }
        except Exception as e:
            services_status["bitcoin"] = {
                "status": "unhealthy",
                "error": str(e)
            }
    
    # Verificar Properties API si está disponible
    if PROPERTIES_AVAILABLE:
        try:
            properties_model = load_properties_model()
            services_status["properties"] = {
                "status": "healthy",
                "model_type": properties_model['model_info']['type'],
                "description": "Properties Price Prediction Model"
            }
        except Exception as e:
            services_status["properties"] = {
                "status": "unhealthy",
                "error": str(e)
            }
    
    # Verificar Movies API si está disponible
    if MOVIES_AVAILABLE:
        try:
            movies_model, movies_df, ratings_df = load_movies_model_and_data()
            services_status["movies"] = {
                "status": "healthy",
                "model_type": movies_model['model_info']['type'],
                "movies_count": len(movies_df),
                "description": "Movies Recommendation System"
            }
        except Exception as e:
            services_status["movies"] = {
                "status": "unhealthy",
                "error": str(e)
            }
    
    overall_status = "healthy" if all(
        service["status"] == "healthy" 
        for service in services_status.values()
    ) else "partial"
    
    return {
        "overall_status": overall_status,
        "services": services_status,
        "bitcoin_available": BITCOIN_AVAILABLE,
        "properties_available": PROPERTIES_AVAILABLE,
        "movies_available": MOVIES_AVAILABLE
    }

@app.get("/models")
def list_models():
    """Lista todos los modelos disponibles"""
    models = [
        {
            "name": "coordinator",
            "description": "Coordinador LLM que decide qué modelo usar",
            "endpoint": "/ask",
            "type": "llm_coordinator",
            "status": "active"
        }
    ]
    
    if BITCOIN_AVAILABLE:
        models.append({
            "name": "bitcoin",
            "description": "Predicción de precios de Bitcoin usando Random Forest",
            "endpoint": "/bitcoin/models/bitcoin/predict",
            "type": "Random Forest",
            "status": "active"
        })
    
    if PROPERTIES_AVAILABLE:
        models.append({
            "name": "properties",
            "description": "Predicción de precios de propiedades usando Random Forest",
            "endpoint": "/properties/models/properties/predict",
            "type": "Random Forest",
            "status": "active"
        })
    
    if MOVIES_AVAILABLE:
        models.append({
            "name": "movies",
            "description": "Sistema de recomendación de películas usando KNN",
            "endpoints": [
                "/movies/models/movies/recommend",
                "/movies/models/movies/predict-rating"
            ],
            "type": "K-Nearest Neighbors",
            "status": "active"
        })
    
    return {"available_models": models}

if __name__ == "__main__":
    print("Iniciando AI Models API Hub...")
    print("Modelos disponibles:")
    print("   • LLM Coordinator")
    if BITCOIN_AVAILABLE:
        print("   • Bitcoin Price Prediction (Random Forest)")
    if PROPERTIES_AVAILABLE:
        print("   • Properties Price Prediction (Random Forest)")
    if MOVIES_AVAILABLE:
        print("   • Movies Recommendation System (KNN)")
    print("Documentación disponible en: http://localhost:8000/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
