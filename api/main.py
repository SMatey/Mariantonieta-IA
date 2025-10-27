from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import sys
import os
from typing import Dict, Any
from api.models.main_models import QueryRequest, QueryResponse, HealthResponse

from . import constants as const
from .config_logger import get_main_logger, configure_fastapi_logging, disable_console_logging

# Configurar logging - SOLO A ARCHIVOS, NO CONSOLA
logger = get_main_logger(console_output=False)
configure_fastapi_logging()
disable_console_logging()

# Agregar el directorio padre al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.coordinator import interpretar_y_ejecutar

# Importar las APIs de los modelos
try:
    from .routes.bitcoin_api import app as bitcoin_app, load_bitcoin_model
    BITCOIN_AVAILABLE = True
    logger.info("Bitcoin API importada exitosamente")
except ImportError as e:
    logger.warning(f"Bitcoin API no disponible: {e}")
    BITCOIN_AVAILABLE = False

try:
    from .routes.movies_api import app as movies_app, load_movies_model_and_data
    MOVIES_AVAILABLE = True
    logger.info("Movies API importada exitosamente")
except ImportError as e:
    logger.warning(f"Movies API no disponible: {e}")
    MOVIES_AVAILABLE = False

try:
    from .routes.flights_api import app as flights_app, load_flights_model
    FLIGHTS_AVAILABLE = True
    logger.info("Flights API importada exitosamente")
except ImportError as e:
    logger.warning(f"Flights API no disponible: {e}")
    FLIGHTS_AVAILABLE = False

try:
    from .routes.acv_api import app as acv_app, load_acv_model
    ACV_AVAILABLE = True
    logger.info("ACV API importada exitosamente")
except ImportError as e:
    logger.warning(f"ACV API no disponible: {e}")
    ACV_AVAILABLE = False

try:
    from .routes.avocado_api import app as avocado_app, load_avocado_model
    AVOCADO_AVAILABLE = True
    logger.info("Avocado API importada exitosamente")
except ImportError as e:
    logger.warning(f"Avocado API no disponible: {e}")
    AVOCADO_AVAILABLE = False

try:
    from api.routes import face_routes
    from face_recognition.azure_face_service import face_client 
    FACE_AVAILABLE = True
except Exception as e:
    print(f"Face API no disponible: {e}.")
    FACE_AVAILABLE = False

try:
    from .stt import router as stt_router
    STT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  STT Router no disponible: {e}")
    STT_AVAILABLE = False

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

# Montar las sub-aplicaciones de los modelos
if BITCOIN_AVAILABLE:
    app.mount("/bitcoin", bitcoin_app)

if MOVIES_AVAILABLE:
    app.mount("/movies", movies_app)

if FLIGHTS_AVAILABLE:
    app.mount("/flights", flights_app)

if ACV_AVAILABLE:
    app.mount("/acv", acv_app)

if AVOCADO_AVAILABLE:
    app.mount("/avocado", avocado_app)

if FACE_AVAILABLE:
    app.include_router(face_routes.router)

if STT_AVAILABLE:
    app.include_router(stt_router)

@app.get("/", response_model=HealthResponse)
def root():
    """Endpoint principal con información de la API"""
    available_models = ["coordinator"]
    if BITCOIN_AVAILABLE:
        available_models.append("bitcoin")
    if MOVIES_AVAILABLE:
        available_models.append("movies")
    if FLIGHTS_AVAILABLE:
        available_models.append("flights")
    if ACV_AVAILABLE:
        available_models.append("acv")
    if AVOCADO_AVAILABLE:
        available_models.append("avocado")
    if FACE_AVAILABLE:
        available_models.append("face")
    
    return HealthResponse(
        status="active",
        available_models=available_models,
        message="Bienvenido al AI Models API Hub"
    )

@app.post("/ask", response_model=QueryResponse)
def ask_user(request: QueryRequest):
    """Endpoint original del coordinador LLM"""
    import time
    start_time = time.time()
    
    try:
        query = request.query
        logger.info(f"Consulta recibida: {query}")
        
        respuesta = interpretar_y_ejecutar(query)
        
        execution_time = time.time() - start_time
        logger.info(f"Consulta procesada exitosamente en {execution_time:.3f}s")
        
        return QueryResponse(respuesta=respuesta)
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Error procesando consulta en {execution_time:.3f}s: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en coordinador: {str(e)}")

@app.get("/health")
def health_check():
    """Verifica el estado de todos los servicios"""
    logger.info("Iniciando health check de todos los servicios")
    services_status = {}
    
    # Verificar coordinador LLM
    try:
        # Test básico del coordinador
        test_response = interpretar_y_ejecutar("test")
        services_status["coordinator"] = {
            "status": "healthy",
            "description": "LLM Coordinator activo"
        }
        logger.info("Coordinador LLM: healthy")
    except Exception as e:
        services_status["coordinator"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        logger.error(f"Coordinador LLM: unhealthy - {str(e)}")
    
    # Verificar Bitcoin API si está disponible
    if BITCOIN_AVAILABLE:
        try:
            bitcoin_model = load_bitcoin_model()
            services_status["bitcoin"] = {
                "status": "healthy",
                "model_r2": bitcoin_model['training_info']['r2_score'],
                "description": "Bitcoin Price Prediction Model"
            }
            logger.info("Bitcoin API: healthy")
        except Exception as e:
            services_status["bitcoin"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            logger.error(f"Bitcoin API: unhealthy - {str(e)}")
    
    
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
            logger.info("Movies API: healthy")
        except Exception as e:
            services_status["movies"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            logger.error(f"Movies API: unhealthy - {str(e)}")
    
    # Verificar Flights API si está disponible
    if FLIGHTS_AVAILABLE:
        try:
            flights_model = load_flights_model()
            services_status["flights"] = {
                "status": "healthy",
                "model_type": flights_model['model_info']['type'],
                "description": "Flight Delay Prediction Model"
            }
            logger.info("Flights API: healthy")
        except Exception as e:
            services_status["flights"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            logger.error(f"Flights API: unhealthy - {str(e)}")
    
    # Verificar ACV API si está disponible
    if ACV_AVAILABLE:
        try:
            acv_model = load_acv_model()
            services_status["acv"] = {
                "status": "healthy",
                "model_type": acv_model['model_info']['type'],
                "accuracy": acv_model['model_info']['accuracy'],
                "description": "Stroke Risk Prediction Model"
            }
            logger.info("ACV API: healthy")
        except Exception as e:
            services_status["acv"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            logger.error(f"ACV API: unhealthy - {str(e)}")
    
    # Verificar Avocado API si está disponible
    if AVOCADO_AVAILABLE:
        try:
            avocado_model = load_avocado_model()
            services_status["avocado"] = {
                "status": "healthy",
                "model_type": avocado_model['model_info']['type'],
                "features_count": avocado_model['model_info']['features_count'],
                "description": "Avocado Price Prediction Model"
            }
            logger.info("Avocado API: healthy")
        except Exception as e:
            services_status["avocado"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            logger.error(f"Avocado API: unhealthy - {str(e)}")

    if FACE_AVAILABLE:
        try:
            # Check Azure
            if not (face_client and face_client.endpoint):
                raise Exception("Azure FaceClient no inicializado.")
            
            # (No podemos chequear Google Vision fácilmente sin credenciales,
            # así que confiamos en que la importación fue exitosa)
                
            services_status["face"] = {
               "status": "healthy",
               "description": "Híbrido: Detección Azure + Emoción Google Vision",
               "azure_endpoint": face_client.endpoint
            }
        except Exception as e:
            services_status["face"] = {
                "status": "unhealthy",
                "error": str(e)
            }
    
    overall_status = "healthy" if all(
        service["status"] == "healthy" 
        for service in services_status.values()
    ) else "partial"
    
    logger.info(f"Health check completado - Estado general: {overall_status}")
    
    return {
        "overall_status": overall_status,
        "services": services_status,
        "bitcoin_available": BITCOIN_AVAILABLE,
        "movies_available": MOVIES_AVAILABLE,
        "flights_available": FLIGHTS_AVAILABLE,
        "acv_available": ACV_AVAILABLE,
        "avocado_available": AVOCADO_AVAILABLE,
        "face_available": FACE_AVAILABLE
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
    
    if FLIGHTS_AVAILABLE:
        models.append({
            "name": "flights",
            "description": "Predicción de retrasos de vuelos usando Random Forest",
            "endpoint": "/flights/predict",
            "type": "Random Forest",
            "status": "active"
        })
    
    if ACV_AVAILABLE:
        models.append({
            "name": "acv",
            "description": "Predicción de riesgo de accidente cerebrovascular usando Árbol de Decisión",
            "endpoint": "/acv/predict",
            "type": "Decision Tree",
            "status": "active"
        })
    
    if AVOCADO_AVAILABLE:
        models.append({
            "name": "avocado",
            "description": "Predicción de precios de aguacate usando CatBoost",
            "endpoint": "/avocado/predict",
            "type": "CatBoost Regressor",
            "status": "active"
        })
    
    if FACE_AVAILABLE:
        models.append({
            "name": "face_emotion",
            "description": "Detección (Azure) + Emoción (Google Vision)",
            "endpoint": "/face/analyze-azure-google", # <-- Coincide con face_routes.py
            "type": "Híbrido (Azure + Google Cloud Vision)",
            "status": "active"
        })
    
    return {"available_models": models}

if __name__ == "__main__":
    logger.info("Iniciando AI Models API Hub...")
    logger.info("Modelos disponibles:")
    logger.info("   • LLM Coordinator")
    if BITCOIN_AVAILABLE:
        logger.info("Bitcoin Price Prediction (Random Forest)")
    if MOVIES_AVAILABLE:
        logger.info("Movies Recommendation System (KNN)")
    if FLIGHTS_AVAILABLE:
        logger.info("Flight Delay Prediction (Random Forest)")
    if ACV_AVAILABLE:
        logger.info("ACV Risk Prediction (Decision Tree)")
    if AVOCADO_AVAILABLE:
        logger.info("Avocado Price Prediction (CatBoost)")
    logger.info("Documentación disponible en: http://localhost:8000/docs")
    if FACE_AVAILABLE:
        logger.info("Face Recognition (Azure + Google Vision)")

    logger.info("Documentación disponible en: http://localhost:8000/docs")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
