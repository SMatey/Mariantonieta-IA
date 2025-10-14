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

# Importar la API de Bitcoin
try:
    from api.routes.bitcoin_api import app as bitcoin_app, load_bitcoin_model
    BITCOIN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Bitcoin API no disponible: {e}")
    BITCOIN_AVAILABLE = False

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

# Montar la sub-aplicación de Bitcoin si está disponible
if BITCOIN_AVAILABLE:
    app.mount("/bitcoin", bitcoin_app) # Si quiere acceder al documentation de bitcoin en /bitcoin/docs

@app.get("/", response_model=HealthResponse)
def root():
    """Endpoint principal con información de la API"""
    available_models = ["coordinator"]
    if BITCOIN_AVAILABLE:
        available_models.append("bitcoin")
    
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
    
    overall_status = "healthy" if all(
        service["status"] == "healthy" 
        for service in services_status.values()
    ) else "partial"
    
    return {
        "overall_status": overall_status,
        "services": services_status,
        "bitcoin_available": BITCOIN_AVAILABLE
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
            "type": "Bosque aleatorio",
            "status": "active"
        })
    
    return {"available_models": models}

if __name__ == "__main__":
    print("Iniciando AI Models API Hub...")
    print("Modelos disponibles:")
    print("   • LLM Coordinator")
    if BITCOIN_AVAILABLE:
        print("   • Bitcoin Price Prediction (Random Forest)")
    print("Documentación disponible en: http://localhost:8000/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
