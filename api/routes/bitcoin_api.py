"""
API de Bitcoin - SOLO DATOS REALES, NO SAMPLES
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib  # Cambiar pickle por joblib
import pandas as pd
import os
import sys
import random
from datetime import datetime
from typing import Dict, Optional
import requests
from ..models.bitcoin_api_models import PredictionRequest, PredictionResponse

# Importar constantes desde el paquete api
from .. import constants as const

app = FastAPI(title="Bitcoin Real Data Prediction API", version="1.0.0")



# Variable global para el modelo
_loaded_model_data = None

def load_bitcoin_model():
    """Carga el modelo Prophet entrenado"""
    global _loaded_model_data
    if _loaded_model_data is not None:
        return _loaded_model_data
    
    try:
        model_path = os.path.join(const.BASE_DIR, 'ml_models', 'prophet_bitcoin_v2_2025-10-24.pkl')
        _loaded_model_data = joblib.load(model_path)  # Usar joblib.load en lugar de pickle.load
        return _loaded_model_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cargando modelo Prophet: {str(e)}")

def make_prophet_prediction(model, dates):
    """Genera predicciones usando el modelo Prophet"""

    # Crear DataFrame con las fechas proporcionadas
    future = pd.DataFrame({'ds': pd.to_datetime(dates)})
    forecast = model.predict(future)

    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


@app.get("/")
def root():
    return {"message": "Bitcoin REAL Data Prediction API", "version": "3.0.0", "note": "Uses real market data"}

@app.post("/models/bitcoin/predict", response_model=dict)
def predict_bitcoin_price(request: PredictionRequest):
    """
    Predice precios de Bitcoin usando modelo Prophet v2
    """
    try:
        model = load_bitcoin_model()
        forecast = make_prophet_prediction(model, request.dates)

        results = [
            {
                "date": row.ds.strftime("%Y-%m-%d"),
                "predicted_price": round(row.yhat, 2),
                "lower_bound": round(row.yhat_lower, 2),
                "upper_bound": round(row.yhat_upper, 2)
            }
            for _, row in forecast.iterrows()
        ]

        return {
            "model_version": "Prophet v2",
            "dates_predicted": len(results),
            "predictions": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")

@app.get("/health")
def health():
    try:
        model_data = load_bitcoin_model()
        return {
            "status": "healthy",
            "model_loaded": True,
            "real_data_available": True,
            "current_btc_price": model_data['current_price'],
            "data_source": model_data['source']
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Bitcoin REAL Data API - NO SAMPLES, ONLY REAL DATA")
    uvicorn.run(app, host="0.0.0.0", port=8000)