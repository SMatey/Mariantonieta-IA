from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any

# Agregar el directorio padre al path para importar constants
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from notebooks import constants as const

app = FastAPI(title="Bitcoin Price Prediction API", version="1.0.0")

class PredictionRequest(BaseModel):
    query: str

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float
    model_info: Dict[str, Any]
    interpretation: str

# Variable global para el modelo cargado
loaded_model_data = None

def load_bitcoin_model():
    """Carga el modelo Random Forest para predicción de Bitcoin"""
    global loaded_model_data
    
    if loaded_model_data is not None:
        return loaded_model_data
    
    try:
        # Cargar el modelo más reciente
        model_path = os.path.join(const.BASE_DIR, 'models', 'bitcoin_random_forest_latest.pkl')
        
        with open(model_path, 'rb') as f:
            loaded_model_data = pickle.load(f)
        
        print(f"✅ Modelo Bitcoin cargado desde: {model_path}")
        return loaded_model_data
        
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"Modelo no encontrado en: {model_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al cargar el modelo: {str(e)}")

def create_sample_features():
    """
    Crea características de ejemplo para demostración.
    En un caso real, estas vendrían de datos en tiempo real.
    """
    # Valores típicos basados en el entrenamiento (aproximados)
    sample_data = {
        'Open': 2500.0,
        'High': 2600.0,
        'Low': 2400.0,
        'Volume': 1000000000.0,  # 1B
        'Market Cap': 45000000000,  # 45B
        'Price_Range': 200.0,
        'Price_Change': 50.0,
        'Price_Change_Pct': 2.0,
        'MA_5': 2450.0,
        'MA_10': 2400.0,
        'MA_20': 2350.0,
        'Volatility_5': 50.0,
        'Volatility_10': 75.0,
        'Close_lag_1': 2450.0,
        'Volume_lag_1': 950000000.0,
        'Close_lag_2': 2400.0,
        'Volume_lag_2': 900000000.0,
        'Close_lag_3': 2380.0,
        'Volume_lag_3': 850000000.0,
        'Close_lag_5': 2300.0,
        'Volume_lag_5': 800000000.0,
        'Day_of_Week': 1,  # Lunes
        'Month': 10,       # Octubre
        'Quarter': 4,      # Q4
        'RSI_14': 55.0,
        'BB_Middle': 2400.0,
        'BB_Std': 100.0,
        'BB_Upper': 2600.0,
        'BB_Lower': 2200.0,
        'BB_Position': 0.6
    }
    
    return sample_data

def make_prediction(features_dict: Dict[str, float]):
    """Hace una predicción usando el modelo cargado"""
    model_data = load_bitcoin_model()
    model = model_data['model']
    feature_columns = model_data['feature_columns']
    
    # Crear DataFrame con las características en el orden correcto
    features_df = pd.DataFrame([features_dict])[feature_columns]
    
    # Hacer predicción
    prediction = model.predict(features_df)[0]
    
    # Calcular intervalo de confianza aproximado basado en RMSE
    rmse = model_data['training_info']['rmse']
    confidence = max(0, min(100, (1 - (rmse / prediction)) * 100))
    
    return prediction, confidence

@app.get("/")
def root():
    """Endpoint de bienvenida"""
    return {"message": "Bitcoin Price Prediction API", "status": "active"}

@app.get("/model/info")
def get_model_info():
    """Obtiene información del modelo cargado"""
    model_data = load_bitcoin_model()
    return {
        "model_type": "Random Forest Regressor",
        "features_count": model_data['training_info']['total_features'],
        "training_date": model_data['training_info']['training_date'],
        "r2_score": model_data['training_info']['r2_score'],
        "rmse": model_data['training_info']['rmse'],
        "mae": model_data['training_info']['mae'],
        "top_features": model_data['feature_importance'].head(5).to_dict('records')
    }

@app.post("/models/bitcoin/predict", response_model=PredictionResponse)
def predict_bitcoin_price(request: PredictionRequest):
    """
    Predice el precio del Bitcoin basado en el query del usuario.
    Por ahora usa datos de ejemplo, pero puede expandirse para parsear el query.
    """
    try:
        # Para esta demo, usamos características de ejemplo
        # En un sistema real, parsearias el query para extraer datos o usarías datos en tiempo real
        sample_features = create_sample_features()
        
        # Hacer predicción
        prediction, confidence = make_prediction(sample_features)
        
        # Obtener información del modelo
        model_data = load_bitcoin_model()
        
        # Crear interpretación basada en la predicción
        current_price = sample_features['Open']  # Precio "actual" de ejemplo
        change = prediction - current_price
        change_pct = (change / current_price) * 100
        
        if change > 0:
            trend = "al alza"
            emoji = "📈"
        elif change < 0:
            trend = "a la baja" 
            emoji = "📉"
        else:
            trend = "estable"
            emoji = "➡️"
        
        interpretation = (
            f"{emoji} Predicción: ${prediction:,.2f} USD\n"
            f"Cambio esperado: ${change:+,.2f} ({change_pct:+.1f}%)\n"
            f"Tendencia: {trend}\n"
            f"Confianza del modelo: {confidence:.1f}%"
        )
        
        return PredictionResponse(
            prediction=round(prediction, 2),
            confidence=round(confidence, 1),
            model_info={
                "model_type": "Random Forest",
                "r2_score": model_data['training_info']['r2_score'],
                "rmse": model_data['training_info']['rmse'],
                "features_used": len(model_data['feature_columns'])
            },
            interpretation=interpretation
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.get("/health")
def health_check():
    """Endpoint de salud del servicio"""
    try:
        model_data = load_bitcoin_model()
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_r2": model_data['training_info']['r2_score']
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    # Cargar el modelo al iniciar
    load_bitcoin_model()
    print("🚀 Iniciando Bitcoin Prediction API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)