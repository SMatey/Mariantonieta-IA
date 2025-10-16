"""
API de Propiedades - Predicción de Precios usando Random Forest
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, Optional

# Agregar el directorio padre al path para importar constants
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from notebooks import constants as const

app = FastAPI(title="Properties Price Prediction API", version="1.0.0")

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

# Variable global para el modelo
_loaded_model_data = None

def load_properties_model():
    """Carga el modelo Random Forest de propiedades"""
    global _loaded_model_data
    if _loaded_model_data is not None:
        return _loaded_model_data
    
    try:
        model_path = os.path.join(const.BASE_DIR, 'models', 'random_forest_properties.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Crear estructura de datos del modelo
        _loaded_model_data = {
            'model': model,
            'feature_columns': [
                'bathroomcnt', 'bedroomcnt', 'finishedsquarefeet', 'latitude', 
                'longitude', 'lotsizesquarefeet', 'yearbuilt', 'taxamount',
                'assessmentyear', 'landtaxvaluedollarcnt', 'structuretaxvaluedollarcnt',
                'censustractandblock'
            ],
            'model_info': {
                'type': 'Random Forest Regressor',
                'features_count': 12
            }
        }
        return _loaded_model_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cargando modelo: {str(e)}")

def create_features_from_property_data(request: PropertyPredictionRequest):
    """
    Crea las características para el modelo a partir de los datos de la propiedad
    """
    # Valores por defecto basados en estadísticas típicas
    defaults = {
        'bathroomcnt': 2.0,
        'bedroomcnt': 3.0,
        'finishedsquarefeet': 1500.0,
        'latitude': 34.0,  # Aproximado para California
        'longitude': -118.0,
        'lotsizesquarefeet': 7000.0,
        'yearbuilt': 1980.0,
        'taxamount': 5000.0,
        'assessmentyear': 2017.0,
        'landtaxvaluedollarcnt': 200000.0,
        'structuretaxvaluedollarcnt': 300000.0,
        'censustractandblock': 6037000000.0
    }
    
    # Usar valores del usuario o defaults
    features_dict = {}
    for feature in defaults.keys():
        user_value = getattr(request, feature, None)
        features_dict[feature] = user_value if user_value is not None else defaults[feature]
    
    return features_dict

def make_property_prediction(features_dict):
    """Hace predicción de precio de propiedad"""
    model_data = load_properties_model()
    model = model_data['model']
    feature_columns = model_data['feature_columns']
    
    # Crear DataFrame con las características
    features_df = pd.DataFrame([features_dict])[feature_columns]
    
    # Hacer predicción
    prediction = model.predict(features_df)[0]
    
    # Calcular confianza basada en las características de entrada
    # Una confianza mayor si más características fueron proporcionadas por el usuario
    user_provided_features = sum(1 for key in features_dict.keys() 
                                if getattr(PropertyPredictionRequest.parse_obj({'query': ''}), key, None) is not None)
    confidence = min(95, 60 + (user_provided_features * 3))
    
    return prediction, confidence

@app.get("/")
def root():
    return {"message": "Properties Price Prediction API", "version": "1.0.0"}

@app.post("/models/properties/predict", response_model=PropertyPredictionResponse)
def predict_property_price(request: PropertyPredictionRequest):
    """
    Predice el precio de una propiedad usando Random Forest
    """
    try:
        # 1. Crear características
        features_dict = create_features_from_property_data(request)
        
        # 2. Hacer predicción
        prediction, confidence = make_property_prediction(features_dict)
        
        # 3. Información del modelo
        model_data = load_properties_model()
        
        # 4. Determinar características proporcionadas
        user_features = []
        for feature in features_dict.keys():
            if getattr(request, feature, None) is not None:
                user_features.append(f"{feature}: {features_dict[feature]}")
        
        # 5. Crear interpretación
        price_millions = prediction / 1_000_000
        
        if price_millions > 2:
            category = "propiedad de lujo"
            emoji = "🏰"
        elif price_millions > 1:
            category = "propiedad premium"
            emoji = "🏡"
        elif price_millions > 0.5:
            category = "propiedad de precio medio"
            emoji = "🏠"
        else:
            category = "propiedad económica"
            emoji = "🏘️"
        
        # Factores importantes
        factors = []
        if features_dict['finishedsquarefeet'] > 2000:
            factors.append("casa grande")
        if features_dict['bedroomcnt'] >= 4:
            factors.append("muchas habitaciones")
        if features_dict['yearbuilt'] > 2000:
            factors.append("construcción reciente")
        
        factors_text = ", ".join(factors) if factors else "características estándar"
        
        interpretation = (
            f"{emoji} PRECIO ESTIMADO: ${prediction:,.2f} USD\n"
            f"Categoría: {category}\n"
            f"Factores clave: {factors_text}\n"
            f"Tamaño: {features_dict['finishedsquarefeet']:,.0f} sq ft\n"
            f"Habitaciones: {features_dict['bedroomcnt']:.0f} bed, {features_dict['bathroomcnt']:.0f} bath\n"
            f"Año construcción: {features_dict['yearbuilt']:.0f}\n"
            f"Confianza: {confidence:.1f}%\n"
            f"Características evaluadas: {len(features_dict)}"
        )
        
        if user_features:
            interpretation += f"\nDatos proporcionados: {', '.join(user_features[:3])}{'...' if len(user_features) > 3 else ''}"
        
        return PropertyPredictionResponse(
            prediction=round(prediction, 2),
            confidence=round(confidence, 1),
            model_info={
                "model_type": "Random Forest Regressor",
                "features_used": len(features_dict),
                "user_provided_features": len(user_features),
                "prediction_range": f"${prediction*0.8:,.0f} - ${prediction*1.2:,.0f}"
            },
            interpretation=interpretation
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/health")
def health():
    try:
        model_data = load_properties_model()
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_type": model_data['model_info']['type'],
            "features_count": model_data['model_info']['features_count']
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("🏠 Properties Price Prediction API")
    uvicorn.run(app, host="0.0.0.0", port=8001)