"""
API de Propiedades - Predicción de Precios usando Random Forest
"""
from fastapi import FastAPI, HTTPException
import pickle
import pandas as pd
import numpy as np
import os
from ..models.properties_api_models import PropertyPredictionRequest, PropertyPredictionResponse

# Importar constantes desde el paquete api
from .. import constants as const

app = FastAPI(title="Properties Price Prediction API", version="1.0.0")

# Variable global para el modelo
_loaded_model_data = None

def load_properties_model():
    """Carga el modelo Random Forest de propiedades (modelo ya entrenado, NO Pipeline)"""
    global _loaded_model_data
    if _loaded_model_data is not None:
        return _loaded_model_data
    
    try:
        model_path = os.path.join(const.BASE_DIR, 'ml_models', 'random_forest_properties.pkl')
        with open(model_path, 'rb') as f:
            # El modelo guardado es solo el RandomForestRegressor, no un Pipeline
            model = pickle.load(f)
        
        # Estas son todas las características que el modelo espera (del modelo.feature_names_in_)
        expected_columns = [
            'airconditioningtypeid', 'architecturalstyletypeid', 'basementsqft',
            'bathroomcnt', 'bedroomcnt', 'buildingclasstypeid', 'buildingqualitytypeid',
            'calculatedbathnbr', 'finishedfloor1squarefeet',
            'calculatedfinishedsquarefeet', 'finishedsquarefeet12',
            'finishedsquarefeet13', 'finishedsquarefeet15', 'finishedsquarefeet50',
            'finishedsquarefeet6', 'fips', 'fullbathcnt', 'garagecarcnt',
            'garagetotalsqft', 'heatingorsystemtypeid', 'latitude', 'longitude',
            'lotsizesquarefeet', 'poolcnt', 'propertylandusetypeid',
            'rawcensustractandblock', 'regionidcity', 'regionidcounty',
            'regionidneighborhood', 'regionidzip', 'roomcnt', 'threequarterbathnbr',
            'typeconstructiontypeid', 'unitcnt', 'yardbuildingsqft17',
            'yardbuildingsqft26', 'yearbuilt', 'numberofstories',
            'structuretaxvaluedollarcnt', 'assessmentyear', 'landtaxvaluedollarcnt',
            'taxamount', 'taxdelinquencyflag', 'taxdelinquencyyear',
            'censustractandblock'
        ]
        
        _loaded_model_data = {
            'model': model,  # RandomForestRegressor directo
            'expected_columns': expected_columns,
            'model_info': {
                'type': 'Random Forest Regressor',
                'features_count': len(expected_columns),
                'preprocessing': 'Modelo entrenado con todas las características del dataset'
            }
        }
        print("✅ Modelo Random Forest de propiedades cargado exitosamente")
        return _loaded_model_data
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        raise HTTPException(status_code=500, detail=f"Error cargando modelo: {str(e)}")

def create_features_from_property_data(request: PropertyPredictionRequest):
    """
    Crea todas las 45 características que el modelo espera desde el request.
    Usa valores por defecto razonables para características no proporcionadas.
    """
    model_data = load_properties_model()
    expected_columns = model_data["expected_columns"]
    
    # Características básicas que pueden venir del request
    bedrooms = request.bedrooms if hasattr(request, 'bedrooms') and request.bedrooms else 3.0
    bathrooms = request.bathrooms if hasattr(request, 'bathrooms') and request.bathrooms else 2.0
    square_feet = request.square_feet if hasattr(request, 'square_feet') and request.square_feet else 1500.0
    latitude = request.latitude if hasattr(request, 'latitude') and request.latitude else 34.0522
    longitude = request.longitude if hasattr(request, 'longitude') and request.longitude else -118.2437
    lot_size = request.lot_size if hasattr(request, 'lot_size') and request.lot_size else 5000.0
    year_built = request.year_built if hasattr(request, 'year_built') and request.year_built else 1990
    tax_amount = request.tax_amount if hasattr(request, 'tax_amount') and request.tax_amount else 5000.0
    land_value = request.land_value if hasattr(request, 'land_value') and request.land_value else 200000.0
    structure_value = request.structure_value if hasattr(request, 'structure_value') and request.structure_value else 300000.0
    
    # Crear el diccionario en el orden EXACTO que espera el modelo
    features_dict = {
        'airconditioningtypeid': 0.0,
        'architecturalstyletypeid': 0.0,
        'basementsqft': 0.0,
        'bathroomcnt': bathrooms,
        'bedroomcnt': bedrooms,
        'buildingclasstypeid': 0.0,
        'buildingqualitytypeid': 0.0,
        'calculatedbathnbr': bathrooms,
        'finishedfloor1squarefeet': 0.0,
        'calculatedfinishedsquarefeet': square_feet,
        'finishedsquarefeet12': 0.0,
        'finishedsquarefeet13': 0.0,
        'finishedsquarefeet15': 0.0,
        'finishedsquarefeet50': 0.0,
        'finishedsquarefeet6': 0.0,
        'fips': 6037.0,  # Los Angeles County
        'fullbathcnt': max(1, int(bathrooms)),
        'garagecarcnt': 0.0,
        'garagetotalsqft': 0.0,
        'heatingorsystemtypeid': 0.0,
        'latitude': latitude,
        'longitude': longitude,
        'lotsizesquarefeet': lot_size,
        'poolcnt': 0.0,
        'propertylandusetypeid': 0.0,
        'rawcensustractandblock': 0.0,
        'regionidcity': 0.0,
        'regionidcounty': 0.0,
        'regionidneighborhood': 0.0,
        'regionidzip': 0.0,
        'roomcnt': bedrooms + 1,  # Habitaciones + sala
        'threequarterbathnbr': 0.0,
        'typeconstructiontypeid': 0.0,
        'unitcnt': 1.0,
        'yardbuildingsqft17': 0.0,
        'yardbuildingsqft26': 0.0,
        'yearbuilt': year_built,
        'numberofstories': 0.0,
        'structuretaxvaluedollarcnt': structure_value,
        'assessmentyear': 2016,
        'landtaxvaluedollarcnt': land_value,
        'taxamount': tax_amount,
        'taxdelinquencyflag': 0.0,
        'taxdelinquencyyear': 0.0,
        'censustractandblock': 6037.0
    }
    
    print(f"🔍 Features creadas ({len(features_dict)}): {list(features_dict.keys())}")
    return features_dict

@app.get("/")
def root():
    return {"message": "Properties Price Prediction API", "version": "1.0.0"}

@app.post("/predict", response_model=PropertyPredictionResponse)
def predict_property_price(request: PropertyPredictionRequest):
    """Predice el precio de una propiedad usando el modelo Random Forest entrenado"""
    try:
        # Cargar modelo
        model_data = load_properties_model()
        model = model_data['model']  # RandomForestRegressor directo
        expected_columns = model_data['expected_columns']
        
        # Crear DataFrame de entrada
        feature_data = create_features_from_property_data(request)
        
        # Verificar que tenemos todas las columnas esperadas
        if len(feature_data) != len(expected_columns):
            print(f"❌ Error: Número incorrecto de features. Esperado: {len(expected_columns)}, Recibido: {len(feature_data)}")
            print(f"Features esperadas: {expected_columns}")
            print(f"Features recibidas: {list(feature_data.keys())}")
            raise HTTPException(status_code=400, detail="Número incorrecto de características")
        
        # Convertir a DataFrame con el orden correcto
        df_input = pd.DataFrame([feature_data])[expected_columns]
        
        print(f"📊 Input DataFrame shape: {df_input.shape}")
        print(f"📊 Input values: {df_input.iloc[0].to_dict()}")
        
        # Hacer predicción directa (el modelo ya está entrenado y no necesita preprocessing)
        prediction = model.predict(df_input)[0]
        
        # Validar predicción
        if pd.isna(prediction) or prediction <= 0:
            print(f"❌ Predicción inválida: {prediction}")
            raise HTTPException(status_code=500, detail="Predicción inválida generada por el modelo")
        
        print(f"✅ Predicción exitosa: ${prediction:,.2f}")
        
        return PropertyPredictionResponse(
            prediction=float(prediction),
            message="Predicción exitosa usando Random Forest",
            model_info=model_data['model_info']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error en predicción: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.get("/health")
def health():
    try:
        model_data = load_properties_model()
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_type": model_data['model_info']['type'],
            "features_count": model_data['model_info']['features_count'],
            "preprocessing": model_data['model_info']['preprocessing']
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}