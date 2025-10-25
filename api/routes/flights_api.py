"""
API de Vuelos - Predicción de retrasos usando Machine Learning
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, Optional
from datetime import datetime
from ..models.flights_api_models import FlightPredictionRequest, FlightPredictionResponse

# Importar constantes desde el paquete api
from .. import constants as const

app = FastAPI(title="Flight Delay Prediction API", version="1.0.0")



# Variable global para el modelo
_loaded_model_data = None

def load_flights_model():
    """
    Carga el modelo REAL de predicción de retrasos de vuelos.
    """
    global _loaded_model_data
    
    if _loaded_model_data is not None:
        return _loaded_model_data
    
    try:
        model_path = os.path.join(const.ML_MODELS_PATH, "flight_delay_v1_2025-10-25.pkl")
        print(f"Cargando modelo de vuelos desde: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
        
        # Cargar el modelo real usando joblib
        _loaded_model_data = joblib.load(model_path)
        
        print("Modelo de vuelos cargado exitosamente")
        return _loaded_model_data
        
    except Exception as e:
        print(f"Error cargando modelo de vuelos: {e}")
        raise HTTPException(status_code=500, detail=f"Error cargando modelo: {str(e)}")

def transform_user_data_to_model_format(flight_data):
    """
    Transforma los datos del coordinador al formato esperado por el modelo ML.
    """
    from datetime import datetime
    
    try:
        print(f"Datos recibidos: {flight_data}")
        
        # Procesar fecha con validación de None
        date_str = flight_data.get('date')
        if not date_str:
            date_str = datetime.now().strftime("%Y-%m-%d")
        
        flight_date = datetime.strptime(date_str, "%Y-%m-%d")
        flight_date = flight_date.replace(year=2008)

        
        # Procesar hora de salida con validación
        departure_time_str = flight_data.get('departure_time')
        if not departure_time_str:
            departure_time_str = '12:00'
            
        if ':' in departure_time_str:
            hour, minute = departure_time_str.split(':')
            dep_time = int(hour) * 100 + int(minute)
        else:
            dep_time = 1200
        
        # Obtener aeropuertos con valores por defecto
        origin = flight_data.get('origin')
        destination = flight_data.get('destination')
        
        if not origin:
            origin = 'LAX'
        if not destination:
            destination = 'SFO'
            
        print(f"🔍 Origin: {origin}, Destination: {destination}")
        
        # Estimar distancia
        distance = flight_data.get('distance')
        if not distance:
            distance_map = {
                ('LAX', 'SFO'): 337, ('SFO', 'LAX'): 337,
                ('DEN', 'LAS'): 628, ('LAS', 'DEN'): 628,
                ('JFK', 'LAX'): 2475, ('LAX', 'JFK'): 2475,
                ('ORD', 'LAX'): 1745, ('LAX', 'ORD'): 1745,
            }
            distance = distance_map.get((origin, destination), 1000)
        
        # Calcular hora de llegada
        flight_duration_hours = distance / 500
        total_minutes = int(flight_duration_hours * 60) + 30
        
        dep_hour = dep_time // 100
        dep_minute = dep_time % 100
        total_dep_minutes = dep_hour * 60 + dep_minute
        arr_minutes = total_dep_minutes + total_minutes
        
        arr_hour = (arr_minutes // 60) % 24
        arr_min = arr_minutes % 60
        crs_arr_time = arr_hour * 100 + arr_min
        
        # Obtener aerolínea con valor por defecto
        airline = flight_data.get('airline')
        if not airline:
            airline = 'AA'
            
        # Formatear datos para el modelo
        model_input = {
            "Month": flight_date.month,
            "DayofMonth": flight_date.day,
            "DayOfWeek": flight_date.weekday() + 1,
            "DepTime": dep_time,
            "CRSDepTime": dep_time,
            "CRSArrTime": crs_arr_time,
            "UniqueCarrier": airline,
            "Origin": origin,
            "Dest": destination,
            "Distance": int(distance),
            "DepDelay": float(flight_data.get('delay_at_departure') or 0)
        }
        
        print(f"🔍 Datos transformados: {model_input}")
        return model_input
        
    except Exception as e:
        print(f"Error transformando datos: {e}")
        print(f"Datos problemáticos: {flight_data}")
        raise ValueError(f"No se pudieron transformar los datos: {str(e)}")

def make_flight_prediction(model_data, flight_data):
    """Genera predicciones usando el modelo ML"""
    
    try:
        # Extraer el modelo real del diccionario
        model = model_data['model']
        feature_columns = model_data['feature_columns']
        
        # Transformar datos al formato del modelo
        model_input = transform_user_data_to_model_format(flight_data)
        
        # El modelo espera valores numéricos para las variables categóricas
        # Mapear las variables categóricas manualmente según el entrenamiento
        
        # Mapeo de aerolíneas (estos valores deben coincidir con el entrenamiento)
        airline_mapping = {
            'WN': 0, 'AA': 1, 'DL': 2, 'UA': 3, 'US': 4, 'NW': 5, 'CO': 6, 
            'FL': 7, 'AS': 8, 'B6': 9, 'YV': 10, 'OO': 11, 'XE': 12, 'EV': 13,
            'F9': 14, '9E': 15, 'HA': 16, 'MQ': 17, 'OH': 18, 'TZ': 19
        }
        
        # Mapeo de aeropuertos (muestra - estos deben coincidir con el entrenamiento)
        airport_mapping = {
            'ATL': 0, 'BOS': 1, 'BWI': 2, 'CLT': 3, 'DCA': 4, 'DEN': 5, 'DFW': 6,
            'DTW': 7, 'EWR': 8, 'FLL': 9, 'IAD': 10, 'IAH': 11, 'JFK': 12, 'LAS': 13,
            'LAX': 14, 'LGA': 15, 'MCO': 16, 'MDW': 17, 'MIA': 18, 'MSP': 19, 'ORD': 20,
            'PHL': 21, 'PHX': 22, 'SEA': 23, 'SFO': 24, 'SLC': 25, 'TPA': 26
        }
        
        # Aplicar mapeos
        if 'UniqueCarrier' in model_input:
            carrier = model_input['UniqueCarrier']
            model_input['UniqueCarrier'] = airline_mapping.get(carrier, 1)  # Default AA
            
        if 'Origin' in model_input:
            origin = model_input['Origin']
            model_input['Origin'] = airport_mapping.get(origin, 14)  # Default LAX
            
        if 'Dest' in model_input:
            dest = model_input['Dest']
            model_input['Dest'] = airport_mapping.get(dest, 24)  # Default SFO
        
        # Crear DataFrame con el orden correcto de columnas
        input_df = pd.DataFrame([model_input])
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)
        
        print(f"🔍 Input final para el modelo: {input_df.iloc[0].to_dict()}")
        
        # Hacer predicción con el modelo REAL
        prediction = model.predict(input_df)[0]
        
        # Calcular confianza basada en el R² del entrenamiento
        training_info = model_data.get('training_info', {})
        r2_score = training_info.get('r2_score', 0.8)
        confidence = min(r2_score * 100, 95.0)  # Convertir R² a porcentaje
        
        return float(prediction), float(confidence)
        
    except Exception as e:
        print(f"❌ Error en predicción ML: {e}")
        import traceback
        traceback.print_exc()
        raise ValueError(f"Error en predicción: {str(e)}")

@app.get("/")
def root():
    return {
        "message": "Flight Delay Prediction API", 
        "version": "1.0.0", 
        "description": "Predice retrasos de vuelos basado en características del vuelo"
    }

@app.post("/models/flights/predict", response_model=FlightPredictionResponse)
def predict_flight_delay(request: FlightPredictionRequest):
    """
    Predice el retraso de un vuelo basado en sus características
    """
    try:
        # Validar y corregir fecha si es necesaria
        validated_date = request.date
        if request.date:
            try:
                # Verificar si la fecha es válida
                parsed_date = datetime.strptime(request.date, "%Y-%m-%d")
                # Si la fecha es muy antigua, usar fecha actual
                if parsed_date.year < 2020:
                    validated_date = datetime.now().strftime("%Y-%m-%d")
            except ValueError:
                # Si la fecha no es válida, usar fecha actual
                validated_date = datetime.now().strftime("%Y-%m-%d")
        else:
            # Si no hay fecha, usar fecha actual
            validated_date = datetime.now().strftime("%Y-%m-%d")
        
        # Preparar datos del vuelo
        flight_data = {
            'date': validated_date,
            'departure_time': request.departure_time,
            'origin': request.origin,
            'destination': request.destination,
            'airline': request.airline,
            'distance': request.distance,
            'delay_at_departure': request.delay_at_departure
        }
        
        # Cargar modelo ML y hacer predicción
        model_data = load_flights_model()
        predicted_delay, confidence = make_flight_prediction(model_data, flight_data)
        
        # Información del vuelo para la respuesta
        flight_info = {
            'route': f"{request.origin} → {request.destination}",
            'airline': request.airline,
            'departure': f"{validated_date} {request.departure_time}",
            'distance_km': request.distance,
            'departure_delay': request.delay_at_departure
        }
        
        # Información del modelo REAL
        model_info = {
            'model_type': 'Machine Learning',
            'features_used': ['Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'CRSDepTime', 'CRSArrTime', 'UniqueCarrier', 'Origin', 'Dest', 'Distance', 'DepDelay'],
            'confidence_level': confidence
        }
        
        # Generar interpretación
        if predicted_delay <= 5:
            delay_category = "puntual"
            emoji = "✅"
        elif predicted_delay <= 15:
            delay_category = "retraso leve"
            emoji = "🟡"
        elif predicted_delay <= 30:
            delay_category = "retraso moderado"
            emoji = "🟠"
        else:
            delay_category = "retraso significativo"
            emoji = "🔴"
        
        interpretation = f"{emoji} Predicción para vuelo {request.airline} {request.origin}-{request.destination}: " \
                        f"{predicted_delay:.0f} minutos de retraso ({delay_category}). " \
                        f"Confianza: {confidence:.1f}%"
        
        return FlightPredictionResponse(
            query=request.query,
            prediction=predicted_delay,
            confidence=confidence,
            flight_info=flight_info,
            model_info=model_info,
            interpretation=interpretation
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción de vuelos: {str(e)}")

@app.get("/health")
def health():
    try:
        model = load_flights_model()
        
        return {
            "status": "healthy",
            "model_available": True,
            "model_type": "Machine Learning - Random Forest",
            "model_file": "flight_delay_v1_2025-10-25.pkl",
            "endpoints": ["/models/flights/predict"],
            "supported_airlines": ["UA", "AA", "DL", "WN", "B6", "AS", "NK", "F9", "G4", "SY"],
            "supported_airports": ["SFO", "JFK", "LAX", "ORD", "DFW", "DEN", "ATL", "SEA", "LAS", "PHX"]
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "model_available": False}

if __name__ == "__main__":
    import uvicorn
    print("✈️ Flight Delay Prediction API")
    uvicorn.run(app, host="0.0.0.0", port=8003)