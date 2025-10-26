"""
API de Predicción de ACV (Accidente Cerebrovascular) usando Árbol de Decisión
"""
from fastapi import FastAPI, HTTPException
import pickle
import pandas as pd
import numpy as np
import os
from ..models.acv_api_models import ACVPredictionRequest, ACVPredictionResponse

# Importar constantes desde el paquete api
from .. import constants as const

app = FastAPI(title="ACV Prediction API", version="1.0.0")

# Variable global para el modelo
_loaded_model_data = None

def load_acv_model():
    """Carga el modelo de Árbol de Decisión para predicción de ACV"""
    global _loaded_model_data
    if _loaded_model_data is not None:
        return _loaded_model_data
    
    try:
        model_path = os.path.join(const.BASE_DIR, 'ml_models', 'ACV_decision_tree_model.pkl')
        with open(model_path, 'rb') as f:
            modelo = pickle.load(f)
        
        # Características que el modelo espera (según el notebook)
        # NOTA: El modelo fue entrenado con 'stroke' en los datos de entrada aunque no la use para predicción
        expected_features = [
            'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
            'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status', 'stroke'
        ]
        
        _loaded_model_data = {
            'model': modelo,
            'expected_features': expected_features,
            'model_info': {
                'type': 'Decision Tree Pipeline',
                'features_count': len(expected_features) - 1,  # Restar 1 porque 'stroke' no cuenta como feature real
                'preprocessing': 'StandardScaler + OneHotEncoder + Median Imputation',
                'accuracy': 1.0  # Según el notebook
            }
        }
        print("✅ Modelo de ACV cargado exitosamente")
        return _loaded_model_data
        
    except Exception as e:
        print(f"❌ Error cargando modelo de ACV: {e}")
        raise HTTPException(status_code=500, detail=f"Error cargando modelo: {str(e)}")

def create_features_from_acv_data(request: ACVPredictionRequest):
    """
    Crea las características para el modelo desde el request.
    Usa valores por defecto médicamente razonables si no se proporcionan.
    """
    # Valores por defecto basados en la población general
    features_dict = {
        'gender': request.gender if request.gender else 'Female',  # Más común en el dataset
        'age': request.age if request.age else 45.0,  # Edad promedio adulta
        'hypertension': request.hypertension if request.hypertension is not None else 0,
        'heart_disease': request.heart_disease if request.heart_disease is not None else 0,
        'ever_married': request.ever_married if request.ever_married else 'Yes',  # Más común
        'work_type': request.work_type if request.work_type else 'Private',  # Más común
        'Residence_type': request.residence_type if request.residence_type else 'Urban',  # Ligeramente más común
        'avg_glucose_level': request.avg_glucose_level if request.avg_glucose_level else 106.0,  # Normal
        'bmi': request.bmi if request.bmi else 28.9,  # Promedio del dataset
        'smoking_status': request.smoking_status if request.smoking_status else 'never smoked'  # Más común
    }
    
    print(f"🔍 Features de ACV creadas: {features_dict}")
    return features_dict

def calculate_risk_level(probability):
    """Calcula el nivel de riesgo basado en la probabilidad"""
    if probability < 0.3:
        return "Bajo"
    elif probability < 0.7:
        return "Moderado"
    else:
        return "Alto"

def get_recommendations(prediction, probability, features_dict):
    """Genera recomendaciones basadas en la predicción y características del paciente"""
    recommendations = []
    
    if prediction == 1:  # Alto riesgo de ACV
        recommendations.extend([
            "⚠️ Consulte a un médico inmediatamente para evaluación cardiovascular",
            "📍 Considere un chequeo neurológico especializado",
            "💊 Siga estrictamente las indicaciones médicas para medicamentos"
        ])
    
    # Recomendaciones específicas basadas en características
    if features_dict.get('age', 0) > 65:
        recommendations.append("👴 Mantenga controles médicos regulares debido a la edad avanzada")
    
    if features_dict.get('hypertension') == 1:
        recommendations.append("🩺 Control estricto de la presión arterial")
    
    if features_dict.get('heart_disease') == 1:
        recommendations.append("❤️ Seguimiento cardiológico especializado")
    
    if features_dict.get('avg_glucose_level', 0) > 126:
        recommendations.append("🍯 Control de glucosa en sangre - posible diabetes")
    
    if features_dict.get('bmi', 0) > 30:
        recommendations.append("🏃‍♀️ Plan de reducción de peso y ejercicio regular")
    
    if features_dict.get('smoking_status') in ['smokes', 'formerly smoked']:
        recommendations.append("🚭 Cesación completa del tabaquismo")
    
    # Recomendaciones generales
    recommendations.extend([
        "🥗 Dieta mediterránea rica en frutas y verduras",
        "🏃‍♂️ Ejercicio regular (30 min/día, 5 días/semana)",
        "😴 Mantener patrones de sueño saludables (7-8 horas)",
        "🧘‍♀️ Manejo del estrés mediante técnicas de relajación"
    ])
    
    return recommendations

@app.get("/")
def root():
    return {"message": "ACV Prediction API", "version": "1.0.0"}

@app.post("/predict", response_model=ACVPredictionResponse)
def predict_acv_risk(request: ACVPredictionRequest):
    """Predice el riesgo de ACV usando el modelo de Árbol de Decisión entrenado"""
    try:
        # Cargar modelo
        model_data = load_acv_model()
        modelo = model_data['model']
        expected_features = model_data['expected_features']
        
        # Crear características de entrada
        features_dict = create_features_from_acv_data(request)
        
        # Convertir a DataFrame con las columnas en el orden correcto
        # Agregar 'stroke' con valor dummy (0) ya que el modelo lo espera pero no lo usa
        features_dict['stroke'] = 0  # Valor dummy, no afecta la predicción
        input_df = pd.DataFrame([features_dict])[expected_features]
        
        print(f"📊 Input DataFrame shape: {input_df.shape}")
        print(f"📊 Input features: {input_df.iloc[0].to_dict()}")
        
        # Hacer predicción
        prediction = modelo.predict(input_df)[0]
        probabilities = modelo.predict_proba(input_df)[0]
        
        # La probabilidad de ACV es la probabilidad de la clase 1
        acv_probability = probabilities[1] if len(probabilities) > 1 else 0.0
        
        # Calcular nivel de riesgo
        risk_level = calculate_risk_level(acv_probability)
        
        # Calcular confianza (basada en la certeza de la predicción)
        confidence = max(probabilities) * 100
        
        # Generar recomendaciones
        recommendations = get_recommendations(prediction, acv_probability, features_dict)
        
        # Mensaje explicativo
        if prediction == 1:
            message = f"⚠️ ALTO RIESGO: El modelo predice riesgo elevado de ACV con {acv_probability:.1%} de probabilidad"
        else:
            message = f"✅ BAJO RIESGO: El modelo predice bajo riesgo de ACV con {(1-acv_probability):.1%} de probabilidad"
        
        print(f"✅ Predicción exitosa: {prediction} (probabilidad: {acv_probability:.3f})")
        
        return ACVPredictionResponse(
            prediction=int(prediction),
            probability=float(acv_probability),
            risk_level=risk_level,
            confidence=float(confidence),
            message=message,
            model_info=model_data['model_info'],
            recommendations=recommendations
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error en predicción de ACV: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.get("/health")
def health():
    try:
        model_data = load_acv_model()
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_type": model_data['model_info']['type'],
            "features_count": model_data['model_info']['features_count'],
            "preprocessing": model_data['model_info']['preprocessing'],
            "accuracy": model_data['model_info']['accuracy']
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}