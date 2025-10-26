"""
Modelos de datos para la API de predicción de ACV (Accidente Cerebrovascular)
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class ACVPredictionRequest(BaseModel):
    """Modelo para las solicitudes de predicción de ACV"""
    query: str = Field(..., description="Consulta del usuario en lenguaje natural")
    
    # Características demográficas
    age: Optional[float] = Field(None, description="Edad del paciente", ge=0, le=120)
    gender: Optional[str] = Field(None, description="Género: 'Male', 'Female', 'Other'")
    
    # Condiciones médicas
    hypertension: Optional[int] = Field(None, description="Hipertensión: 0 = No, 1 = Sí", ge=0, le=1)
    heart_disease: Optional[int] = Field(None, description="Enfermedad cardíaca: 0 = No, 1 = Sí", ge=0, le=1)
    
    # Estilo de vida
    ever_married: Optional[str] = Field(None, description="Estado civil: 'Yes', 'No'")
    work_type: Optional[str] = Field(None, description="Tipo de trabajo: 'Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'")
    residence_type: Optional[str] = Field(None, description="Tipo de residencia: 'Urban', 'Rural'")
    smoking_status: Optional[str] = Field(None, description="Estado de fumador: 'never smoked', 'formerly smoked', 'smokes', 'Unknown'")
    
    # Indicadores de salud
    avg_glucose_level: Optional[float] = Field(None, description="Nivel promedio de glucosa", ge=0)
    bmi: Optional[float] = Field(None, description="Índice de masa corporal", ge=10, le=100)

class ACVPredictionResponse(BaseModel):
    """Modelo para las respuestas de predicción de ACV"""
    prediction: int = Field(..., description="Predicción: 0 = Sin riesgo de ACV, 1 = Con riesgo de ACV")
    probability: float = Field(..., description="Probabilidad de riesgo de ACV (0-1)")
    risk_level: str = Field(..., description="Nivel de riesgo: 'Bajo', 'Moderado', 'Alto'")
    confidence: float = Field(..., description="Confianza en la predicción (0-100)")
    message: str = Field(..., description="Mensaje explicativo del resultado")
    model_info: Dict[str, Any] = Field(..., description="Información del modelo utilizado")
    recommendations: list = Field(..., description="Recomendaciones basadas en el resultado")

class ACVHealthResponse(BaseModel):
    """Modelo para el estado de salud de la API de ACV"""
    status: str = Field(..., description="Estado del servicio")
    model_loaded: bool = Field(..., description="Si el modelo está cargado")
    model_type: str = Field(..., description="Tipo de modelo")
    features_count: int = Field(..., description="Número de características del modelo")
    preprocessing: str = Field(..., description="Información del preprocesamiento")