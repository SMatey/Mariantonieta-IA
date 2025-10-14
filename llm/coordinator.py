from langchain_community.llms import Ollama
import requests
import json

llm = Ollama(model="llama3")

# Configuración de modelos disponibles
MODELS_CONFIG = {
    "bitcoin": {
        "endpoint": "http://localhost:8000/bitcoin/models/bitcoin/predict",
        "description": "Para predicciones de precios de Bitcoin, criptomonedas, análisis financiero",
        "available": True,
        "response_type": "prediction"
    },
    # TODO: Agregar configuración para otros modelos cuando estén implementados
    "wine": {
        "endpoint": "http://localhost:8000/wine/classify",  # TODO: Implementar endpoint
        "description": "Para clasificación de vinos basada en características químicas",
        "available": False,
        "response_type": "classification"
    },
    "churn": {
        "endpoint": "http://localhost:8000/churn/predict",  # TODO: Implementar endpoint
        "description": "Para predicción de abandono de clientes",
        "available": False,
        "response_type": "prediction"
    },
    "movies": {
        "endpoint": "http://localhost:8000/movies/recommend",  # TODO: Implementar endpoint
        "description": "Para recomendaciones de películas personalizadas",
        "available": False,
        "response_type": "recommendation"
    },
    "emotions": {
        "endpoint": "http://localhost:8000/emotions/analyze",  # TODO: Implementar endpoint
        "description": "Para análisis de emociones en texto",
        "available": False,
        "response_type": "classification"
    }
}

def get_available_models():
    """Retorna lista de modelos disponibles"""
    return {name: config for name, config in MODELS_CONFIG.items() if config["available"]}

def interpretar_y_ejecutar(query: str):
    """
    Coordinador principal que decide qué modelo usar y ejecuta la consulta
    """
    # Paso 1: el LLM decide qué modelo usar
    available_models = get_available_models()
    
    # Construir la descripción de modelos disponibles dinámicamente
    models_description = "\n".join([
        f"    - {name}: {config['description']}"
        for name, config in MODELS_CONFIG.items()
        if config["available"]
    ])
    
    # Agregar modelos no disponibles
    unavailable_models = "\n".join([
        f"    - {name}: {config['description']} (no disponible aún)"
        for name, config in MODELS_CONFIG.items()
        if not config["available"]
    ])
    
    decision_prompt = f"""
    Eres un coordinador de modelos de IA. Analiza la siguiente consulta y decide qué modelo usar.

    Consulta: "{query}"

    Modelos disponibles:
{models_description}

    Modelos en desarrollo:
{unavailable_models}

    Responde SOLO con el nombre del modelo más apropiado ({', '.join(MODELS_CONFIG.keys())}).
    Si no hay un modelo apropiado, responde "ninguno".
    """
    
    decision = llm.invoke(decision_prompt)
    modelo = decision.strip().lower()

    # Paso 2: verificar si el modelo está disponible y hacer la consulta
    if modelo in MODELS_CONFIG:
        model_config = MODELS_CONFIG[modelo]
        
        if not model_config["available"]:
            return f"El modelo '{modelo}' está en desarrollo y no está disponible aún. Actualmente solo tengo disponible: {', '.join(get_available_models().keys())}"
        
        # Hacer la consulta al modelo
        try:
            data = {"query": query}
            response = requests.post(model_config["endpoint"], json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
            else:
                return f"Error al consultar el modelo {modelo}: {response.status_code} - {response.text}"
                
        except requests.exceptions.RequestException as e:
            return f"Error de conexión con el modelo {modelo}: {str(e)}"
        except Exception as e:
            return f"Error inesperado al consultar {modelo}: {str(e)}"
    else:
        if modelo == "ninguno":
            available_list = ', '.join(get_available_models().keys())
            return f"Lo siento, no tengo un modelo específico para responder a esa consulta. Actualmente puedo ayudarte con: {available_list}"
        else:
            return f"El modelo '{modelo}' no existe. Modelos disponibles: {', '.join(get_available_models().keys())}"

    # Paso 3: interpreta el resultado con el LLM
    interpretation_prompt = f"""
    Un modelo de {modelo} (tipo: {model_config['response_type']}) devolvió este resultado para la consulta "{query}":

    Resultado: {json.dumps(result, indent=2)}

    Tu tarea es interpretar este resultado y explicárselo al usuario de forma natural, clara y útil.

    Instrucciones específicas según el tipo de modelo:
    - Si es 'prediction' (predicción): Incluye el valor predicho, tendencia y nivel de confianza
    - Si es 'classification' (clasificación): Explica la categoría predicha y probabilidad
    - Si es 'recommendation' (recomendación): Lista las recomendaciones principales y razones

    Instrucciones generales:
    1. Explica qué significa el resultado en términos simples
    2. Menciona cualquier limitación o consideración importante
    3. Sé conciso pero informativo
    4. Usa emojis apropiados para hacer la respuesta más amigable

    Respuesta:
    """

    try:
        explicacion = llm.invoke(interpretation_prompt)
        return explicacion
    except Exception as e:
        # Si falla la interpretación, devolver el resultado de forma más amigable
        return format_fallback_response(modelo, result, model_config['response_type'])

def format_fallback_response(modelo: str, result: dict, response_type: str):
    """
    Formatea una respuesta de respaldo cuando falla la interpretación del LLM
    """
    try:
        if response_type == "prediction":
            if modelo == "bitcoin" and "prediction" in result:
                prediction = result.get("prediction", 0)
                confidence = result.get("confidence", 0)
                return f"💰 Predicción de Bitcoin: ${prediction:,.2f} USD (Confianza: {confidence:.1f}%)"
            # TODO: Agregar formato para otros modelos de predicción (churn, etc.)
            
        elif response_type == "classification":
            # TODO: Implementar formato para modelos de clasificación (wine, emotions)
            if "predicted_class" in result:
                predicted_class = result.get("predicted_class", "Desconocido")
                probability = result.get("probability", 0)
                return f"🎯 Clasificación: {predicted_class} (Probabilidad: {probability:.1f}%)"
                
        elif response_type == "recommendation":
            # TODO: Implementar formato para modelos de recomendación (movies)
            if "recommendations" in result:
                recs = result.get("recommendations", [])[:3]  # Top 3
                return f"🎬 Recomendaciones: {', '.join(recs)}"
        
        # Respuesta genérica si no hay formato específico
        return f"Resultado del modelo {modelo}: {json.dumps(result, indent=2)}"
        
    except Exception:
        return f"Resultado del modelo {modelo}: {result}"
