try:
    from langchain_ollama import OllamaLLM
    llm = OllamaLLM(model="llama3")
except ImportError:
    # Fallback a la versión antigua si no está instalado langchain-ollama
    from langchain_community.llms import Ollama
    llm = Ollama(model="llama3")

import requests
import json
from .extract_params import (
    extract_bitcoin_parameters,
    extract_flights_parameters,
    extract_properties_parameters,
    extract_movies_parameters,
    extract_acv_parameters,
    extract_avocado_parameters
)
from .available_models import MODELS_CONFIG


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
            
            # Extraer parámetros específicos según el modelo
            if modelo == "bitcoin":
                bitcoin_params = extract_bitcoin_parameters(query, llm)
                if bitcoin_params:
                    data.update(bitcoin_params)
                    print(f"🎯 Parámetros extraídos para Bitcoin: {bitcoin_params}")
            
            elif modelo == "flights":
                flights_params = extract_flights_parameters(query, llm)
                if flights_params:
                    data.update(flights_params)
                    print(f"✈️ Parámetros extraídos para Vuelos: {flights_params}")
            
            elif modelo == "properties":
                properties_params = extract_properties_parameters(query, llm)
                if properties_params:
                    data.update(properties_params)
                    print(f"🏠 Parámetros extraídos para Propiedades: {properties_params}")
            
            elif modelo == "movies":
                movies_params = extract_movies_parameters(query, llm)
                if movies_params:
                    data.update(movies_params)
                    print(f"🎬 Parámetros extraídos para Películas: {movies_params}")
                
                # Para películas, podríamos necesitar un endpoint diferente si es predicción de rating
                if "user_id" in data and "movie_id" in data:
                    model_config["endpoint"] = "http://localhost:8000/movies/models/movies/predict-rating"
            
            elif modelo == "acv":
                acv_params = extract_acv_parameters(query, llm)
                if acv_params:
                    data.update(acv_params)
                    print(f"🏥 Parámetros extraídos para ACV: {acv_params}")
            
            elif modelo == "avocado":
                avocado_params = extract_avocado_parameters(query, llm)
                if avocado_params:
                    data.update(avocado_params)
                    print(f"🥑 Parámetros extraídos para Aguacate: {avocado_params}")
            
            response = requests.post(model_config["endpoint"], json=data, timeout=60)
            
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
    - Si es 'time_series_prediction' (predicción temporal): Explica las tendencias, fechas específicas, valores predichos y intervalos de confianza
    - Si es 'flight_prediction' (predicción de vuelos): Explica el retraso esperado, factores que influyen, nivel de confianza y recomendaciones
    - Si es 'prediction' (predicción): Incluye el valor predicho, tendencia y nivel de confianza
    - Si es 'classification' (clasificación): Explica la categoría predicha y probabilidad
    - Si es 'recommendation' (recomendación): Lista las recomendaciones principales y razones
    - Si es 'medical_classification' (clasificación médica): Explica el nivel de riesgo, probabilidad, factores de riesgo identificados y recomendaciones médicas

    Para predicciones de Bitcoin con Prophet:
    - Menciona las fechas específicas y sus precios predichos
    - Explica la tendencia general (alcista, bajista, estable)
    - Incluye los intervalos de confianza si están disponibles
    - Menciona limitaciones del modelo (predicciones son estimaciones)

    Instrucciones generales:
    1. Explica qué significa el resultado en términos simples
    2. Menciona cualquier limitación o consideración importante
    3. Sé conciso pero informativo
    4. Usa emojis apropiados para hacer la respuesta más amigable

    Respuesta:
    """

    try:
        # Siempre usar el LLM para generar una respuesta conversacional completa
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
                return f"💰 Predicción Bitcoin: ${prediction:,.2f} USD (Confianza: {confidence:.1f}%)"
                
        elif response_type == "prediction":
            if modelo == "properties" and "prediction" in result:
                prediction = result.get("prediction", 0)
                confidence = result.get("confidence", 0)
                return f"🏠 Precio estimado de propiedad: ${prediction:,.2f} USD (Confianza: {confidence:.1f}%)"
            
            # TODO: Agregar formato para otros modelos de predicción (churn, etc.)
            
        elif response_type == "classification":
            # TODO: Implementar formato para modelos de clasificación (wine, emotions)
            if "predicted_class" in result:
                predicted_class = result.get("predicted_class", "Desconocido")
                probability = result.get("probability", 0)
                return f"🎯 Clasificación: {predicted_class} (Probabilidad: {probability:.1f}%)"
        
        elif response_type == "medical_classification":
            if modelo == "acv":
                prediction = result.get("prediction", 0)
                probability = result.get("probability", 0)
                risk_level = result.get("risk_level", "Desconocido")
                confidence = result.get("confidence", 0)
                
                if prediction == 1:
                    return f"⚠️ RIESGO ALTO de ACV: {probability:.1%} probabilidad - Nivel: {risk_level} (Confianza: {confidence:.1f}%)"
                else:
                    return f"✅ RIESGO BAJO de ACV: {(1-probability):.1%} probabilidad - Nivel: {risk_level} (Confianza: {confidence:.1f}%)"
                
        elif response_type == "recommendation":
            if modelo == "movies":
                if "recommendations" in result:
                    recs = result.get("recommendations", [])[:3]  # Top 3
                    if recs:
                        movie_titles = [rec.get("title", "Película desconocida") for rec in recs]
                        return f"🎬 Recomendaciones de películas: {', '.join(movie_titles)}"
                
                elif "predicted_rating" in result:
                    rating = result.get("predicted_rating", 0)
                    confidence = result.get("confidence", 0)
                    movie_title = result.get("model_info", {}).get("movie_title", "Película")
                    return f"🎬 Rating predicho para {movie_title}: {rating:.1f}/5.0 (Confianza: {confidence:.1f}%)"
        
        # Respuesta genérica si no hay formato específico
        return f"Resultado del modelo {modelo}: {json.dumps(result, indent=2)}"
        
    except Exception:
        return f"Resultado del modelo {modelo}: {result}"
