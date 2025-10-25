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
    "properties": {
        "endpoint": "http://localhost:8000/properties/models/properties/predict",
        "description": "Para predicción de precios de propiedades inmobiliarias, casas, apartamentos",
        "available": True,
        "response_type": "prediction"
    },
    "movies": {
        "endpoint": "http://localhost:8000/movies/models/movies/recommend",
        "description": "Para recomendaciones de películas personalizadas basadas en preferencias",
        "available": True,
        "response_type": "recommendation"
    },
    "flights": {
        "endpoint": "http://localhost:8000/flights/models/flights/predict",
        "description": "Para predicciones de retrasos de vuelos, análisis de puntualidad y planificación de viajes",
        "available": True,
        "response_type": "flight_prediction"
    },
    "wine": {
        "endpoint": "http://localhost:8000/wine/classify",
        "description": "Para clasificación de vinos basada en características químicas",
        "available": False,
        "response_type": "classification"
    },
    "churn": {
        "endpoint": "http://localhost:8000/churn/predict",
        "description": "Para predicción de abandono de clientes",
        "available": False,
        "response_type": "prediction"
    },
    "emotions": {
        "endpoint": "http://localhost:8000/emotions/analyze",
        "description": "Para análisis de emociones en texto",
        "available": False,
        "response_type": "classification"
    }
}

def extract_bitcoin_parameters(query: str):
    """
    Extrae parámetros numéricos del texto para el modelo Bitcoin
    """
    extraction_prompt = f"""
    Extrae valores numéricos específicos para predicción de Bitcoin del siguiente texto:
    
    "{query}"
    
    Busca y extrae SOLO los valores que se mencionen explícitamente:
    - Precio actual/open (ej: "precio actual 32500", "bitcoin está en 31000")
    - Precio máximo/high (ej: "máximo 33000", "high 32800")
    - Precio mínimo/low (ej: "mínimo 31500", "low 31200")
    - Volumen (ej: "volumen 2B", "2 billones de volumen", "1.5B USD")
    - Market cap (ej: "market cap 600B", "capitalización 700 billones")
    - RSI (ej: "RSI 65", "RSI de 72.5")
    - Medias móviles (ej: "MA5 31800", "media móvil 20 días 31500")
    
    Responde SOLO en formato JSON válido con los valores encontrados:
    {{
        "open_price": 32500.0,
        "high_price": null,
        "volume": 2000000000.0,
        "rsi_14": 65.0
    }}
    
    Si NO encuentras un valor específico, usa null.
    NO inventes valores, SOLO extrae los mencionados explícitamente.
    """
    
    try:
        extraction_result = llm.invoke(extraction_prompt)
        # Intentar parsear como JSON
        import json
        import re
        
        # Limpiar la respuesta para extraer solo el JSON
        json_match = re.search(r'\{.*\}', extraction_result, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            extracted_params = json.loads(json_str)
            # Filtrar valores null
            filtered_params = {k: v for k, v in extracted_params.items() if v is not None}
            return filtered_params
        else:
            return {}
    except Exception as e:
        print(f"Error extrayendo parámetros: {e}")
        return {}

def extract_properties_parameters(query: str):
    """
    Extrae parámetros para predicción de precios de propiedades
    """
    extraction_prompt = f"""
    Extrae características de propiedades del siguiente texto:
    
    "{query}"
    
    Busca y extrae SOLO los valores mencionados explícitamente:
    - Baños (ej: "3 baños", "2.5 bathrooms", "4 bath")
    - Habitaciones (ej: "4 habitaciones", "3 bedrooms", "5 bed")
    - Pies cuadrados (ej: "2500 sq ft", "1800 pies cuadrados", "3000 square feet")
    - Año construcción (ej: "construida en 1990", "built in 2005", "año 2010")
    - Tamaño del lote (ej: "7000 sq ft lot", "0.5 acres", "5000 pies cuadrados de terreno")
    - Coordenadas (ej: "latitud 34.05", "longitude -118.25")
    - Impuestos (ej: "taxes $5000", "impuestos 4500 anuales")
    
    Responde SOLO en formato JSON válido:
    {{
        "bathroomcnt": 3.0,
        "bedroomcnt": 4.0,
        "finishedsquarefeet": 2500.0,
        "yearbuilt": 1990.0,
        "lotsizesquarefeet": 7000.0,
        "latitude": null,
        "longitude": null,
        "taxamount": 5000.0
    }}
    
    Si NO encuentras un valor específico, usa null.
    """
    
    try:
        extraction_result = llm.invoke(extraction_prompt)
        import json
        import re
        
        json_match = re.search(r'\{.*\}', extraction_result, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            extracted_params = json.loads(json_str)
            filtered_params = {k: v for k, v in extracted_params.items() if v is not None}
            return filtered_params
        else:
            return {}
    except Exception as e:
        print(f"Error extrayendo parámetros de propiedades: {e}")
        return {}

def extract_flights_parameters(query: str):
    """
    Extrae parámetros para predicción de retrasos de vuelos
    """
    extraction_prompt = f"""
    Extrae información de vuelos del siguiente texto:
    
    "{query}"
    
    Busca y extrae SOLO los valores mencionados explícitamente:
    - Fecha de vuelo (ej: "mañana", "25 de octubre", "2025-10-25", "hoy")
    - Hora de salida (ej: "7:00 AM", "19:30", "3 p.m.", "15:00")
    - Aeropuerto origen (ej: "SFO", "San Francisco", "LAX", "Los Angeles", "Denver", "Las Vegas")
    - Aeropuerto destino (ej: "JFK", "Nueva York", "ORD", "Chicago")
    - Aerolínea (ej: "United", "UA", "American Airlines", "AA", "Delta", "DL", "Southwest", "WN")
    - Distancia (ej: "2586 km", "1500 millas") - SOLO si se menciona explícitamente
    - Retraso en salida (ej: "retraso de 15 minutos", "sale con 20 min de atraso") - SOLO si se menciona explícitamente
    
    INSTRUCCIONES IMPORTANTES:
    - Convierte códigos de aeropuertos a códigos IATA de 3 letras
    - Convierte fechas relativas a formato YYYY-MM-DD (hoy es 2025-10-24)
    - Convierte horas a formato HH:MM (24 horas)
    - Si NO encuentras un valor específico, NO lo incluyas en la respuesta
    - Para delay_at_departure usa SOLO números (ej: 15, 0, 30), NUNCA texto
    
    Mapeo de aerolíneas:
    - Southwest = WN
    - United = UA  
    - American = AA
    - Delta = DL
    - JetBlue = B6
    
    Mapeo de aeropuertos:
    - Denver = DEN
    - Las Vegas = LAS
    - San Francisco = SFO
    - New York JFK = JFK
    - Los Angeles = LAX
    - Chicago = ORD
    
    Responde SOLO en formato JSON válido:
    {{
        "date": "2025-10-24",
        "departure_time": "15:00",
        "origin": "DEN",
        "destination": "LAS",
        "airline": "WN"
    }}
    
    NO incluyas campos con valores null, undefined, o texto descriptivo.
    Si no hay retraso mencionado, NO incluyas delay_at_departure.
    """
    
    try:
        extraction_result = llm.invoke(extraction_prompt)
        import json
        import re
        from datetime import datetime, timedelta
        
        json_match = re.search(r'\{.*\}', extraction_result, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            extracted_params = json.loads(json_str)
            
            # Procesar fecha si es relativa
            if extracted_params.get("date"):
                date_str = extracted_params["date"]
                if "mañana" in date_str.lower() or "tomorrow" in date_str.lower():
                    tomorrow = datetime.now() + timedelta(days=1)
                    extracted_params["date"] = tomorrow.strftime("%Y-%m-%d")
                elif "hoy" in date_str.lower() or "today" in date_str.lower():
                    today = datetime.now()
                    extracted_params["date"] = today.strftime("%Y-%m-%d")
            
            # Validar y limpiar valores numéricos
            if "delay_at_departure" in extracted_params:
                delay_value = extracted_params["delay_at_departure"]
                if isinstance(delay_value, str):
                    # Intentar extraer números del texto
                    import re
                    numbers = re.findall(r'\d+', delay_value)
                    if numbers:
                        extracted_params["delay_at_departure"] = float(numbers[0])
                    else:
                        # Si no hay números, remover el campo
                        del extracted_params["delay_at_departure"]
                elif not isinstance(delay_value, (int, float)):
                    del extracted_params["delay_at_departure"]
            
            if "distance" in extracted_params:
                distance_value = extracted_params["distance"]
                if isinstance(distance_value, str):
                    # Intentar extraer números del texto
                    import re
                    numbers = re.findall(r'\d+', distance_value)
                    if numbers:
                        extracted_params["distance"] = float(numbers[0])
                    else:
                        del extracted_params["distance"]
                elif not isinstance(distance_value, (int, float)):
                    del extracted_params["distance"]
            
            # Filtrar valores null y vacíos
            filtered_params = {k: v for k, v in extracted_params.items() 
                             if v is not None and v != "" and v != "null"}
            return filtered_params
        else:
            return {}
    except Exception as e:
        print(f"Error extrayendo parámetros de vuelos: {e}")
        return {}

def extract_movies_parameters(query: str):
    """
    Extrae parámetros para recomendaciones de películas
    """
    extraction_prompt = f"""
    Extrae información para recomendaciones de películas del siguiente texto:
    
    "{query}"
    
    Busca y extrae SOLO los valores mencionados explícitamente:
    - ID de película (ej: "película ID 5", "movie 10", "film 25")
    - ID de usuario (ej: "usuario 15", "user 8", "mi ID es 20")
    - Título de película (ej: "Toy Story", "Jumanji", "Heat")
    - Género (ej: "acción", "comedia", "drama", "thriller")
    - Número de recomendaciones (ej: "5 películas", "recomienda 3", "top 10")
    
    Responde SOLO en formato JSON válido:
    {{
        "movie_id": 5,
        "user_id": 15,
        "movie_title": "Toy Story",
        "genre": "acción",
        "num_recommendations": 5
    }}
    
    Si NO encuentras un valor específico, usa null.
    """
    
    try:
        extraction_result = llm.invoke(extraction_prompt)
        import json
        import re
        
        json_match = re.search(r'\{.*\}', extraction_result, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            extracted_params = json.loads(json_str)
            filtered_params = {k: v for k, v in extracted_params.items() if v is not None}
            return filtered_params
        else:
            return {}
    except Exception as e:
        print(f"Error extrayendo parámetros de películas: {e}")
        return {}

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
                bitcoin_params = extract_bitcoin_parameters(query)
                if bitcoin_params:
                    data.update(bitcoin_params)
                    print(f"🎯 Parámetros extraídos para Bitcoin: {bitcoin_params}")
            
            elif modelo == "flights":
                flights_params = extract_flights_parameters(query)
                if flights_params:
                    data.update(flights_params)
                    print(f"✈️ Parámetros extraídos para Vuelos: {flights_params}")
            
            elif modelo == "properties":
                properties_params = extract_properties_parameters(query)
                if properties_params:
                    data.update(properties_params)
                    print(f"🏠 Parámetros extraídos para Propiedades: {properties_params}")
            
            elif modelo == "movies":
                movies_params = extract_movies_parameters(query)
                if movies_params:
                    data.update(movies_params)
                    print(f"🎬 Parámetros extraídos para Películas: {movies_params}")
                
                # Para películas, podríamos necesitar un endpoint diferente si es predicción de rating
                if "user_id" in data and "movie_id" in data:
                    model_config["endpoint"] = "http://localhost:8000/movies/models/movies/predict-rating"
            
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
    - Si es 'time_series_prediction' (predicción temporal): Explica las tendencias, fechas específicas, valores predichos y intervalos de confianza
    - Si es 'flight_prediction' (predicción de vuelos): Explica el retraso esperado, factores que influyen, nivel de confianza y recomendaciones
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
        if response_type == "flight_prediction" and modelo == "flights":
            # Nuevo formato para predicción de vuelos
            if "prediction" in result:
                delay_minutes = result.get("prediction", 0)
                confidence = result.get("confidence", 0)
                flight_info = result.get("flight_info", {})
                
                route = flight_info.get("route", "Vuelo")
                airline = flight_info.get("airline", "")
                departure = flight_info.get("departure", "")
                
                if delay_minutes <= 5:
                    status_emoji = "✅"
                    status = "puntual"
                elif delay_minutes <= 15:
                    status_emoji = "🟡"
                    status = "retraso leve"
                elif delay_minutes <= 30:
                    status_emoji = "🟠"
                    status = "retraso moderado"
                else:
                    status_emoji = "🔴"
                    status = "retraso significativo"
                
                response = f"{status_emoji} Predicción vuelo {airline} {route}:\n"
                response += f"🕐 Retraso esperado: {delay_minutes:.0f} minutos ({status})\n"
                response += f"📅 Salida: {departure}\n"
                response += f"🎯 Confianza: {confidence:.1f}%"
                return response
                
        elif response_type == "time_series_prediction" and modelo == "bitcoin":
            # Nuevo formato para el modelo Prophet de Bitcoin
            if "predictions" in result:
                predictions = result.get("predictions", [])
                if predictions:
                    # Mostrar las primeras 3 predicciones
                    preview = predictions[:3]
                    formatted_preds = []
                    for pred in preview:
                        date = pred.get("date", "Fecha desconocida")
                        price = pred.get("predicted_price", 0)
                        formatted_preds.append(f"{date}: ${price:,.2f}")
                    
                    total_days = len(predictions)
                    confidence = result.get("confidence", 0)
                    model_type = result.get("model_info", {}).get("model_type", "Prophet")
                    
                    response = f"📈 Predicciones Bitcoin ({model_type}):\n"
                    response += "\n".join(formatted_preds)
                    if total_days > 3:
                        response += f"\n... y {total_days - 3} días más"
                    response += f"\n\n🎯 Confianza del modelo: {confidence:.1f}%"
                    return response
            
            elif "prediction" in result:
                # Formato de respaldo para predicción única
                prediction = result.get("prediction", 0)
                confidence = result.get("confidence", 0)
                return f"💰 Predicción de Bitcoin: ${prediction:,.2f} USD (Confianza: {confidence:.1f}%)"
            
            elif modelo == "properties" and "prediction" in result:
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
