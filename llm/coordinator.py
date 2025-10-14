from langchain_community.llms import Ollama
import requests
import json

llm = Ollama(model="llama3")

# Diccionario de modelos disponibles
ENDPOINTS = {
    "bitcoin": "http://localhost:8000/bitcoin/models/bitcoin/predict",
}

def interpretar_y_ejecutar(query: str):
    """
    Coordinador principal que decide qué modelo usar y ejecuta la consulta
    """
    # Paso 1: el LLM decide qué modelo usar
    decision_prompt = f"""
    Eres un coordinador de modelos de IA. Analiza la siguiente consulta y decide qué modelo usar.

    Consulta: "{query}"

    Modelos disponibles:
    - bitcoin: Para predicciones de precios de Bitcoin, criptomonedas, análisis financiero
    - wine: Para clasificación de vinos (no disponible aún)
    - churn: Para predicción de abandono de clientes (no disponible aún)
    - movies: Para recomendaciones de películas (no disponible aún)
    - emotions: Para análisis de emociones (no disponible aún)

    Responde SOLO con el nombre del modelo más apropiado (bitcoin, wine, churn, movies, emotions).
    Si no hay un modelo apropiado, responde "ninguno".
    """
    
    decision = llm.invoke(decision_prompt)
    modelo = decision.strip().lower()

    # Paso 2: llama al modelo correspondiente
    if modelo in ENDPOINTS:
        try:
            data = {"query": query}
            response = requests.post(ENDPOINTS[modelo], json=data, timeout=30)
            
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
            return "Lo siento, no tengo un modelo específico para responder a esa consulta. Actualmente solo puedo ayudarte con predicciones de precios de Bitcoin."
        else:
            return f"El modelo '{modelo}' no está disponible actualmente. Solo tengo disponible el modelo de Bitcoin para predicciones de precios de criptomonedas."

    # Paso 3: interpreta el resultado con el LLM
    interpretation_prompt = f"""
    Un modelo de {modelo} devolvió este resultado para la consulta "{query}":

    Resultado: {json.dumps(result, indent=2)}

    Tu tarea es interpretar este resultado y explicárselo al usuario de forma natural, clara y útil.

    Instrucciones:
    1. Si es una predicción de Bitcoin, incluye el precio predicho, la tendencia y el nivel de confianza
    2. Explica qué significa el resultado en términos simples
    3. Menciona cualquier limitación o consideración importante
    4. Sé conciso pero informativo
    5. Usa emojis apropiados para hacer la respuesta más amigable

    Respuesta:
    """

    try:
        explicacion = llm.invoke(interpretation_prompt)
        return explicacion
    except Exception as e:
        # Si falla la interpretación, devolver el resultado raw de forma más amigable
        if modelo == "bitcoin" and "prediction" in result:
            prediction = result.get("prediction", 0)
            confidence = result.get("confidence", 0)
            return f"💰 Predicción de Bitcoin: ${prediction:,.2f} USD (Confianza: {confidence:.1f}%)"
        else:
            return f"Resultado del modelo {modelo}: {result}"
