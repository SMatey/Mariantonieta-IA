from langchain_community.llms import Ollama

def extract_bitcoin_parameters(query: str, llm: Ollama):
    """
    Extrae parámetros para el modelo Prophet de Bitcoin (series de tiempo)
    """
    extraction_prompt = f"""
    Extrae información específica para predicción de Bitcoin usando Prophet del siguiente texto:
    
    "{query}"
    
    Busca y extrae SOLO los valores que se mencionen explícitamente:
    - Fechas específicas para predicción (ej: "precio para 2025-01-15", "predice el 25 de diciembre", "qué precio tendrá el 1 de enero")
    - Rango de fechas (ej: "próxima semana", "próximos 30 días", "siguiente mes")
    
    Si se menciona una fecha específica, conviértela a formato YYYY-MM-DD.
    Si se menciona un rango relativo, calcula las fechas correspondientes desde hoy (2025-10-24).
    
    Responde SOLO en formato JSON válido:
    {{
        "dates": ["2025-01-15", "2025-01-16"],
        "query": "predicción de precio de Bitcoin para enero 2025"
    }}
    """

    try:
        extraction_result = llm.invoke(extraction_prompt)
        import json
        import re
        from datetime import datetime, timedelta
        
        # Limpiar la respuesta para extraer solo el JSON
        json_match = re.search(r'\{.*\}', extraction_result, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            extracted_params = json.loads(json_str)
            
            # Validar que las fechas estén en formato correcto
            dates = extracted_params.get("dates", [])
            if not dates:
                # Generar fechas por defecto (próximos 7 días)
                today = datetime.now()
                dates = [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 8)]
                extracted_params["dates"] = dates
                
            return extracted_params
        else:
            # Respaldo: próximos 7 días
            today = datetime.now()
            dates = [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 8)]
            return {"dates": dates, "query": query}
            
    except Exception as e:
        print(f"Error extrayendo parámetros de Bitcoin: {e}")
        # Respaldo: próximos 7 días
        from datetime import datetime, timedelta
        today = datetime.now()
        dates = [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 8)]
        return {"dates": dates, "query": query}
    

def extract_properties_parameters(query: str, llm: Ollama):
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


def extract_flights_parameters(query: str, llm: Ollama):
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
            
            # Procesar y validar fecha
            if extracted_params.get("date"):
                date_str = extracted_params["date"]
                try:
                    # Manejar fechas relativas
                    if "mañana" in date_str.lower() or "tomorrow" in date_str.lower():
                        tomorrow = datetime.now() + timedelta(days=1)
                        extracted_params["date"] = tomorrow.strftime("%Y-%m-%d")
                    elif "hoy" in date_str.lower() or "today" in date_str.lower():
                        today = datetime.now()
                        extracted_params["date"] = today.strftime("%Y-%m-%d")
                    else:
                        # Validar formato de fecha existente
                        if date_str:
                            # Intentar parsear la fecha para validarla
                            parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
                            # Verificar que no sea una fecha muy antigua o futura
                            current_year = datetime.now().year
                            if parsed_date.year < current_year - 1 or parsed_date.year > current_year + 2:
                                # Usar fecha de hoy si la fecha no es válida
                                today = datetime.now()
                                extracted_params["date"] = today.strftime("%Y-%m-%d")
                            # Si el día es 00, corregirlo a 01
                            if date_str.endswith("-00"):
                                corrected_date = date_str[:-2] + "01"
                                try:
                                    datetime.strptime(corrected_date, "%Y-%m-%d")
                                    extracted_params["date"] = corrected_date
                                except:
                                    today = datetime.now()
                                    extracted_params["date"] = today.strftime("%Y-%m-%d")
                except (ValueError, AttributeError):
                    # Si hay error parseando la fecha, usar hoy
                    today = datetime.now()
                    extracted_params["date"] = today.strftime("%Y-%m-%d")
            else:
                # Si no hay fecha especificada, usar hoy por defecto
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
    

def extract_movies_parameters(query: str, llm: Ollama):
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


def extract_acv_parameters(query: str, llm: Ollama):
    """
    Extrae parámetros para evaluación de riesgo de ACV (Accidente Cerebrovascular)
    """
    extraction_prompt = f"""
    Extrae información médica para evaluación de riesgo de ACV del siguiente texto:
    
    "{query}"
    
    Busca y extrae SOLO los valores mencionados explícitamente:
    - Edad (ej: "65 años", "age 45", "tiene 70")
    - Género (ej: "mujer", "hombre", "male", "female", "masculino", "femenino")
    - Hipertensión (ej: "tiene hipertensión", "presión alta", "sin hipertensión", "hipertension", "hypertension")
    - Enfermedad cardíaca (ej: "problemas del corazón", "enfermedad cardíaca", "sin problemas cardiacos", "heart disease")
    - Estado civil (ej: "casado", "soltero", "married", "single")
    - Tipo de trabajo (ej: "sector privado", "autónomo", "gobierno", "empleado", "private", "self-employed", "government")
    - Tipo de residencia (ej: "urbano", "rural", "ciudad", "campo", "urban", "rural")
    - Nivel de glucosa (ej: "glucosa 120", "azúcar en sangre 95", "glucose level 110")
    - IMC/BMI (ej: "BMI 28", "índice masa corporal 25", "sobrepeso", "obeso")
    - Estado de fumador (ej: "fumador", "no fuma", "ex fumador", "nunca fumó", "smoker", "never smoked", "formerly smoked")
    
    CONVERSIONES IMPORTANTES:
    - Género: "mujer"/"femenino"/"female" → "Female", "hombre"/"masculino"/"male" → "Male"
    - Hipertensión/Enfermedad cardíaca: "sí"/"tiene"/"con" → 1, "no"/"sin" → 0
    - Estado civil: "casado"/"married" → "Yes", "soltero"/"single" → "No"
    - Trabajo: "privado"/"empleado" → "Private", "autónomo" → "Self-employed", "gobierno" → "Govt_job"
    - Residencia: "ciudad"/"urbano" → "Urban", "campo"/"rural" → "Rural"
    - Fumador: "fuma"/"fumador" → "smokes", "no fuma"/"nunca fumó" → "never smoked", "ex fumador"/"antes fumaba" → "formerly smoked"
    
    Responde SOLO en formato JSON válido:
    {{
        "age": 65.0,
        "gender": "Female",
        "hypertension": 1,
        "heart_disease": 0,
        "ever_married": "Yes",
        "work_type": "Private",
        "residence_type": "Urban",
        "avg_glucose_level": 120.0,
        "bmi": 28.5,
        "smoking_status": "never smoked"
    }}
    
    Si NO encuentras un valor específico, usa null.
    NO incluyas campos con valores null en la respuesta final.
    """
    
    try:
        extraction_result = llm.invoke(extraction_prompt)
        import json
        import re
        
        json_match = re.search(r'\{.*\}', extraction_result, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            extracted_params = json.loads(json_str)
            
            # Validar y convertir valores específicos
            if "hypertension" in extracted_params:
                hyp_value = extracted_params["hypertension"]
                if isinstance(hyp_value, str):
                    if any(word in hyp_value.lower() for word in ["sí", "si", "tiene", "con", "yes", "1"]):
                        extracted_params["hypertension"] = 1
                    elif any(word in hyp_value.lower() for word in ["no", "sin", "0"]):
                        extracted_params["hypertension"] = 0
                    else:
                        del extracted_params["hypertension"]
                elif hyp_value not in [0, 1]:
                    del extracted_params["hypertension"]
            
            if "heart_disease" in extracted_params:
                hd_value = extracted_params["heart_disease"]
                if isinstance(hd_value, str):
                    if any(word in hd_value.lower() for word in ["sí", "si", "tiene", "con", "yes", "1"]):
                        extracted_params["heart_disease"] = 1
                    elif any(word in hd_value.lower() for word in ["no", "sin", "0"]):
                        extracted_params["heart_disease"] = 0
                    else:
                        del extracted_params["heart_disease"]
                elif hd_value not in [0, 1]:
                    del extracted_params["heart_disease"]
            
            # Validar valores numéricos
            if "age" in extracted_params:
                age_value = extracted_params["age"]
                if isinstance(age_value, str):
                    numbers = re.findall(r'\d+', age_value)
                    if numbers:
                        age = float(numbers[0])
                        if 0 <= age <= 120:
                            extracted_params["age"] = age
                        else:
                            del extracted_params["age"]
                    else:
                        del extracted_params["age"]
                elif not isinstance(age_value, (int, float)) or not (0 <= age_value <= 120):
                    del extracted_params["age"]
            
            if "avg_glucose_level" in extracted_params:
                glucose_value = extracted_params["avg_glucose_level"]
                if isinstance(glucose_value, str):
                    numbers = re.findall(r'\d+\.?\d*', glucose_value)
                    if numbers:
                        glucose = float(numbers[0])
                        if 50 <= glucose <= 400:  # Rango médicamente válido
                            extracted_params["avg_glucose_level"] = glucose
                        else:
                            del extracted_params["avg_glucose_level"]
                    else:
                        del extracted_params["avg_glucose_level"]
                elif not isinstance(glucose_value, (int, float)) or not (50 <= glucose_value <= 400):
                    del extracted_params["avg_glucose_level"]
            
            if "bmi" in extracted_params:
                bmi_value = extracted_params["bmi"]
                if isinstance(bmi_value, str):
                    numbers = re.findall(r'\d+\.?\d*', bmi_value)
                    if numbers:
                        bmi = float(numbers[0])
                        if 10 <= bmi <= 60:  # Rango médicamente válido
                            extracted_params["bmi"] = bmi
                        else:
                            del extracted_params["bmi"]
                    else:
                        del extracted_params["bmi"]
                elif not isinstance(bmi_value, (int, float)) or not (10 <= bmi_value <= 60):
                    del extracted_params["bmi"]
            
            # Filtrar valores null y vacíos
            filtered_params = {k: v for k, v in extracted_params.items() 
                             if v is not None and v != "" and v != "null"}
            return filtered_params
        else:
            return {}
    except Exception as e:
        print(f"Error extrayendo parámetros de ACV: {e}")
        return {}