import cv2
import numpy as np
from deepface import DeepFace

try:
    print("Cargando modelo de emoción de DeepFace (puede tardar)...")
    DeepFace.analyze(
        np.zeros((100, 100, 3), dtype=np.uint8), 
        actions=['emotion'], 
        enforce_detection=False
    )
    print("Modelo DeepFace cargado.")
    DEEPFACE_LOADED = True
except Exception as e:
    print(f"Error cargando modelo DeepFace: {e}")
    DEEPFACE_LOADED = False

def analyze_emotion_with_deepface(image_stream):
    """
    Analiza una imagen (en formato stream) y detecta rostros y emociones
    usando la librería local DeepFace.
    """
    if not DEEPFACE_LOADED:
        return {"error": "El detector DeepFace no pudo inicializarse."}
    
    try:
        # 1. Leer el stream en un buffer de numpy
        file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
        
        # 2. Decodificar en una imagen de OpenCV (DeepFace lo prefiere en BGR)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return {"error": "No se pudo decodificar la imagen para DeepFace."}

        # 3. Analizar la imagen
        # 'actions' le dice que solo busque emociones
        # 'enforce_detection=False' evita que lance un error si no hay cara
        detected_faces = DeepFace.analyze(
            image, 
            actions=['emotion'], 
            enforce_detection=False
        )
        
        # DeepFace devuelve una lista de dicts, uno por cara
        if not detected_faces or not detected_faces[0]['dominant_emotion']:
             return []

        # 4. Formatear la salida para que sea consistente
        results = []
        for face in detected_faces:
            region = face['region']
            emotions = face['emotion']
            main_emotion = face['dominant_emotion']
            
            results.append({
                "detection_source": "deepface_local",
                "main_emotion": main_emotion,
                "score": emotions[main_emotion],
                "position": {
                    "left": region['x'],
                    "top": region['y'],
                    "width": region['w'],
                    "height": region['h']
                },
                "all_emotions": emotions
            })
            
        return results

    except Exception as e:
        # DeepFace lanza una excepción si no hay rostro y enforce_detection=True
        # Con enforce_detection=False, no debería, pero lo capturamos por si acaso.
        if "Face could not be detected" in str(e):
            return []
        print(f"Error al analizar con DeepFace: {e}")
        return {"error": str(e)}