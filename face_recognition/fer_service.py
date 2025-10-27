import cv2
import numpy as np
from fer import FER

# Inicializamos el detector de FER
try:
    print("Cargando modelo FER (esto puede tardar la primera vez)...")
    detector = FER(mtcnn=True)
    print("Modelo FER cargado.")
except Exception as e:
    print(f"Error cargando modelo FER: {e}")
    detector = None

def analyze_emotion_with_fer(image_stream):
    """
    Analiza una imagen (en formato stream) y detecta rostros y emociones
    usando la librería local FER.
    """
    if detector is None:
        return {"error": "El detector FER no pudo inicializarse."}
    
    try:
        # 1. Leer el stream de la imagen en un buffer de numpy
        file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
        
        # 2. Decodificar el buffer en una imagen de OpenCV
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return {"error": "No se pudo decodificar la imagen para FER."}

        # 3. Detectar emociones
        detected_faces = detector.detect_emotions(image)
        
        if not detected_faces:
            return []

        # 4. Formatear la salida
        results = []
        for face in detected_faces:
            box = face["box"]
            emotions = face["emotions"]
            main_emotion = max(emotions, key=emotions.get)
            
            results.append({
                "detection_source": "fer_local",
                "main_emotion": main_emotion,
                "score": emotions[main_emotion],
                "position": {
                    "left": box[0],
                    "top": box[1],
                    "width": box[2],
                    "height": box[3]
                },
                "all_emotions": emotions
            })
            
        return results

    except Exception as e:
        print(f"Error al analizar con FER: {e}")
        return {"error": str(e)}