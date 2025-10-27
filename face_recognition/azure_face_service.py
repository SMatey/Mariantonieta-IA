import os
from dotenv import load_dotenv
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face import FaceClient

# Cargar variables de entorno
load_dotenv()
AZURE_FACE_ENDPOINT = os.getenv("AZURE_FACE_ENDPOINT")
AZURE_FACE_KEY = os.getenv("AZURE_FACE_KEY")

# Autenticar el cliente
credentials = CognitiveServicesCredentials(AZURE_FACE_KEY)
face_client = FaceClient(AZURE_FACE_ENDPOINT, credentials)

def detect_face_with_azure(image_stream):
    """
    Detecta la POSICIÓN de los rostros en una imagen usando Azure.
    Ya no detecta emoción.
    
    Args:
        image_stream: Un stream de archivo (ej. image.file)
    """
    
    # --- MODIFICACIÓN CLAVE ---
    # Ya no pedimos 'emotion'. Si la lista está vacía, solo devuelve
    # el 'faceId' y 'faceRectangle' (la posición).
    face_attributes = [] 
    
    try:
        # 1. Leemos el stream completo a 'bytes'
        image_data = image_stream.read()
        
        if not image_data:
            return {"error": "La imagen recibida para Azure está vacía."}

        # 2. Usamos 'detect_with_data' para evitar errores de stream
        detected_faces = face_client.face.detect_with_data(
            image=image_data,
            return_face_attributes=face_attributes,
            detection_model='detection_03'
        )
        
        # --- FIN DE LA MODIFICACIÓN ---

        if not detected_faces:
            return []

        # Procesar los resultados (solo queremos la posición)
        results = []
        for face in detected_faces:
            results.append({
                "detection_source": "azure",
                "position": face.face_rectangle.as_dict()
            })
            
        return results

    except Exception as e:
        print(f"Error al analizar con Azure: {e}")
        return {"error": str(e)}