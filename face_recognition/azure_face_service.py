# azure_face_service.py
import os, json, io
from dotenv import load_dotenv
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face import FaceClient

load_dotenv()
AZURE_FACE_ENDPOINT = (os.getenv("AZURE_FACE_ENDPOINT") or "").rstrip("/")
AZURE_FACE_KEY = os.getenv("AZURE_FACE_KEY")
if not AZURE_FACE_ENDPOINT or not AZURE_FACE_KEY:
    raise RuntimeError("Faltan AZURE_FACE_ENDPOINT o AZURE_FACE_KEY")

credentials = CognitiveServicesCredentials(AZURE_FACE_KEY)
face_client = FaceClient(AZURE_FACE_ENDPOINT, credentials)

def _azure_err_text(e: Exception) -> str:
    # Esta función de ayuda está bien
    try:
        return json.loads(e.response.text).get("error", {}).get("message") or str(e)
    except Exception:
        return str(e)

def detect_face_with_azure(image_stream: io.BytesIO):
    """
    Detecta la posición de los rostros usando Azure.
    Recibe un stream io.BytesIO que viene de face_routes.py.
    """
    try:
        # Aseguramos que el puntero esté al inicio (byte 0)
        image_stream.seek(0) 
        
        # --- MODIFICACIÓN CLAVE ---
        # Eliminamos el 'try...except' y la llamada al obsoleto 'detection_01'.
        # Llamamos directamente al modelo moderno 'detection_03'.
        detected = face_client.face.detect_with_stream(
            image=image_stream,
            detection_model="detection_03" # Usar solo el modelo 03
        )
        # --- FIN DE LA MODIFICACIÓN ---

        out = []
        for f in detected or []:
            r = f.face_rectangle  # top,left,width,height
            out.append({
                "detection_source": "azure",
                "position": {"top": r.top, "left": r.left, "width": r.width, "height": r.height}
            })
        return out
        
    except Exception as e:
        # Si 'detection_03' falla, ahora sí es un error real.
        return {"error": f"Azure Face (detect_with_stream): {_azure_err_text(e)}"}