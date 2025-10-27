# azure_face_service.py  (reemplazo sugerido)
import os, json, io
from dotenv import load_dotenv
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face import FaceClient

load_dotenv()
AZURE_FACE_ENDPOINT = (os.getenv("AZURE_FACE_ENDPOINT") or "").rstrip("/")  # quita barra final
AZURE_FACE_KEY = os.getenv("AZURE_FACE_KEY")

if not AZURE_FACE_ENDPOINT or not AZURE_FACE_KEY:
    raise RuntimeError("Faltan AZURE_FACE_ENDPOINT o AZURE_FACE_KEY en .env")

credentials = CognitiveServicesCredentials(AZURE_FACE_KEY)
face_client = FaceClient(AZURE_FACE_ENDPOINT, credentials)

def _azure_err_text(e: Exception) -> str:
    try:
        return json.loads(e.response.text).get("error", {}).get("message") or str(e)
    except Exception:
        return str(e)

def detect_face_with_azure(image_stream: io.BytesIO):
    try:
        image_stream.seek(0)

        # 1) ¡No enviar return_face_attributes si no los necesitas!
        # 2) Arranca con detection_01 (más compatible) y, si quieres, luego cambias a 03.
        try:
            detected = face_client.face.detect_with_stream(
                image=image_stream,
                detection_model="detection_01"
            )
        except Exception as e:
            # Si tu recurso sí soporta detection_03, puedes intentar el fallback inverso:
            msg = _azure_err_text(e).lower()
            if "detectionmodel" in msg or "invalid" in msg:
                image_stream.seek(0)
                detected = face_client.face.detect_with_stream(
                    image=image_stream,
                    detection_model="detection_03"
                )
            else:
                return {"error": f"Azure Face (detect_with_stream): {_azure_err_text(e)}"}

        if not detected:
            return []

        out = []
        for f in detected:
            out.append({
                "detection_source": "azure",
                "position": f.face_rectangle.as_dict()
            })
        return out

    except Exception as e:
        return {"error": f"Azure Face (detect_with_stream): {_azure_err_text(e)}"}
