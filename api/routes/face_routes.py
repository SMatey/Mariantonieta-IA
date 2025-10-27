# api/routes/face_routes.py
from fastapi import APIRouter, File, UploadFile, HTTPException
from io import BytesIO

from face_recognition.google_vision_service import detect_faces_with_google

router = APIRouter(prefix="/face", tags=["Face Recognition"]) 
@router.post("/analyze")
async def analyze_face(file: UploadFile = File(...)):
    try:
        name = file.filename.lower()
        if not name.endswith((".jpg", ".jpeg", ".png")):
            raise HTTPException(status_code=400, detail="Formato no soportado (usa .jpg/.jpeg/.png)")
        
        image_stream = BytesIO(await file.read())

        results = detect_faces_with_google(image_stream)

        if len(results) and isinstance(results[0], dict) and "error" in results[0]:
            raise HTTPException(status_code=502, detail=results[0]["error"])

        return {
            "faces": results,
            "meta": {
                "source": "google_vision",
                "notes": "Detección de rostro y emoción con Google Vision."
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {e}")