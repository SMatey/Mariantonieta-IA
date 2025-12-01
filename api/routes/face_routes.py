# api/routes/face_routes.py
from fastapi import APIRouter, File, UploadFile, HTTPException
from io import BytesIO

# Cambiar de Google Vision a tu modelo Keras personalizado
from face_recognition.keras_emotion_service import detect_faces_with_keras

router = APIRouter(prefix="/face", tags=["Face Recognition"]) 
@router.post("/analyze")
async def analyze_face(file: UploadFile = File(...)):
    try:
        name = file.filename.lower()
        if not name.endswith((".jpg", ".jpeg", ".png")):
            raise HTTPException(status_code=400, detail="Formato no soportado (usa .jpg/.jpeg/.png)")
        
        image_stream = BytesIO(await file.read())

        # Usar tu modelo Keras entrenado en lugar de Google Vision
        results = detect_faces_with_keras(image_stream)

        if len(results) and isinstance(results[0], dict) and "error" in results[0]:
            raise HTTPException(status_code=502, detail=results[0]["error"])

        return {
            "faces": results,
            "meta": {
                "source": "keras_custom_model",
                "notes": "Detección de rostro y emoción con modelo personalizado EfficientNetB3.",
                "emotions": ["interested", "neutral", "disappointed"]
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {e}")