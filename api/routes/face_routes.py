# api/routes/face_routes.py
from fastapi import APIRouter, File, UploadFile, HTTPException
from io import BytesIO
import os
from datetime import datetime
from pathlib import Path

# Cambiar de Google Vision a tu modelo Keras personalizado
from face_recognition.keras_emotion_service import detect_faces_with_keras

router = APIRouter(prefix="/face", tags=["Face Recognition"])

def save_captured_image(image_data: bytes, emotion_result: str = "unknown") -> str:
    """
    Guarda la imagen capturada en la carpeta data con timestamp y emoción detectada
    """
    try:
        # Crear carpeta data/captures si no existe
        data_dir = Path("data/captures")
        data_dir.mkdir(exist_ok=True)
        
        # Generar nombre de archivo único
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # microsegundos cortados
        filename = f"{timestamp}_{emotion_result}.jpg"
        filepath = data_dir / filename
        
        # Guardar imagen
        with open(filepath, "wb") as f:
            f.write(image_data)
        
        return str(filepath)
    except Exception as e:
        print(f"⚠️  Error guardando imagen: {e}")
        return ""

@router.post("/analyze")
async def analyze_face(file: UploadFile = File(...)):
    try:
        name = file.filename.lower()
        if not name.endswith((".jpg", ".jpeg", ".png")):
            raise HTTPException(status_code=400, detail="Formato no soportado (usa .jpg/.jpeg/.png)")
        
        # Leer los datos de la imagen
        image_data = await file.read()
        image_stream = BytesIO(image_data)

        # Usar tu modelo Keras entrenado en lugar de Google Vision
        results = detect_faces_with_keras(image_stream)

        if len(results) and isinstance(results[0], dict) and "error" in results[0]:
            raise HTTPException(status_code=502, detail=results[0]["error"])

        # Determinar la emoción detectada para el nombre del archivo
        detected_emotion = "no_face"
        if len(results) > 0 and "best_emotion" in results[0]:
            detected_emotion = results[0]["best_emotion"]["label"]
        
        # Guardar la imagen capturada
        saved_path = save_captured_image(image_data, detected_emotion)
        
        return {
            "faces": results,
            "meta": {
                "source": "keras_custom_model",
                "notes": "Detección de rostro y emoción con modelo MobileNetV3 personalizado (FER2013).",
                "emotions": ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
                "saved_image": saved_path if saved_path else None,
                "timestamp": datetime.now().isoformat()
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {e}")