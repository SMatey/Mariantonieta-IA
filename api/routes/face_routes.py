from fastapi import APIRouter, UploadFile, File, HTTPException
import io

# Importamos AMBOS servicios
from face_recognition.azure_face_service import detect_face_with_azure
from face_recognition.deepface_service import analyze_emotion_with_deepface

router = APIRouter(
    prefix="/face",
    tags=["Face Recognition & Emotion"]
)

@router.post("/analyze-emotion")
async def handle_analyze_emotion(image: UploadFile = File(...)):
    """
    Endpoint híbrido:
    1. Detecta rostro con Azure (Requisito)
    2. Detecta emoción con DeepFace (Local)
    """
    
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo no es una imagen.")

    try:
        # Leemos la imagen UNA SOLA VEZ en memoria
        image_data = await image.read()
        
        # Creamos dos "copias" del stream en memoria para que
        # cada servicio pueda leerlo de forma independiente.
        azure_stream = io.BytesIO(image_data)
        local_stream = io.BytesIO(image_data)
        
        # 1. Llamar a Azure para la detección de rostro (cumple requisito)
        azure_results = detect_face_with_azure(azure_stream)
        
        # 2. Llamar a DeepFace para el análisis de emoción
        local_results = analyze_emotion_with_deepface(local_stream)
        
        
        if "error" in azure_results:
             raise HTTPException(status_code=500, detail=f"Error Azure: {azure_results['error']}")
        
        if "error" in local_results:
             raise HTTPException(status_code=500, detail=f"Error DeepFace: {local_results['error']}")

        if not azure_results and not local_results:
            return {"message": "No se detectaron rostros por ningún método."}

        # Devolvemos los resultados de ambos análisis
        return {
            "azure_face_detection": azure_results,
            "local_emotion_analysis": local_results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")