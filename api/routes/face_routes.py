from fastapi import APIRouter, UploadFile, File, HTTPException
import io
from face_recognition.azure_face_service import detect_face_with_azure
from face_recognition.fer_service import analyze_emotion_with_fer

router = APIRouter(
    prefix="/face",
    tags=["Face Recognition & Emotion"]
)

@router.post("/analyze-emotion")
async def handle_analyze_emotion(image: UploadFile = File(...)):
    """
    Endpoint híbrido:
    1. Detecta rostro con Azure (Requisito)
    2. Detecta emoción con FER (Local)
    """
    
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo no es una imagen.")

    try:
        # --- MODIFICACIÓN CLAVE ---
        # Leemos la imagen UNA SOLA VEZ en memoria
        image_data = await image.read()
        
        # Creamos dos "copias" del stream en memoria para que
        # cada servicio pueda leerlo de forma independiente.
        azure_stream = io.BytesIO(image_data)
        fer_stream = io.BytesIO(image_data)
        
        # 1. Llamar a Azure para la detección de rostro (cumple requisito)
        azure_results = detect_face_with_azure(azure_stream)
        
        # 2. Llamar a FER para el análisis de emoción
        fer_results = analyze_emotion_with_fer(fer_stream)
        
        # --- FIN DE LA MODIFICACIÓN ---
        
        if "error" in azure_results:
             raise HTTPException(status_code=500, detail=f"Error Azure: {azure_results['error']}")
        
        if "error" in fer_results:
             raise HTTPException(status_code=500, detail=f"Error FER: {fer_results['error']}")

        if not azure_results and not fer_results:
            return {"message": "No se detectaron rostros por ningún método."}

        # Devolvemos los resultados de ambos análisis
        return {
            "azure_face_detection": azure_results,
            "local_emotion_analysis": fer_results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")