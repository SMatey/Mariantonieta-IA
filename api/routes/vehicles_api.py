from fastapi import APIRouter, File, UploadFile, HTTPException
from api.models.vehicles_api_models import (
    ImageAnalysisResponse,
    VideoAnalysisResponse
)
from vehicles.vehicles_service import get_vehicle_service
from pathlib import Path
import time

router = APIRouter(prefix="/vehicles", tags=["Vehicle Detection"])

@router.post("/analyze-image", response_model=ImageAnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    """
    Detecta vehículos en una imagen
    
    Formatos soportados: .jpg, .jpeg, .png
    
    Returns:
        - detections: Lista de vehículos detectados con sus coordenadas y confianza
        - total_vehicles: Número total de vehículos detectados
        - vehicle_counts: Conteo por tipo de vehículo
        - processing_time_ms: Tiempo de procesamiento en milisegundos
        - saved_image: Ruta de la imagen anotada guardada
    """
    try:
        # Validar formato
        filename = file.filename.lower()
        if not filename.endswith((".jpg", ".jpeg", ".png")):
            raise HTTPException(
                status_code=400, 
                detail="Formato no soportado. Usa .jpg, .jpeg o .png"
            )
        
        # Leer imagen
        image_data = await file.read()
        
        # Obtener servicio
        service = get_vehicle_service()
        
        # Detectar vehículos
        result = service.detect_vehicles_in_image(image_data)
        
        # Guardar imagen anotada
        saved_path = None
        if result["detections"]:
            try:
                saved_path = service.save_annotated_image(image_data, result["detections"])
            except Exception as e:
                print(f"⚠️ No se pudo guardar la imagen anotada: {e}")
        
        return ImageAnalysisResponse(
            detections=result["detections"],
            total_vehicles=result["total_vehicles"],
            vehicle_counts=result["vehicle_counts"],
            processing_time_ms=result["processing_time_ms"],
            image_size=result["image_size"],
            saved_image=saved_path
        )
        
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Modelo no disponible: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando imagen: {str(e)}"
        )


@router.post("/analyze-video", response_model=VideoAnalysisResponse)
async def analyze_video(
    file: UploadFile = File(...),
    max_frames: int = 30
):
    """
    Detecta vehículos en un video
    
    Formatos soportados: .mp4, .avi, .mov
    
    Args:
        file: Archivo de video
        max_frames: Máximo número de frames a procesar (default: 30)
    
    Returns:
        - total_frames: Total de frames en el video
        - processed_frames: Frames procesados
        - detections_per_frame: Detecciones en cada frame procesado
        - average_vehicles_per_frame: Promedio de vehículos por frame
        - total_unique_detections: Total de detecciones únicas
        - vehicle_counts: Conteo por tipo de vehículo
        - processing_time_ms: Tiempo de procesamiento
    """
    try:
        # Validar formato
        filename = file.filename.lower()
        if not filename.endswith((".mp4", ".avi", ".mov")):
            raise HTTPException(
                status_code=400,
                detail="Formato no soportado. Usa .mp4, .avi o .mov"
            )
        
        # Validar max_frames
        if max_frames < 1 or max_frames > 100:
            raise HTTPException(
                status_code=400,
                detail="max_frames debe estar entre 1 y 100"
            )
        
        # Leer video
        video_data = await file.read()
        
        # Obtener servicio
        service = get_vehicle_service()
        
        # Detectar vehículos
        result = service.detect_vehicles_in_video(video_data, max_frames)
        
        return VideoAnalysisResponse(
            total_frames=result["total_frames"],
            processed_frames=result["processed_frames"],
            detections_per_frame=result["detections_per_frame"],
            average_vehicles_per_frame=result["average_vehicles_per_frame"],
            total_unique_detections=result["total_unique_detections"],
            vehicle_counts=result["vehicle_counts"],
            processing_time_ms=result["processing_time_ms"]
        )
        
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Modelo no disponible: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando video: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """
    Verifica el estado del servicio de detección de vehículos
    """
    try:
        service = get_vehicle_service()
        return {
            "status": "healthy",
            "service": "Vehicle Detection",
            "model": "YOLOv8",
            "model_path": service.model_path,
            "model_loaded": service.model is not None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "Vehicle Detection",
            "error": str(e)
        }


@router.get("/info")
async def model_info():
    """
    Información sobre el modelo de detección de vehículos
    """
    return {
        "model_name": "YOLOv8 Vehicle Detection",
        "description": "Modelo entrenado para detectar diferentes tipos de vehículos",
        "supported_vehicles": [
            "car", "truck", "bus", "motorcycle"
        ],
        "input_formats": {
            "images": [".jpg", ".jpeg", ".png"],
            "videos": [".mp4", ".avi", ".mov"]
        },
        "endpoints": {
            "analyze_image": "/vehicles/analyze-image",
            "analyze_video": "/vehicles/analyze-video",
            "health": "/vehicles/health",
            "info": "/vehicles/info"
        },
        "parameters": {
            "confidence_threshold": 0.25,
            "iou_threshold": 0.45,
            "max_frames_video": "1-100 (default: 30)"
        }
    }
