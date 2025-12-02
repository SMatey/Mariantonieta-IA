from pydantic import BaseModel
from typing import List, Optional

class BoundingBox(BaseModel):
    """Coordenadas de la caja delimitadora del vehículo detectado"""
    x1: float
    y1: float
    x2: float
    y2: float
    
class VehicleDetection(BaseModel):
    """Información de un vehículo detectado"""
    class_name: str
    confidence: float
    bounding_box: BoundingBox

class ImageAnalysisResponse(BaseModel):
    """Respuesta del análisis de imagen"""
    detections: List[VehicleDetection]
    total_vehicles: int
    vehicle_counts: dict
    processing_time_ms: float
    image_size: dict
    saved_image: Optional[str] = None

class VideoAnalysisResponse(BaseModel):
    """Respuesta del análisis de video"""
    total_frames: int
    processed_frames: int
    detections_per_frame: List[dict]
    average_vehicles_per_frame: float
    total_unique_detections: int
    vehicle_counts: dict
    processing_time_ms: float
    saved_video: Optional[str] = None

class VideoFrameDetection(BaseModel):
    """Detecciones en un frame específico del video"""
    frame_number: int
    timestamp_seconds: float
    detections: List[VehicleDetection]
    vehicle_count: int
