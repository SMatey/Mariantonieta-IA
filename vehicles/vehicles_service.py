import pickle
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union
from io import BytesIO
import time
from datetime import datetime

class VehicleDetectionService:
    """Servicio para detección de vehículos usando YOLO"""
    
    def __init__(self, model_path: str = None):
        """
        Inicializa el servicio de detección
        
        Args:
            model_path: Ruta al archivo modelo_vehiculos.pkl
        """
        self.model = None
        self.model_path = model_path or "dl_models/modelo_vehiculos.pkl"
        self._load_model()
        
    def _load_model(self):
        """Carga el modelo YOLO desde el archivo pickle"""
        try:
            model_file = Path(self.model_path)
            if not model_file.exists():
                raise FileNotFoundError(
                    f"Modelo no encontrado en {self.model_path}. "
                    f"Por favor coloca el archivo modelo_vehiculos.pkl en la carpeta data/"
                )
            
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            print(f"✅ Modelo de vehículos cargado desde {self.model_path}")
        except Exception as e:
            print(f"❌ Error cargando modelo de vehículos: {e}")
            raise
    
    def detect_vehicles_in_image(self, image_data: Union[bytes, BytesIO, np.ndarray]) -> dict:
        """
        Detecta vehículos en una imagen
        
        Args:
            image_data: Datos de la imagen (bytes, BytesIO o numpy array)
            
        Returns:
            Dict con detecciones y metadatos
        """
        start_time = time.time()
        
        # Convertir datos a numpy array si es necesario
        if isinstance(image_data, bytes):
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image_data, BytesIO):
            image_data.seek(0)
            nparr = np.frombuffer(image_data.read(), np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            image = image_data
        
        if image is None:
            raise ValueError("No se pudo leer la imagen")
        
        # Realizar detección con YOLO
        results = self.model.predict(image, conf=0.25, iou=0.45)
        
        # Procesar resultados
        detections = []
        vehicle_counts = {}
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extraer información de la detección
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = result.names[class_id]
                
                # Contar por tipo de vehículo
                vehicle_counts[class_name] = vehicle_counts.get(class_name, 0) + 1
                
                detections.append({
                    "class_name": class_name,
                    "confidence": confidence,
                    "bounding_box": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2)
                    }
                })
        
        processing_time = (time.time() - start_time) * 1000  # en ms
        
        return {
            "detections": detections,
            "total_vehicles": len(detections),
            "vehicle_counts": vehicle_counts,
            "processing_time_ms": processing_time,
            "image_size": {
                "width": image.shape[1],
                "height": image.shape[0]
            }
        }
    
    def detect_vehicles_in_video(self, video_data: bytes, max_frames: int = 30) -> dict:
        """
        Detecta vehículos en un video
        
        Args:
            video_data: Datos del video en bytes
            max_frames: Máximo número de frames a procesar
            
        Returns:
            Dict con detecciones por frame y estadísticas
        """
        start_time = time.time()
        
        # Guardar video temporalmente
        temp_video_path = f"data/temp_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        Path("data").mkdir(exist_ok=True)
        
        with open(temp_video_path, 'wb') as f:
            f.write(video_data)
        
        # Abrir video
        cap = cv2.VideoCapture(temp_video_path)
        
        if not cap.isOpened():
            raise ValueError("No se pudo abrir el video")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calcular intervalo para procesar max_frames distribuidos uniformemente
        frame_interval = max(1, total_frames // max_frames)
        
        detections_per_frame = []
        all_vehicle_counts = {}
        frame_count = 0
        processed_count = 0
        
        while cap.isOpened() and processed_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Procesar solo frames en el intervalo
            if frame_count % frame_interval == 0:
                # Detectar vehículos en el frame
                results = self.model.predict(frame, conf=0.25, iou=0.45, verbose=False)
                
                frame_detections = []
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = result.names[class_id]
                        
                        # Contar vehículos
                        all_vehicle_counts[class_name] = all_vehicle_counts.get(class_name, 0) + 1
                        
                        frame_detections.append({
                            "class_name": class_name,
                            "confidence": confidence,
                            "bounding_box": {
                                "x1": float(x1),
                                "y1": float(y1),
                                "x2": float(x2),
                                "y2": float(y2)
                            }
                        })
                
                detections_per_frame.append({
                    "frame_number": frame_count,
                    "timestamp_seconds": frame_count / fps if fps > 0 else 0,
                    "detections": frame_detections,
                    "vehicle_count": len(frame_detections)
                })
                
                processed_count += 1
            
            frame_count += 1
        
        cap.release()
        
        # Limpiar archivo temporal
        try:
            Path(temp_video_path).unlink()
        except:
            pass
        
        processing_time = (time.time() - start_time) * 1000
        
        # Calcular estadísticas
        avg_vehicles = sum(d["vehicle_count"] for d in detections_per_frame) / len(detections_per_frame) if detections_per_frame else 0
        
        return {
            "total_frames": total_frames,
            "processed_frames": processed_count,
            "detections_per_frame": detections_per_frame,
            "average_vehicles_per_frame": avg_vehicles,
            "total_unique_detections": sum(all_vehicle_counts.values()),
            "vehicle_counts": all_vehicle_counts,
            "processing_time_ms": processing_time
        }
    
    def draw_detections(self, image: np.ndarray, detections: List[dict]) -> np.ndarray:
        """
        Dibuja las detecciones en la imagen
        
        Args:
            image: Imagen numpy array
            detections: Lista de detecciones
            
        Returns:
            Imagen con detecciones dibujadas
        """
        annotated_image = image.copy()
        
        for det in detections:
            bbox = det["bounding_box"]
            x1, y1 = int(bbox["x1"]), int(bbox["y1"])
            x2, y2 = int(bbox["x2"]), int(bbox["y2"])
            
            # Dibujar rectángulo
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Agregar etiqueta
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.putText(annotated_image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return annotated_image
    
    def save_annotated_image(self, image_data: Union[bytes, BytesIO], detections: List[dict]) -> str:
        """
        Guarda una imagen con las detecciones dibujadas
        
        Args:
            image_data: Datos de la imagen
            detections: Lista de detecciones
            
        Returns:
            Ruta del archivo guardado
        """
        # Convertir a numpy array
        if isinstance(image_data, bytes):
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            image_data.seek(0)
            nparr = np.frombuffer(image_data.read(), np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Dibujar detecciones
        annotated = self.draw_detections(image, detections)
        
        # Guardar
        output_dir = Path("data/vehicle_detections")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"vehicles_{timestamp}.jpg"
        filepath = output_dir / filename
        
        cv2.imwrite(str(filepath), annotated)
        
        return str(filepath)


# Instancia global del servicio
_vehicle_service = None

def get_vehicle_service() -> VehicleDetectionService:
    """Obtiene la instancia del servicio de detección de vehículos"""
    global _vehicle_service
    if _vehicle_service is None:
        _vehicle_service = VehicleDetectionService()
    return _vehicle_service
