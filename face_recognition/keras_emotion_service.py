# keras_emotion_service.py
import io
import cv2
import numpy as np
import tensorflow as tf
from typing import List, Dict, Any
from PIL import Image

class EmotionRecognitionService:
    def __init__(self, model_path: str = "dl_models/emotion_model.keras"):
        """
        Inicializa el servicio de reconocimiento de emociones con tu modelo entrenado
        """
        self.model_path = model_path
        self.model = None
        self.face_cascade = None
        self.emotion_labels = ['interested', 'neutral', 'disappointed']
        self.img_size = (224, 224)
        
        # Mapeo de emociones para compatibilidad con Google Vision API
        self.emotion_mapping = {
            'interested': 'joy',      # interested -> joy
            'neutral': 'neutral',     # neutral -> neutral (nuevo)
            'disappointed': 'sorrow'  # disappointed -> sorrow
        }
        
        # Mapeo inverso para scoring
        self.google_emotions = ['joy', 'sorrow', 'anger', 'surprise']
        
        self._load_model()
        self._load_face_detector()
    
    def _load_model(self):
        """Carga el modelo de reconocimiento de emociones"""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"✅ Modelo cargado exitosamente desde {self.model_path}")
        except Exception as e:
            print(f"❌ Error al cargar modelo: {e}")
            # Fallback: intentar cargar solo pesos si el modelo completo falla
            try:
                self._create_model_architecture()
                print("⚠️  Usando arquitectura por defecto con pesos guardados")
            except Exception as e2:
                print(f"❌ Error crítico: {e2}")
                raise e2
    
    def _create_model_architecture(self):
        """Recrea la arquitectura del modelo como fallback"""
        from tensorflow.keras import layers, models
        
        # Recrear la arquitectura exacta que usaste
        base_model = tf.keras.applications.EfficientNetB3(
            include_top=False,
            weights="imagenet",
            input_shape=self.img_size + (3,)
        )
        
        inputs = layers.Input(shape=self.img_size + (3,))
        x = tf.keras.applications.efficientnet.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(256, activation="relu")(x)  
        x = layers.Dropout(0.4)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(3, activation="softmax", name="predictions")(x)
        
        self.model = models.Model(inputs, outputs, name="emotion_model_simplified")
        
        # Intentar cargar pesos
        try:
            self.model.load_weights("emotion_model_final_weights.h5")
        except:
            print("⚠️  No se pudieron cargar los pesos guardados")
    
    def _load_face_detector(self):
        """Carga el detector de rostros OpenCV"""
        self.face_net = None
        self.face_cascade = None
        
        # Intentar cargar ONNX primero
        try:
            self.face_net = cv2.dnn.readNetFromONNX("ml_models/face_detection_yunet_2023mar.onnx")
            print("✅ Detector de rostros ONNX cargado")
            return
        except Exception as e:
            print(f"⚠️  Error cargando ONNX: {e}")
            
        # Fallback a Haar Cascade
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.face_cascade.empty():
                raise Exception("Haar cascade vacío")
            print("✅ Haar Cascade cargado como fallback")
            return
        except Exception as e:
            print(f"⚠️  Error cargando Haar Cascade: {e}")
            
        # Último fallback: usar detector simple basado en MediaPipe o crear uno básico
        print("⚠️  Usando detector básico como último fallback")
        self.face_cascade = "basic"  # Marcador para detector básico
    
    def _detect_faces_onnx(self, image: np.ndarray) -> List[Dict[str, int]]:
        """Detecta rostros usando el modelo ONNX YuNet"""
        try:
            height, width = image.shape[:2]
            
            # Configurar YuNet
            self.face_net.setInputSize([width, height])
            self.face_net.setConfidenceThreshold(0.7)
            self.face_net.setNMSThreshold(0.3)
            
            # Detectar rostros
            faces = self.face_net.detect(image)
            
            face_boxes = []
            if faces[1] is not None:
                for face in faces[1]:
                    x, y, w, h = face[:4].astype(int)
                    face_boxes.append({
                        "left": max(0, x),
                        "top": max(0, y), 
                        "width": min(w, width - x),
                        "height": min(h, height - y)
                    })
            return face_boxes
        except:
            return self._detect_faces_haar(image)
    
    def _detect_faces_haar(self, image: np.ndarray) -> List[Dict[str, int]]:
        """Detecta rostros usando Haar Cascade como fallback"""
        if self.face_cascade is None:
            return self._detect_faces_basic(image)
            
        if self.face_cascade == "basic":
            return self._detect_faces_basic(image)
            
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            face_boxes = []
            for (x, y, w, h) in faces:
                face_boxes.append({
                    "left": x,
                    "top": y,
                    "width": w,
                    "height": h
                })
            return face_boxes
        except Exception as e:
            print(f"Error en Haar Cascade: {e}")
            return self._detect_faces_basic(image)
    
    def _detect_faces_basic(self, image: np.ndarray) -> List[Dict[str, int]]:
        """Detector básico que asume que toda la imagen es un rostro"""
        height, width = image.shape[:2]
        
        # Si no hay detectores disponibles, usar toda la imagen
        return [{
            "left": 0,
            "top": 0,
            "width": width,
            "height": height
        }]
    
    def _preprocess_face(self, face_img: np.ndarray) -> np.ndarray:
        """Preprocesa la imagen del rostro para el modelo"""
        # Redimensionar a 224x224
        face_resized = cv2.resize(face_img, self.img_size)
        
        # Convertir a RGB
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # Normalizar a [0, 1]
        face_normalized = face_rgb.astype(np.float32) / 255.0
        
        # Agregar dimensión de batch
        face_batch = np.expand_dims(face_normalized, axis=0)
        
        return face_batch
    
    def _score_to_likelihood_text(self, score: float) -> str:
        """Convierte score numérico a texto compatible con Google Vision"""
        if score >= 0.8:
            return "VERY_LIKELY"
        elif score >= 0.6:
            return "LIKELY"
        elif score >= 0.4:
            return "POSSIBLE"
        elif score >= 0.2:
            return "UNLIKELY"
        elif score > 0:
            return "VERY_UNLIKELY"
        else:
            return "UNKNOWN"
    
    def _create_google_compatible_response(self, emotion_scores: Dict[str, float]) -> Dict[str, Any]:
        """Crea una respuesta compatible con la estructura de Google Vision"""
        
        # Encontrar la emoción principal
        best_emotion_key = max(emotion_scores, key=emotion_scores.get)
        best_emotion_mapped = self.emotion_mapping.get(best_emotion_key, 'neutral')
        best_score = emotion_scores[best_emotion_key]
        
        # Crear scores para todas las emociones de Google Vision
        google_scores = {
            'joy': 0.0,
            'sorrow': 0.0,
            'anger': 0.0,
            'surprise': 0.0
        }
        
        # Mapear las emociones de nuestro modelo a Google Vision
        for our_emotion, google_emotion in self.emotion_mapping.items():
            if our_emotion in emotion_scores and google_emotion in google_scores:
                google_scores[google_emotion] = emotion_scores[our_emotion]
        
        # Si es neutral, distribuir un poco entre las emociones
        if best_emotion_key == 'neutral':
            # Para neutral, dar scores bajos a todas las emociones
            base_score = emotion_scores['neutral'] * 0.2  # 20% del score de neutral
            google_scores = {emotion: base_score for emotion in google_scores}
            best_emotion_mapped = 'joy'  # Default para neutral
            best_score = base_score
        
        # Asegurar que ningún score sea exactamente 0 para evitar "UNKNOWN"
        for emotion in google_scores:
            if google_scores[emotion] == 0:
                google_scores[emotion] = 0.1  # Score mínimo
        
        # Convertir scores a texto
        likelihoods = {
            emotion: self._score_to_likelihood_text(score)
            for emotion, score in google_scores.items()
        }
        
        return {
            'likelihoods': likelihoods,
            'best_emotion': {
                'label': best_emotion_mapped,
                'score': float(best_score)
            },
            'google_scores': google_scores,
            'original_scores': emotion_scores
        }
    def _predict_emotion(self, face_img: np.ndarray) -> Dict[str, Any]:
        """Predice la emoción de un rostro y devuelve formato compatible"""
        try:
            # Preprocesar la imagen
            preprocessed = self._preprocess_face(face_img)
            
            # Hacer predicción
            prediction = self.model.predict(preprocessed, verbose=0)
            
            # Obtener probabilidades
            probabilities = prediction[0]
            
            # Crear diccionario de scores originales
            emotion_scores = {}
            for i, label in enumerate(self.emotion_labels):
                emotion_scores[label] = float(probabilities[i])
            
            # Crear respuesta compatible con Google Vision
            compatible_response = self._create_google_compatible_response(emotion_scores)
            
            return compatible_response
            
        except Exception as e:
            # Respuesta de error compatible
            return {
                'likelihoods': {
                    'joy': 'UNKNOWN',
                    'sorrow': 'UNKNOWN', 
                    'anger': 'UNKNOWN',
                    'surprise': 'UNKNOWN'
                },
                'best_emotion': {'label': 'joy', 'score': 0.0},
                'google_scores': {'joy': 0.0, 'sorrow': 0.0, 'anger': 0.0, 'surprise': 0.0},
                'original_scores': {'interested': 0.0, 'neutral': 0.0, 'disappointed': 0.0},
                'error': str(e)
            }

def detect_faces_with_keras(image_stream: io.BytesIO) -> List[Dict[str, Any]]:
    """
    Función principal que reemplaza detect_faces_with_google
    """
    try:
        image_stream.seek(0)
        
        # Cargar imagen
        pil_image = Image.open(image_stream)
        image_array = np.array(pil_image)
        
        # Convertir RGB a BGR para OpenCV
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_array
        
        # Inicializar servicio (singleton pattern)
        if not hasattr(detect_faces_with_keras, '_service'):
            detect_faces_with_keras._service = EmotionRecognitionService()
        
        service = detect_faces_with_keras._service
        
        # Detectar rostros
        face_boxes = []
        if service.face_net is not None:
            face_boxes = service._detect_faces_onnx(image_bgr)
        elif service.face_cascade is not None:
            face_boxes = service._detect_faces_haar(image_bgr)
        else:
            # Fallback final: usar toda la imagen
            face_boxes = service._detect_faces_basic(image_bgr)
        
        faces_result = []
        
        for face_box in face_boxes:
            # Extraer región del rostro
            x = face_box["left"]
            y = face_box["top"] 
            w = face_box["width"]
            h = face_box["height"]
            
            face_roi = image_bgr[y:y+h, x:x+w]
            
            if face_roi.size == 0:
                continue
                
            # Predecir emoción
            emotion_result = service._predict_emotion(face_roi)
            
            # Formatear resultado 100% compatible con Google Vision API
            faces_result.append({
                "detection_source": "keras_custom_model",
                "position": face_box,
                "likelihoods": emotion_result["likelihoods"],
                "best_emotion": emotion_result["best_emotion"],
                # Información adicional para debugging (opcional)
                "model_info": {
                    "original_emotions": emotion_result.get("original_scores", {}),
                    "architecture": "EfficientNetB3-Custom",
                    "confidence": emotion_result["best_emotion"]["score"]
                }
            })
        
        return faces_result
        
    except Exception as e:
        return [{"error": f"Keras Model Error: {str(e)}"}]

# Función de compatibilidad para mantener la interfaz
def detect_faces_with_emotion(image_stream: io.BytesIO) -> List[Dict[str, Any]]:
    """Alias para compatibilidad"""
    return detect_faces_with_keras(image_stream)