# voice_vision/face_embed.py
from __future__ import annotations
import os
import numpy as np
import cv2  # requiere opencv-contrib
from functools import lru_cache

# Rutas por defecto (puedes sobreescribir con .env)
DEF_DET = os.getenv("FACE_DETECTOR_ONNX", "./models/face_detection_yunet_2023mar.onnx")
DEF_REC = os.getenv("FACE_EMBED_ONNX",     "./models/face_recognition_sface_2021dec.onnx")

@lru_cache(maxsize=1)
def _get_models():
    if not os.path.exists(DEF_DET):
        raise FileNotFoundError(f"No se encontró el detector: {DEF_DET}")
    if not os.path.exists(DEF_REC):
        raise FileNotFoundError(f"No se encontró el reconocedor: {DEF_REC}")

    # Detector YuNet
    # input size se ajusta a cada imagen antes de detectar
    detector = cv2.FaceDetectorYN.create(
        model=DEF_DET,
        config="",
        input_size=(320, 320),
        score_threshold=0.85,
        nms_threshold=0.3,
        top_k=5000
    )
    # Reconocedor SFace
    recog = cv2.FaceRecognizerSF.create(model=DEF_REC, config="")
    return detector, recog

def _bytes_to_bgr(content: bytes):
    arr = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No se pudo decodificar la imagen (bytes inválidos).")
    return img

def embed_from_bytes(content: bytes) -> np.ndarray | None:
    """
    Devuelve un vector L2-normalizado (np.ndarray shape (512,)) o None si no hay rostro.
    """
    detector, recog = _get_models()
    img = _bytes_to_bgr(content)
    h, w = img.shape[:2]
    detector.setInputSize((w, h))
    # faces: (N, 15) [x,y,w,h, l0x,l0y, l1x,l1y, l2x,l2y, l3x,l3y, l4x,l4y, score]
    faces = detector.detect(img)[1]
    if faces is None or len(faces) == 0:
        return None

    # Tomamos el rostro con mayor score
    faces = sorted(faces, key=lambda f: float(f[-1]), reverse=True)
    face = faces[0]

    # Alinear y extraer embedding
    aligned = recog.alignCrop(img, face)
    feat = recog.feature(aligned)  # (512,) float32
    # L2 normalizar
    feat = feat / (np.linalg.norm(feat) + 1e-12)
    return feat.astype(np.float32)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))
