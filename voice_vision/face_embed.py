# voice_vision/face_embed.py
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
from typing import Optional, Tuple
import os

import cv2  # requiere opencv-contrib-python
import numpy as np
from dotenv import load_dotenv

# Carga .env por si se importa este módulo de manera aislada
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Rutas de modelos (puedes cambiarlas en .env)
DETECTOR_PATH = Path(os.getenv("FACE_DETECTOR_ONNX", str(ROOT / "models" / "face_detection_yunet_2023mar.onnx")))
EMBEDDER_PATH = Path(os.getenv("FACE_EMBED_ONNX",   str(ROOT / "models" / "face_recognition_sface_2021dec.onnx")))

def _assert_models_exist() -> None:
    if not DETECTOR_PATH.exists():
        raise FileNotFoundError(
            f"Modelo de detección no encontrado: {DETECTOR_PATH}\n"
            "Asegúrate de tener FACE_DETECTOR_ONNX en .env o coloca el onnx en ./models/"
        )
    if not EMBEDDER_PATH.exists():
        raise FileNotFoundError(
            f"Modelo de embedding no encontrado: {EMBEDDER_PATH}\n"
            "Asegúrate de tener FACE_EMBED_ONNX en .env o coloca el onnx en ./models/"
        )

def _check_opencv_contrib() -> None:
    # FaceDetectorYN/FaceRecognizerSF están en contrib
    has_yn   = hasattr(cv2, "FaceDetectorYN_create")
    has_sface = hasattr(cv2, "FaceRecognizerSF_create")
    if not (has_yn and has_sface):
        raise RuntimeError(
            "OpenCV contrib requerido. Instala 'opencv-contrib-python==4.10.0.84'."
        )

@lru_cache(maxsize=1)
def _get_detector() -> "cv2.FaceDetectorYN":
    _assert_models_exist()
    _check_opencv_contrib()
    # Se crea sin input size; se fija por imagen
    det = cv2.FaceDetectorYN_create(
        model=str(DETECTOR_PATH),
        config="",
        input_size=(320, 320),  # placeholder, se actualiza por imagen
        score_threshold=0.9,
        nms_threshold=0.3,
        top_k=5000
    )
    return det

@lru_cache(maxsize=1)
def _get_recognizer() -> "cv2.FaceRecognizerSF":
    _assert_models_exist()
    _check_opencv_contrib()
    rec = cv2.FaceRecognizerSF_create(
        model=str(EMBEDDER_PATH),
        config=""
    )
    return rec

def _decode_image_from_bytes(content: bytes) -> Optional[np.ndarray]:
    if not content:
        return None
    arr = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def _select_largest_face(faces: np.ndarray) -> np.ndarray:
    """
    faces: array (N, 15) YuNet: [x, y, w, h, score, rx, ry, lx, ly, nx, ny, mr_x, mr_y, ml_x, ml_y]
    Devuelve la fila del rostro con mayor área.
    """
    if faces is None or len(faces) == 0:
        return np.array([])
    areas = faces[:, 2] * faces[:, 3]  # w * h
    idx = int(np.argmax(areas))
    return faces[idx]

def _set_input_size(det: "cv2.FaceDetectorYN", img: np.ndarray) -> None:
    h, w = img.shape[:2]
    det.setInputSize((w, h))

def _detect_faces(img: np.ndarray) -> np.ndarray:
    det = _get_detector()
    _set_input_size(det, img)
    out = det.detect(img)
    # OpenCV 4.10 devuelve directamente np.ndarray o (faces, scores) segun build; normalizamos
    faces = None
    if isinstance(out, tuple):
        # nuevas versiones devuelven (faces, scores) o (faces,)
        faces = out[0]
    else:
        faces = out
    return faces if faces is not None else np.empty((0, 15), dtype=np.float32)

def embed_from_bytes(content: bytes) -> Optional[np.ndarray]:
    """
    Devuelve un embedding 128D (float32, normalizado) del rostro más grande en la imagen.
    Si no detecta rostro, retorna None.
    """
    img = _decode_image_from_bytes(content)
    if img is None:
        return None

    faces = _detect_faces(img)
    if faces is None or len(faces) == 0:
        return None

    face = _select_largest_face(faces)
    if face.size == 0:
        return None

    rec = _get_recognizer()
    # Alinear y recortar según landmarks de YuNet
    aligned = rec.alignCrop(img, face)
    # Extraer embedding SFace (128D)
    feat = rec.feature(aligned)  # np.ndarray shape (128,) o (1,128) según build
    feat = np.asarray(feat, dtype=np.float32).reshape(-1)
    # Normalizar L2 para similitud coseno
    n = np.linalg.norm(feat) + 1e-12
    feat = feat / n
    return feat

def embed_and_box_from_bytes(content: bytes) -> Tuple[Optional[np.ndarray], Optional[Tuple[int,int,int,int]]]:
    """
    Devuelve (embedding, bbox) donde bbox es (x, y, w, h) del rostro más grande.
    Si no hay rostro, (None, None).
    """
    img = _decode_image_from_bytes(content)
    if img is None:
        return None, None

    faces = _detect_faces(img)
    if faces is None or len(faces) == 0:
        return None, None

    face = _select_largest_face(faces)
    if face.size == 0:
        return None, None

    rec = _get_recognizer()
    aligned = rec.alignCrop(img, face)
    feat = rec.feature(aligned)
    feat = np.asarray(feat, dtype=np.float32).reshape(-1)
    feat = feat / (np.linalg.norm(feat) + 1e-12)

    x, y, w, h = face[:4].astype(int).tolist()
    return feat, (x, y, w, h)

def warmup() -> None:
    """
    Carga modelos y hace una pasada dummy para que el primer request sea rápido.
    """
    # Imagen negra 320x320
    dummy = np.zeros((320, 320, 3), dtype=np.uint8)
    det = _get_detector()
    _set_input_size(det, dummy)
    _ = det.detect(dummy)
    _ = _get_recognizer()
