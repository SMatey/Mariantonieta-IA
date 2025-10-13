# voice_vision/local_face_cv.py
from __future__ import annotations
from pathlib import Path
import urllib.request, os
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "face_recognition_sface_2021dec.onnx"
MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

# Descarga el modelo si no existe
if not MODEL_PATH.exists():
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH.as_posix())

# Crea el recognizer (128-D)
RECO = cv2.FaceRecognizerSF_create(MODEL_PATH.as_posix(), "")

def _crop(img: np.ndarray, rect: dict | None) -> np.ndarray:
    if not rect:
        return img
    x, y, w, h = rect["left"], rect["top"], rect["width"], rect["height"]
    H, W = img.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    return img[y0:y1, x0:x1]

def embed_from_bytes(content: bytes, face_rect: dict | None = None) -> list[float]:
    """
    Devuelve un embedding L2-normalizado (float32 list, len=128).
    Si pasas face_rect (de Azure Detect), recortamos primero.
    """
    arr = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Imagen inválida")
    faceimg = _crop(img, face_rect)
    feat = RECO.feature(faceimg)  # np.ndarray (1,128)
    feat = feat.astype("float32").reshape(-1)
    n = np.linalg.norm(feat) + 1e-10
    feat = (feat / n).astype("float32")
    return feat.tolist()
