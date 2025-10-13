# voice_vision/google_emotion.py
from __future__ import annotations
from typing import Dict, List, Any
from functools import lru_cache
from google.cloud import vision
import httpx

# Puntuación 0..5 según enums de Vision:
# UNKNOWN(0), VERY_UNLIKELY(1), UNLIKELY(2), POSSIBLE(3), LIKELY(4), VERY_LIKELY(5)
LIKELY_SCORE = {
    vision.Likelihood.UNKNOWN: 0,
    vision.Likelihood.VERY_UNLIKELY: 1,
    vision.Likelihood.UNLIKELY: 2,
    vision.Likelihood.POSSIBLE: 3,
    vision.Likelihood.LIKELY: 4,
    vision.Likelihood.VERY_LIKELY: 5,
}

@lru_cache(maxsize=1)
def _get_client() -> vision.ImageAnnotatorClient:
    # Usa credenciales de GOOGLE_APPLICATION_CREDENTIALS o ADC
    return vision.ImageAnnotatorClient()

def _score(v: int) -> int:
    try:
        return LIKELY_SCORE.get(vision.Likelihood(v), 0)
    except Exception:
        return 0

def _dominant(scores: Dict[str, int]) -> str:
    if not scores or max(scores.values()) == 0:
        return "none"
    order = ["joy", "sorrow", "anger", "surprise"]  # desempate estable
    return max(scores.items(), key=lambda kv: (kv[1], -order.index(kv[0])))[0]

# --- NUEVO: mapea emociones a sentimiento ---
def _emotion_to_sentiment(scores: dict[str, int]) -> str:
    # Simple: joy alta => positivo; sorrow/anger altas => negativo; si no, neutral
    if scores.get("joy", 0) >= 4:
        return "positive"
    if max(scores.get("sorrow", 0), scores.get("anger", 0)) >= 3:
        return "negative"
    return "neutral"

def _face_dict(f) -> Dict[str, Any]:
    # boundingPoly como lista de puntos (x,y)
    poly = [{"x": v.x, "y": v.y} for v in f.bounding_poly.vertices]
    likelihoods = {
        "joy":       vision.Likelihood(f.joy_likelihood).name,
        "sorrow":    vision.Likelihood(f.sorrow_likelihood).name,
        "anger":     vision.Likelihood(f.anger_likelihood).name,
        "surprise":  vision.Likelihood(f.surprise_likelihood).name,
    }
    scores = {
        "joy":       _score(f.joy_likelihood),
        "sorrow":    _score(f.sorrow_likelihood),
        "anger":     _score(f.anger_likelihood),
        "surprise":  _score(f.surprise_likelihood),
    }
    sentiment = _emotion_to_sentiment(scores)  # <-- NUEVO
    return {
        "boundingPoly": poly,
        "rollAngle": f.roll_angle,
        "panAngle": f.pan_angle,
        "tiltAngle": f.tilt_angle,
        "detectionConfidence": float(getattr(f, "detection_confidence", 0.0) or 0.0),
        "likelihoods": likelihoods,
        "scores": scores,
        "top_emotion": _dominant(scores),
        "sentiment": sentiment,  # <-- NUEVO
    }

def detect_emotions_from_bytes(content: bytes) -> List[Dict[str, Any]]:
    """
    Devuelve TODAS las caras con polígonos, likelihoods, scores (0..5),
    top_emotion y sentiment.
    """
    try:
        client = _get_client()
        image = vision.Image(content=content)
        resp = client.face_detection(image=image)
        if resp.error.message:
            raise RuntimeError(resp.error.message)
        return [_face_dict(f) for f in (resp.face_annotations or [])]
    except Exception as e:
        # Re-lanza con mensaje claro (útil si faltan credenciales)
        raise RuntimeError(f"Vision API error: {e}") from e

def detect_emotion_from_bytes(content: bytes) -> Dict[str, Any]:
    """
    Resumen de UNA cara (la primera). Si no hay caras, devuelve 0s, 'none' y 'neutral'.
    """
    faces = detect_emotions_from_bytes(content)
    if not faces:
        return {
            "joy": 0, "sorrow": 0, "anger": 0, "surprise": 0,
            "dominant": "none", "sentiment": "neutral"
        }
    top = faces[0]
    # Campos simples + dominante + sentimiento
    return {
        "joy": top["scores"]["joy"],
        "sorrow": top["scores"]["sorrow"],
        "anger": top["scores"]["anger"],
        "surprise": top["scores"]["surprise"],
        "dominant": top["top_emotion"],
        "sentiment": top["sentiment"],
    }

def detect_emotion_from_url(image_url: str) -> Dict[str, Any]:
    """
    Descarga la imagen por URL y devuelve el resumen de UNA cara (como detect_emotion_from_bytes).
    """
    with httpx.Client(timeout=20.0) as http:
        r = http.get(image_url)
        r.raise_for_status()
    return detect_emotion_from_bytes(r.content)
