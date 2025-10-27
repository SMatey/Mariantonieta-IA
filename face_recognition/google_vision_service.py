# google_vision_service.py
import io
from typing import List, Dict, Any
from google.cloud import vision

# (Variables LIKELIHOOD omitidas por brevedad)
LIKELIHOOD_TO_TEXT = {
    vision.Likelihood.UNKNOWN: "UNKNOWN",
    vision.Likelihood.VERY_UNLIKELY: "VERY_UNLIKELY",
    vision.Likelihood.UNLIKELY: "UNLIKELY",
    vision.Likelihood.POSSIBLE: "POSSIBLE",
    vision.Likelihood.LIKELY: "LIKELY",
    vision.Likelihood.VERY_LIKELY: "VERY_LIKELY",
}
LIKELIHOOD_TO_SCORE = {
    vision.Likelihood.UNKNOWN: 0.0,
    vision.Likelihood.VERY_UNLIKELY: 0.0,
    vision.Likelihood.UNLIKELY: 0.25,
    vision.Likelihood.POSSIBLE: 0.5,
    vision.Likelihood.LIKELY: 0.75,
    vision.Likelihood.VERY_LIKELY: 0.95,
}


def _poly_to_bbox(poly) -> Dict[str, int]:
    xs = [v.x for v in poly.vertices]
    ys = [v.y for v in poly.vertices]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return {"left": x1, "top": y1, "width": x2 - x1, "height": y2 - y1}

def detect_faces_with_google(image_stream: io.BytesIO) -> List[Dict[str, Any]]:
    # --- LA LÍNEA MÁGICA ---
    image_stream.seek(0) # Rebobinamos el stream de Google
    # --- FIN ---
    
    client = vision.ImageAnnotatorClient()
    content = image_stream.read()
    image = vision.Image(content=content)
    response = client.face_detection(image=image)

    if response.error and response.error.message:
        return [{"error": f"Google Vision: {response.error.message}"}]

    faces = []
    for f in (response.face_annotations or []):
        bbox = _poly_to_bbox(f.fd_bounding_poly or f.bounding_poly)
        like_text = {
            "joy": LIKELIHOOD_TO_TEXT.get(f.joy_likelihood, "UNKNOWN"),
            "sorrow": LIKELIHOOD_TO_TEXT.get(f.sorrow_likelihood, "UNKNOWN"),
            "anger": LIKELIHOOD_TO_TEXT.get(f.anger_likelihood, "UNKNOWN"),
            "surprise": LIKELIHOOD_TO_TEXT.get(f.surprise_likelihood, "UNKNOWN"),
        }
        scores = {
            "joy":      LIKELIHOOD_TO_SCORE.get(f.joy_likelihood, 0.0),
            "sorrow":   LIKELIHOOD_TO_SCORE.get(f.sorrow_likelihood, 0.0),
            "anger":    LIKELIHOOD_TO_SCORE.get(f.anger_likelihood, 0.0),
            "surprise": LIKELIHOOD_TO_SCORE.get(f.surprise_likelihood, 0.0),
        }
        best = max(scores, key=scores.get)
        faces.append({
            "detection_source": "google_vision",
            "position": bbox,
            "likelihoods": like_text,
            "best_emotion": {"label": best, "score": scores[best]}
        })
    return faces