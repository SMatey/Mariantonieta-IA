# fusion_service.py
from typing import Dict, Any, List, Optional

def _rect_to_xyxy(r: Dict[str, int]):
    x1 = r["left"]
    y1 = r["top"]
    x2 = r["left"] + r["width"]
    y2 = r["top"]  + r["height"]
    return x1, y1, x2, y2

def _iou(a: Dict[str, int], b: Dict[str, int]) -> float:
    ax1, ay1, ax2, ay2 = _rect_to_xyxy(a)
    bx1, by1, bx2, by2 = _rect_to_xyxy(b)
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def fuse_azure_google(azure_faces: List[Dict[str, Any]], google_faces: List[Dict[str, Any]], iou_threshold: float = 0.2):
    """
    Devuelve lista con: rectángulo de Azure + emociones de Google emparejadas por IoU.
    """
    fused = []
    used_g = set()
    for i, az in enumerate(azure_faces or []):
        az_rect = az.get("position") or {}
        # Buscar la cara de Google con mayor IoU
        best_idx, best_iou = None, 0.0
        for j, gv in enumerate(google_faces or []):
            if j in used_g: 
                continue
            gv_rect = gv.get("position") or {}
            iou = _iou(az_rect, gv_rect)
            if iou > best_iou:
                best_iou, best_idx = iou, j
        if best_idx is not None and best_iou >= iou_threshold:
            used_g.add(best_idx)
            gf = google_faces[best_idx]
            fused.append({
                "position": az_rect,              # bounding box de AZURE
                "azure_detection": True,
                "google_emotions": gf.get("likelihoods"),
                "google_best_emotion": gf.get("best_emotion"),
                "match": {"iou": best_iou}
            })
        else:
            # No se encontró match de Google para esta cara de Azure
            fused.append({
                "position": az_rect,
                "azure_detection": True,
                "google_emotions": None,
                "google_best_emotion": None,
                "match": {"iou": best_iou}
            })
    return fused
