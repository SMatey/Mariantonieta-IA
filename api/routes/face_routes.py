# face_routes.py (agrega este endpoint)
from fastapi import APIRouter, File, UploadFile, HTTPException
from io import BytesIO
from face_recognition.azure_face_service import detect_face_with_azure
from face_recognition.google_vision_service import detect_faces_with_google
from face_recognition.fusion_service import fuse_azure_google

router = APIRouter(prefix="/face", tags=["face"])

@router.post("/analyze-azure-google")
async def analyze_azure_google(file: UploadFile = File(...)):
    try:
        name = file.filename.lower()
        if not name.endswith((".jpg", ".jpeg", ".png")):
            raise HTTPException(status_code=400, detail="Formato no soportado (usa .jpg/.jpeg/.png)")
        buf = BytesIO(await file.read())

        # 1) Detección con Azure
        az = detect_face_with_azure(BytesIO(buf.getvalue()))
        if isinstance(az, dict) and "error" in az:
            raise HTTPException(status_code=502, detail=az["error"])

        # 2) Emociones con Google (sobre la misma imagen)
        gv = detect_faces_with_google(BytesIO(buf.getvalue()))
        if len(gv) and isinstance(gv[0], dict) and "error" in gv[0]:
            raise HTTPException(status_code=502, detail=gv[0]["error"])

        # 3) Fusión por IoU
        fused = fuse_azure_google(az, gv, iou_threshold=0.2)

        return {
            "faces": fused,
            "meta": {
                "azure_faces": len(az or []),
                "google_faces": len(gv or []),
                "notes": "Rostros detectados por Azure; emociones estimadas por Google Vision; emparejados por IoU>=0.2"
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno (Azure+Google): {e}")
