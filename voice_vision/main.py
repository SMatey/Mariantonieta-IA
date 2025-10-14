# voice_vision/main.py
from __future__ import annotations
from pathlib import Path
from typing import List
import sys
import os
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from voice_vision.google_emotion import detect_emotions_from_bytes

# Carga .env antes de usar envs/almacenes
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

# --- Clientes / utilidades propias ---
from voice_vision.face_recognition import AzureFaceClient
from voice_vision.google_emotion import detect_emotion_from_bytes
from voice_vision.face_embed import embed_from_bytes  # Fallback local (SFace)
from voice_vision.user_store import load_users, add_user_blobs, ROOT as DATA_ROOT
from storage.blob_store import upload_user_photo, download_user_photo

app = FastAPI(title="Maria Antonieta Face API", version="0.1.0")

# CORS (ajusta dominios en producción)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in (os.getenv("ALLOWED_ORIGINS") or "*").split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instancia el cliente y, si se puede, crea/asegura el PersonGroup
client = AzureFaceClient()
try:
    client.ensure_person_group()
except Exception:
    # Si la suscripción/región no soporta Identify, el cliente quedará en modo detección-only.
    pass


# ------------------------ Modelos de petición ------------------------
class EnrollRequest(BaseModel):
    person_name: str = Field(..., min_length=1)
    images: List[str] = Field(
        ..., min_items=1, description="URLs públicas (3–10 recomendado) de la misma persona"
    )

class IdentifyRequest(BaseModel):
    image: str


def _first_rect_from_detect(faces: list[dict]) -> dict | None:
    if not faces:
        return None
    rect = faces[0].get("faceRectangle")
    return rect  # {"top":..,"left":..,"width":..,"height":..}


# ------------------------ Endpoints ------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "personGroup": getattr(client, "person_group_id", None),
        "supportsIdentity": getattr(client, "supports_identity", True),
    }


@app.post("/faces/detect")
def detect(req: IdentifyRequest):
    """Detección de rostros (Azure). Devuelve bbox y faceId por rostro."""
    try:
        faces = client.detect(req.image, return_face_id=True)
        return {"faces": faces}
    except httpx.HTTPStatusError as he:
        raise HTTPException(status_code=he.response.status_code, detail=he.response.text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/faces/persons")
def list_persons():
    """Lista de personas enroladas (Identify). En modo detección-only devolverá []."""
    try:
        return client.list_persons()
    except httpx.HTTPStatusError as he:
        raise HTTPException(status_code=he.response.status_code, detail=he.response.text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/faces/enroll")
def enroll(req: EnrollRequest):
    """Enrolar persona (requiere Identify). En detección-only devolverá 501."""
    if not getattr(client, "supports_identity", True):
        raise HTTPException(
            status_code=501,
            detail="Identification is disabled for this subscription/region. Modo detección-only.",
        )
    try:
        person_id = client.create_person(req.person_name)
        for url in req.images:
            client.add_person_face(person_id, url)
        client.train_and_wait()
        return {"personId": person_id, "enrolledFaces": len(req.images), "trained": True}
    except httpx.HTTPStatusError as he:
        raise HTTPException(status_code=he.response.status_code, detail=he.response.text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/faces/identify")
def identify(req: IdentifyRequest):
    """Identificar persona (si Identify está disponible). En detección-only retorna detecciones y un mensaje."""
    try:
        faces = client.detect(req.image, return_face_id=True)
        if not faces:
            return {"faces": [], "results": [], "message": "No se detectaron rostros"}

        if not getattr(client, "supports_identity", True):
            return {
                "faces": faces,
                "results": [],
                "message": "Identification unavailable: modo detección-only",
            }

        face_ids = [f["faceId"] for f in faces if "faceId" in f]
        results = client.identify(face_ids)

        enriched = []
        for f, r in zip(faces, results):
            cand = (r.get("candidates") or [])
            best = cand[0] if cand else None
            enriched.append(
                {
                    "faceRectangle": f.get("faceRectangle"),
                    "faceId": f.get("faceId"),
                    "bestCandidate": best,  # {personId, confidence} o None
                }
            )
        return {"results": enriched}
    except httpx.HTTPStatusError as he:
        raise HTTPException(status_code=he.response.status_code, detail=he.response.text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/auth/enroll_user_multipart")
async def enroll_user_multipart(user_id: str = Form(...), files: List[UploadFile] = File(...)):
    """
    Guarda fotos de referencia del usuario en Azure Blob y registra sus rutas (user_store).
    """
    try:
        blobs: List[str] = []
        for f in files:
            if not f.content_type or not f.content_type.startswith("image/"):
                continue
            content = await f.read()
            if not content:
                continue

            # Sube a blob y guarda ruta
            path = upload_user_photo(user_id, content)
            blobs.append(path)

        if not blobs:
            raise HTTPException(status_code=400, detail="No se recibieron imágenes válidas.")

        added = add_user_blobs(user_id, blobs)
        return {"ok": True, "user_id": user_id, "added_blobs": added, "blob_paths": blobs}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/auth/login_multipart")
async def login_multipart(
    user_id: str = Form(...),
    file: UploadFile = File(...),
    threshold: float = 0.70,   # Azure Verify: confianza mínima
    cos_sim_min: float = 0.70  # Fallback local SFace: similitud coseno mínima
):
    """
    Login con 2 rutas:
      A) Azure Verify (si está disponible) contra fotos de referencia en Blob
      B) Fallback local: embeddings SFace y similitud coseno contra referencias del usuario
    Siempre devuelve emoción (Google Vision) como información adicional.
    """
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Archivo vacío")

        # Emoción (no bloquea el login si falla Vision)
        try:
            emotion = detect_emotion_from_bytes(content)
        except Exception:
            emotion = {"joy": 0, "sorrow": 0, "anger": 0, "surprise": 0, "dominant": "none"}

        users = load_users()
        rec = users.get(user_id)
        if not rec:
            return {"user_id": user_id, "login": "failed", "reason": "usuario sin referencias", "emotion": emotion}

        # 1) Azure Verify (si está disponible)
        if getattr(client, "supports_identity", True):
            try:
                faces_live = client.detect_bytes(content, return_face_id=True)
                if faces_live:
                    live_id = faces_live[0].get("faceId")
                    best_conf = 0.0
                    for blob_path in rec.get("blobs", []):
                        try:
                            ref_bytes = download_user_photo(blob_path)
                        except Exception:
                            continue
                        ref_faces = client.detect_bytes(ref_bytes, return_face_id=True)
                        if not ref_faces:
                            continue
                        ref_id = ref_faces[0].get("faceId")
                        verify = client.verify_faces(live_id, ref_id)
                        conf = float(verify.get("confidence", 0.0))
                        if verify.get("isIdentical") and conf > best_conf:
                            best_conf = conf
                            if conf >= threshold:
                                return {
                                    "user_id": user_id,
                                    "login": "success",
                                    "via": "azure-verify",
                                    "confidence": round(conf, 4),
                                    "threshold_used": threshold,
                                    "emotion": emotion
                                }
            except httpx.HTTPStatusError as he:
                # Si Azure devuelve 403/400/404 seguimos por fallback local
                if he.response.status_code not in (403, 400, 404):
                    raise

        # 2) Fallback local con SFace (sin Cosmos)
        import numpy as np

        live = embed_from_bytes(content)  # np.ndarray (128,)
        if live is None:
            raise HTTPException(status_code=422, detail="No se detectó rostro en la imagen")

        best_sim = -1.0
        for blob_path in rec.get("blobs", []):
            try:
                ref_bytes = download_user_photo(blob_path)
            except Exception:
                continue
            ref = embed_from_bytes(ref_bytes)
            if ref is None:
                continue
            # similitud coseno
            num = float(np.dot(live, ref))
            den = float(np.linalg.norm(live) * np.linalg.norm(ref) + 1e-12)
            sim = num / den
            if sim > best_sim:
                best_sim = sim

        login_ok = best_sim >= cos_sim_min
        return {
            "user_id": user_id,
            "login": "success" if login_ok else "failed",
            "via": "sface-local",
            "cosine_similarity": round(best_sim, 4),
            "cosine_threshold_used": cos_sim_min,
            "emotion": emotion
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/emotion/analyze")
async def emotion_analyze(file: UploadFile = File(...)):
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Archivo vacío")
        faces = detect_emotions_from_bytes(content)  # lista de caras con top_emotion & sentiment
        return {"facesCount": len(faces), "faces": faces}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))