# voice_vision/main.py
from __future__ import annotations
from pathlib import Path
from typing import List
import sys
import httpx
from dotenv import load_dotenv
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Embeddings (SFace)
from voice_vision.face_embed import embed_from_bytes

# Cosmos + Blob (IMPORTS DESPUÉS de load_dotenv)
from storage.cosmos_store import upsert_face_vector, topk_by_vector
from storage.blob_store import upload_user_photo, download_user_photo

# Clientes utilitarios
from voice_vision.face_recognition import AzureFaceClient
from voice_vision.google_emotion import detect_emotion_from_bytes
from voice_vision.user_store import load_users, save_users, add_user_blobs, ROOT as DATA_ROOT


app = FastAPI(title="Maria Antonieta Face API", version="0.1.0")

# CORS abierto en desarrollo (ajusta dominios en prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
        "personGroup": client.person_group_id,
        "supportsIdentity": getattr(client, "supports_identity", True),
    }


@app.post("/faces/detect")
def detect(req: IdentifyRequest):
    """Detección de rostros (siempre disponible). Devuelve bbox y faceId por rostro."""
    try:
        faces = client.detect(req.image, return_face_id=True)
        return {"faces": faces}
    except httpx.HTTPStatusError as he:
        raise HTTPException(status_code=he.response.status_code, detail=he.response.text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/faces/persons")
def list_persons():
    """Lista de personas enroladas. En modo detección-only devolverá []."""
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
    try:
        blobs = []
        for f in files:
            if not f.content_type or not f.content_type.startswith("image/"):
                continue
            content = await f.read()
            if not content:
                continue

            # 1) sube a blob y guarda ruta
            path = upload_user_photo(user_id, content)
            blobs.append(path)

            # 2) saca embedding y súbelo a Cosmos
            emb = embed_from_bytes(content)
            if emb is not None:
                upsert_face_vector(user_id, path, emb.tolist())

        if not blobs:
            raise HTTPException(status_code=400, detail="No se recibieron imágenes válidas.")

        stored = add_user_blobs(user_id, blobs)
        return {"ok": True, "user_id": user_id, "stored_files": stored}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/auth/login_multipart")
async def login_multipart(
    user_id: str = Form(...),
    file: UploadFile = File(...),
    threshold: float = 0.70,    # umbral de Azure Verify (si aplica)
    cos_dist_max: float = 0.30, # umbral de distancia para Cosmos (cosine distance aprox)
    topk: int = 5
):
    """
    Login con 2 rutas:
      A) Azure Verify (si está disponible)
      B) Embeddings locales (SFace) + Cosmos Vector Search (sin Limited Access)
    Siempre devuelve emoción por Vision.
    """
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Archivo vacío")

        # 0) Emoción (no bloquea el login)
        emotion = detect_emotion_from_bytes(content)

        # 1) Azure Identify/Verify primero (si está disponible)
        if getattr(client, "supports_identity", True):
            try:
                faces_live = client.detect_bytes(content, return_face_id=True)
                if faces_live:
                    live_id = faces_live[0].get("faceId")
                    users = load_users()
                    rec = users.get(user_id)
                    if rec:
                        # blobs
                        best_conf = 0.0
                        matched = False
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
                                best_conf, matched = conf, True
                                if conf >= threshold:
                                    return {
                                        "user_id": user_id,
                                        "login": "success",
                                        "via": "azure-verify",
                                        "confidence": round(best_conf, 4),
                                        "threshold_used": threshold,
                                        "emotion": emotion
                                    }
                        # files (en disco)
                        if not matched:
                            for rel in rec.get("files", []):
                                path = (DATA_ROOT / rel).resolve()
                                if not path.exists():
                                    continue
                                ref_bytes = path.read_bytes()
                                ref_faces = client.detect_bytes(ref_bytes, return_face_id=True)
                                if not ref_faces:
                                    continue
                                ref_id = ref_faces[0].get("faceId")
                                verify = client.verify_faces(live_id, ref_id)
                                conf = float(verify.get("confidence", 0.0))
                                if verify.get("isIdentical") and conf > best_conf:
                                    best_conf, matched = conf, True
                                    if conf >= threshold:
                                        return {
                                            "user_id": user_id,
                                            "login": "success",
                                            "via": "azure-verify",
                                            "confidence": round(best_conf, 4),
                                            "threshold_used": threshold,
                                            "emotion": emotion
                                        }
                        # urls
                        if not matched:
                            for ref_url in rec.get("urls", []):
                                ref_faces = client.detect(ref_url, return_face_id=True)
                                if not ref_faces:
                                    continue
                                ref_id = ref_faces[0].get("faceId")
                                verify = client.verify_faces(live_id, ref_id)
                                conf = float(verify.get("confidence", 0.0))
                                if verify.get("isIdentical") and conf > best_conf:
                                    best_conf, matched = conf, True
                                    if conf >= threshold:
                                        return {
                                            "user_id": user_id,
                                            "login": "success",
                                            "via": "azure-verify",
                                            "confidence": round(best_conf, 4),
                                            "threshold_used": threshold,
                                            "emotion": emotion
                                        }
            except httpx.HTTPStatusError as he:
                # Si Azure devuelve 403/400/404 seguimos por embeddings
                if he.response.status_code not in (403, 400, 404):
                    raise

        # 2) Embeddings + Cosmos (sin Limited Access)
        emb = embed_from_bytes(content)
        if emb is None:
            raise HTTPException(status_code=422, detail="No se detectó rostro en la selfie")

        # kNN en Cosmos, **por partición del usuario** para evitar 2206
        hits = topk_by_vector(emb.tolist(), k=topk, user_id=user_id)

        best = hits[0] if hits else None
        if not best:
            return {
                "user_id": user_id,
                "login": "failed",
                "via": "cosmos-embeddings",
                "reason": "sin candidatos",
                "emotion": emotion
            }

        dist = float(best.get("distance", 1.0))
        login_ok = (best.get("userId") == user_id) and (dist <= cos_dist_max)

        return {
            "user_id": user_id,
            "login": "success" if login_ok else "failed",
            "via": "cosmos-embeddings",
            "distance": round(dist, 4),
            "cosine_threshold_used": cos_dist_max,
            "top_hit_user": best.get("userId"),
            "emotion": emotion,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
