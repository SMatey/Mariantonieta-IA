# voice_vision/main.py
# voice_vision/main.py
from __future__ import annotations
from pathlib import Path
from typing import List
import sys
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- Asegura ruta del proyecto y carga .env ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))           # hace visible 'storage/' y el paquete raíz
load_dotenv(PROJECT_ROOT / ".env")

# --- Imports con prefijo de paquete ---
from voice_vision.face_recognition import AzureFaceClient
from voice_vision.google_emotion import detect_emotion_from_bytes
from voice_vision.user_store import load_users, save_users, add_user_blobs, ROOT as DATA_ROOT
from storage.blob_store import upload_user_photo, download_user_photo


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
    """
    Detección de rostros (siempre disponible). Devuelve bbox y faceId por rostro.
    """
    try:
        faces = client.detect(req.image, return_face_id=True)
        return {"faces": faces}
    except httpx.HTTPStatusError as he:
        raise HTTPException(status_code=he.response.status_code, detail=he.response.text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/faces/persons")
def list_persons():
    """
    Lista de personas enroladas. En modo detección-only devolverá [].
    """
    try:
        return client.list_persons()
    except httpx.HTTPStatusError as he:
        raise HTTPException(status_code=he.response.status_code, detail=he.response.text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/faces/enroll")
def enroll(req: EnrollRequest):
    """
    Enrolar persona (requiere Identify). En detección-only devolverá 501.
    """
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
    """
    Identificar persona (si Identify está disponible). En detección-only retorna detecciones y un mensaje.
    """
    try:
        faces = client.detect(req.image, return_face_id=True)
        if not faces:
            return {"faces": [], "results": [], "message": "No se detectaron rostros"}

        # Si no hay Identify disponible, devolvemos solo detecciones
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
            if not f.content_type.startswith("image/"):
                continue
            content = await f.read()
            if content:
                path = upload_user_photo(user_id, content)
                blobs.append(path)
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
    threshold: float = 0.70,
):
    """
    Login usando foto (multipart/form-data). Verifica via Azure Face Verify (1:1)
    contra referencias del usuario:
      - blobs   (Azure Blob Storage, bytes)
      - files   (rutas locales, bytes)
      - urls    (en la nube, por URL)
    Siempre devuelve emoción (Google Vision).
    """
    try:
        # --- 0) Validación de entrada ---
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Archivo vacío")

        # --- 1) faceId de la selfie (bytes) ---
        faces_live = client.detect_bytes(content, return_face_id=True)
        if not faces_live:
            raise HTTPException(status_code=422, detail="No se detectó rostro en la selfie")
        live_id = faces_live[0].get("faceId")

        # --- 2) Cargar referencias del usuario ---
        users = load_users()
        rec = users.get(user_id)
        if not rec:
            raise HTTPException(status_code=404, detail="Usuario no enrolado")

        ref_blobs: List[str] = rec.get("blobs", [])  # Azure Blob: rutas de blob
        ref_files: List[str] = rec.get("files", [])  # locales: rutas relativas
        ref_urls:  List[str] = rec.get("urls", [])   # URLs públicas

        # --- 3) Comparar con Verify 1:1 ---
        best_conf, matched = 0.0, False

        # 3a) Refs en Azure Blob (bytes)
        for blob_path in ref_blobs:
            try:
                ref_bytes = download_user_photo(blob_path)
            except Exception:
                continue
            ref_faces = client.detect_bytes(ref_bytes, return_face_id=True)
            if not ref_faces:
                continue
            ref_id = ref_faces[0].get("faceId")
            try:
                verify = client.verify_faces(live_id, ref_id)
            except httpx.HTTPStatusError as he:
                if he.response.status_code == 403:
                    emotion_only = detect_emotion_from_bytes(content)
                    return {
                        "user_id": user_id,
                        "login": "unavailable",
                        "message": "Azure Verify no disponible en esta suscripción/región",
                        "emotion": emotion_only,
                    }
                raise
            conf = float(verify.get("confidence", 0.0))
            if verify.get("isIdentical") and conf > best_conf:
                best_conf, matched = conf, True
                if conf >= threshold:
                    break

        # 3b) Refs locales (bytes)
        if not matched:
            for rel in ref_files:
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
                        break

        # 3c) Refs por URL
        if not matched:
            for ref_url in ref_urls:
                ref_faces = client.detect(ref_url, return_face_id=True)
                if not ref_faces:
                    continue
                ref_id = ref_faces[0].get("faceId")
                verify = client.verify_faces(live_id, ref_id)
                conf = float(verify.get("confidence", 0.0))
                if verify.get("isIdentical") and conf > best_conf:
                    best_conf, matched = conf, True
                    if conf >= threshold:
                        break

        # --- 4) Emoción (siempre) ---
        emotion = detect_emotion_from_bytes(content)

        # --- 5) Respuesta ---
        return {
            "user_id": user_id,
            "login": "success" if matched else "failed",
            "confidence": round(best_conf, 4),
            "threshold_used": threshold,
            "emotion": emotion,
        }

    except HTTPException:
        raise
    except httpx.HTTPStatusError as he:
        # Propaga código HTTP real de Azure si no lo capturamos arriba
        raise HTTPException(status_code=he.response.status_code, detail=he.response.text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


