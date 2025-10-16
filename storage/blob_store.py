# storage/blob_store.py
from __future__ import annotations
import os
import uuid
from typing import Optional

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.storage.blob import BlobServiceClient, ContentSettings


# -------------------- Config / Cliente --------------------

CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER", "faces")

# Se admiten 3 modos:
# 1) AZURE_STORAGE_CONNECTION_STRING
# 2) AZURE_STORAGE_ACCOUNT_URL + AZURE_STORAGE_SAS_TOKEN
# 3) AZURE_STORAGE_ACCOUNT_NAME + AZURE_STORAGE_ACCOUNT_KEY
def _build_service_client() -> BlobServiceClient:
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if conn_str:
        return BlobServiceClient.from_connection_string(conn_str)

    account_url = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
    sas = os.getenv("AZURE_STORAGE_SAS_TOKEN")
    if account_url and sas:
        return BlobServiceClient(account_url=account_url.rstrip("/"), credential=sas)

    account = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
    if account and key:
        return BlobServiceClient(
            account_url=f"https://{account}.blob.core.windows.net",
            credential=key
        )

    raise RuntimeError(
        "Config de Azure Blob incompleta. Provee "
        "AZURE_STORAGE_CONNECTION_STRING  o  "
        "(AZURE_STORAGE_ACCOUNT_URL + AZURE_STORAGE_SAS_TOKEN)  o  "
        "(AZURE_STORAGE_ACCOUNT_NAME + AZURE_STORAGE_ACCOUNT_KEY)."
    )


_service: BlobServiceClient = _build_service_client()
_container = _service.get_container_client(CONTAINER)

def ensure_container() -> None:
    try:
        _container.create_container()
    except ResourceExistsError:
        pass


# -------------------- Utilidades --------------------

def _guess_image_mime(content: bytes) -> str:
    # Detección mínima por magic numbers
    if len(content) >= 12:
        b0_12 = content[:12]
        if b0_12.startswith(b"\xFF\xD8\xFF"):
            return "image/jpeg"
        if b0_12.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if b0_12.startswith(b"GIF8"):
            return "image/gif"
        if b0_12[:4] == b"RIFF" and b0_12[8:12] == b"WEBP":
            return "image/webp"
        if b0_12.startswith(b"BM"):
            return "image/bmp"
        # HEIC/HEIF: ftyp....heic/heif
        if b0_12[4:8] == b"ftyp" and any(x in b0_12 for x in (b"heic", b"heix", b"hevc", b"hevx", b"mif1")):
            return "image/heic"
    return "image/jpeg"  # fallback seguro


def blob_url(blob_path: str) -> str:
    """Devuelve la URL del blob (útil si el contenedor es público o usas SAS)."""
    return _container.get_blob_client(blob_path).url


# -------------------- API pública --------------------

def upload_user_photo(user_id: str, content: bytes, *, content_type: Optional[str] = None) -> str:
    """
    Sube la foto y retorna la ruta del blob (p. ej. 'users/maria/<id>.jpg').
    Usa content_type si se entrega; si no, lo infiere (básico).
    """
    if not content:
        raise ValueError("Contenido vacío")

    ensure_container()

    ct = content_type or _guess_image_mime(content)
    ext = {
        "image/jpeg": "jpg",
        "image/png": "png",
        "image/gif": "gif",
        "image/webp": "webp",
        "image/bmp": "bmp",
        "image/heic": "heic",
    }.get(ct, "jpg")

    blob_path = f"users/{user_id}/{uuid.uuid4().hex}.{ext}"
    blob = _container.get_blob_client(blob_path)
    blob.upload_blob(
        content,
        overwrite=True,
        content_settings=ContentSettings(content_type=ct)
    )
    return blob_path  # mantenemos compatibilidad con el resto del código


def download_user_photo(blob_path: str) -> bytes:
    """Descarga y devuelve los bytes del blob en 'blob_path'."""
    try:
        return _container.get_blob_client(blob_path).download_blob().readall()
    except ResourceNotFoundError:
        raise FileNotFoundError(f"Blob no encontrado: {blob_path}")
