from __future__ import annotations
import os, uuid
from azure.storage.blob import BlobServiceClient

ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
KEY     = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER", "face-refs")

if not ACCOUNT or not KEY:
    raise RuntimeError("Faltan AZURE_STORAGE_ACCOUNT_NAME / AZURE_STORAGE_ACCOUNT_KEY")

_service = BlobServiceClient(
    account_url=f"https://{ACCOUNT}.blob.core.windows.net",
    credential=KEY
)
_container = _service.get_container_client(CONTAINER)
try:
    _container.create_container()  # si ya existe, lanzará excepción; la ignoramos
except Exception:
    pass

def upload_user_photo(user_id: str, content: bytes) -> str:
    """Sube la foto y retorna la ruta del blob (p. ej. users/maria/<id>.jpg)."""
    blob_path = f"users/{user_id}/{uuid.uuid4().hex}.jpg"
    _container.get_blob_client(blob_path).upload_blob(content, overwrite=True)
    return blob_path

def download_user_photo(blob_path: str) -> bytes:
    return _container.get_blob_client(blob_path).download_blob().readall()
