# voice_vision/user_store.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any
import json
import time
import tempfile
import shutil

# Raíz del proyecto (la usa main.py como DATA_ROOT)
ROOT = Path(__file__).resolve().parents[1]

# Archivo "base de datos" JSON
DB_PATH = ROOT / "data" / "users.json"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# -------------------- Helpers internos --------------------

def _atomic_write(path: Path, data: str) -> None:
    """
    Escritura atómica: escribe a un tmp y luego reemplaza.
    Evita archivos corruptos si el proceso es interrumpido.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent)) as tf:
        tmp_name = tf.name
        tf.write(data)
        tf.flush()
        os_fsync_safe(tf)
    # Reemplazo atómico (en el mismo directorio)
    Path(tmp_name).replace(path)


def os_fsync_safe(tf) -> None:
    """Intenta hacer fsync del archivo temporal; si falla (p.ej. FS no lo soporta) ignora."""
    try:
        import os
        os.fsync(tf.fileno())
    except Exception:
        pass


def _read_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # Si se corrompió, haz un backup y arranca limpio
        bad = path.with_suffix(".json.bak")
        try:
            shutil.copy2(path, bad)
        except Exception:
            pass
        return {}


def _normalize_record(rec: Dict[str, Any] | None) -> Dict[str, Any]:
    """
    Asegura la forma canónica del registro:
    { "urls": [..], "files": [..], "blobs": [..], "updatedAt": 1712345678.12 }
    """
    base = {"urls": [], "files": [], "blobs": []}
    if not rec:
        base["updatedAt"] = time.time()
        return base

    out = {
        "urls": list(dict.fromkeys([x for x in rec.get("urls", []) if isinstance(x, str) and x.strip()])),
        "files": list(dict.fromkeys([x for x in rec.get("files", []) if isinstance(x, str) and x.strip()])),
        "blobs": list(dict.fromkeys([x for x in rec.get("blobs", []) if isinstance(x, str) and x.strip()])),
    }
    out["updatedAt"] = float(rec.get("updatedAt", time.time()))
    return out


# -------------------- API pública (compatible) --------------------

def load_users() -> Dict[str, Dict[str, List[str]]]:
    """
    Carga todos los usuarios desde el JSON.
    Siempre devuelve un dict. Si el archivo no existe o está corrupto, devuelve {}.
    """
    raw = _read_json_file(DB_PATH)
    # Normaliza por si hay basura/estructura vieja
    clean: Dict[str, Dict[str, List[str]]] = {}
    for uid, rec in raw.items():
        if not isinstance(uid, str):
            continue
        clean[uid] = _normalize_record(rec)  # type: ignore
    return clean


def save_users(data: Dict[str, Dict[str, List[str]]]) -> None:
    """
    Guarda el dict completo de usuarios de forma atómica.
    """
    # Normaliza todo antes de guardar
    to_save: Dict[str, Any] = {}
    for uid, rec in (data or {}).items():
        if not isinstance(uid, str):
            continue
        norm = _normalize_record(rec)
        norm["updatedAt"] = time.time()
        to_save[uid] = norm
    _atomic_write(DB_PATH, json.dumps(to_save, ensure_ascii=False, indent=2))


def add_user_blobs(user_id: str, blob_paths: List[str]) -> int:
    """
    Agrega rutas de blobs a un usuario.
    Mantiene compatibilidad con tu código existente (retorna cuántos nuevos se agregaron).
    """
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id inválido")
    new = [b for b in (blob_paths or []) if isinstance(b, str) and b.strip()]
    if not new:
        return 0

    data = load_users()
    rec = _normalize_record(data.get(user_id))
    # dedup + preserva orden
    before = set(rec["blobs"])
    for b in new:
        if b not in before:
            rec["blobs"].append(b)
            before.add(b)
    rec["updatedAt"] = time.time()

    data[user_id] = rec
    save_users(data)
    # Número de agregados realmente nuevos
    return sum(1 for b in new if b in before)


# -------------------- Utilidades adicionales (opcionales) --------------------

def get_user(user_id: str) -> Dict[str, Any]:
    """Obtiene (y normaliza) el registro de un usuario; si no existe, devuelve base vacía."""
    data = load_users()
    return _normalize_record(data.get(user_id))


def add_user_urls(user_id: str, urls: List[str]) -> int:
    """Agrega URLs públicas de fotos para el usuario."""
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id inválido")
    new = [u for u in (urls or []) if isinstance(u, str) and u.strip()]
    if not new:
        return 0
    data = load_users()
    rec = _normalize_record(data.get(user_id))
    before = set(rec["urls"])
    added = 0
    for u in new:
        if u not in before:
            rec["urls"].append(u)
            before.add(u)
            added += 1
    rec["updatedAt"] = time.time()
    data[user_id] = rec
    save_users(data)
    return added


def add_user_files(user_id: str, rel_paths: List[str]) -> int:
    """Agrega rutas de archivos (relativas a DATA_ROOT) al usuario."""
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id inválido")
    new = [p for p in (rel_paths or []) if isinstance(p, str) and p.strip()]
    if not new:
        return 0
    data = load_users()
    rec = _normalize_record(data.get(user_id))
    before = set(rec["files"])
    added = 0
    for p in new:
        if p not in before:
            rec["files"].append(p)
            before.add(p)
            added += 1
    rec["updatedAt"] = time.time()
    data[user_id] = rec
    save_users(data)
    return added


def list_users() -> List[str]:
    """Devuelve la lista de user_ids registrados."""
    return list(load_users().keys())


def remove_user(user_id: str) -> bool:
    """Elimina el registro completo del usuario (no borra blobs remotos)."""
    data = load_users()
    if user_id in data:
        data.pop(user_id, None)
        save_users(data)
        return True
    return False
