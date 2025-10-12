from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import json

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "users.json"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def load_users() -> Dict[str, Dict[str, List[str]]]:
    if not DB_PATH.exists():
        return {}
    return json.loads(DB_PATH.read_text(encoding="utf-8"))

def save_users(data: Dict[str, Dict[str, List[str]]]) -> None:
    DB_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def add_user_blobs(user_id: str, blob_paths: List[str]) -> int:
    data = load_users()
    rec = data.get(user_id, {"urls": [], "files": [], "blobs": []})
    rec.setdefault("blobs", [])
    rec["blobs"].extend(blob_paths)
    data[user_id] = rec
    save_users(data)
    return len(blob_paths)
