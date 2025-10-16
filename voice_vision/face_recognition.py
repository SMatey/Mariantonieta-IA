# voice_vision/face_recognition.py
from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


class AzureFaceClient:
    """
    Cliente ligero para Azure Face API (Detect, Verify 1:1 y PersonGroups para Identify).
    - Si AZURE_FACE_DETECTION_ONLY=true, se deshabilitan métodos de Identify.
    - Si el servicio devuelve 403 en operaciones de Identify, se degrada a detección-only.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        key: Optional[str] = None,
        person_group_id: Optional[str] = None,
    ) -> None:
        ep = endpoint or os.getenv("AZURE_FACE_ENDPOINT") or ""
        k = key or os.getenv("AZURE_FACE_KEY") or ""
        pg = person_group_id or os.getenv("AZURE_FACE_PERSON_GROUP_ID") or "jarvistec_group"
        if not ep or not k:
            raise RuntimeError("Faltan AZURE_FACE_ENDPOINT o AZURE_FACE_KEY (revisa tu .env).")

        self.endpoint = ep.rstrip("/")
        self.key = k
        self.person_group_id = pg

        # Bandera detección-only configurable por .env
        self.supports_identity: bool = str(os.getenv("AZURE_FACE_DETECTION_ONLY", "false")).lower() not in (
            "1",
            "true",
            "yes",
        )

        self._headers_json = {
            "Ocp-Apim-Subscription-Key": self.key,
            "Content-Type": "application/json",
        }
        self._headers_octet = {
            "Ocp-Apim-Subscription-Key": self.key,
            "Content-Type": "application/octet-stream",
        }
        # Timeouts razonables
        self._timeout = httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=10.0)

    # -------------------- Detección --------------------
    def detect(self, image_url: str, return_face_id: bool = True) -> List[Dict[str, Any]]:
        """
        Detecta rostros en una imagen por URL. Si return_face_id=True retorna faceId.
        """
        url = f"{self.endpoint}/face/v1.0/detect"
        params = {
            "returnFaceId": str(return_face_id).lower(),
            "returnFaceLandmarks": "false",
            "recognitionModel": "recognition_04",
            # TTL opcional para faceId (segundos)
            "faceIdTimeToLive": "300",
        }
        with httpx.Client(timeout=self._timeout) as client:
            r = client.post(url, params=params, headers=self._headers_json, json={"url": image_url})
            r.raise_for_status()
            return r.json()

    def detect_bytes(self, content: bytes, return_face_id: bool = True) -> List[Dict[str, Any]]:
        """
        Detecta rostros a partir de bytes (p.ej. foto subida por multipart/form-data).
        """
        url = f"{self.endpoint}/face/v1.0/detect"
        params = {
            "returnFaceId": str(return_face_id).lower(),
            "returnFaceLandmarks": "false",
            "recognitionModel": "recognition_04",
            "faceIdTimeToLive": "300",
        }
        with httpx.Client(timeout=self._timeout) as client:
            r = client.post(url, params=params, headers=self._headers_octet, content=content)
            r.raise_for_status()
            return r.json()

    # -------------------- Person Group / Identify --------------------
    def ensure_person_group(self, name: str = "JarvisTEC Group") -> None:
        """
        Crea o asegura el PersonGroup configurado.
        Degrada a detección-only si la suscripción/región no soporta Identify (403).
        """
        if not self.supports_identity:
            return
        url = f"{self.endpoint}/face/v1.0/persongroups/{self.person_group_id}"
        payload = {"name": name, "recognitionModel": "recognition_04"}
        try:
            with httpx.Client(timeout=self._timeout) as client:
                r = client.put(url, headers=self._headers_json, json=payload)
                # 200/202 OK, 409 ya existe
                if r.status_code not in (200, 202, 409):
                    r.raise_for_status()
        except httpx.HTTPStatusError as he:
            if he.response.status_code == 403:
                self.supports_identity = False
                return
            raise

    def train_person_group(self) -> None:
        if not self.supports_identity:
            raise RuntimeError("IDENTIFICATION_UNAVAILABLE")
        url = f"{self.endpoint}/face/v1.0/persongroups/{self.person_group_id}/train"
        with httpx.Client(timeout=self._timeout) as client:
            r = client.post(url, headers=self._headers_json)
            r.raise_for_status()

    def get_training_status(self) -> Dict[str, Any]:
        if not self.supports_identity:
            raise RuntimeError("IDENTIFICATION_UNAVAILABLE")
        url = f"{self.endpoint}/face/v1.0/persongroups/{self.person_group_id}/training"
        with httpx.Client(timeout=self._timeout) as client:
            r = client.get(url, headers=self._headers_json)
            r.raise_for_status()
            return r.json()

    def create_person(self, name: str) -> str:
        if not self.supports_identity:
            raise RuntimeError("IDENTIFICATION_UNAVAILABLE")
        url = f"{self.endpoint}/face/v1.0/persongroups/{self.person_group_id}/persons"
        with httpx.Client(timeout=self._timeout) as client:
            r = client.post(url, headers=self._headers_json, json={"name": name})
            r.raise_for_status()
            return r.json()["personId"]

    def add_person_face(self, person_id: str, image_url: str) -> str:
        if not self.supports_identity:
            raise RuntimeError("IDENTIFICATION_UNAVAILABLE")
        url = f"{self.endpoint}/face/v1.0/persongroups/{self.person_group_id}/persons/{person_id}/persistedFaces"
        with httpx.Client(timeout=self._timeout) as client:
            r = client.post(url, headers=self._headers_json, json={"url": image_url})
            r.raise_for_status()
            return r.json()["persistedFaceId"]

    def list_persons(self) -> List[Dict[str, Any]]:
        if not self.supports_identity:
            return []  # en modo detección-only devolvemos []
        url = f"{self.endpoint}/face/v1.0/persongroups/{self.person_group_id}/persons"
        with httpx.Client(timeout=self._timeout) as client:
            r = client.get(url, headers=self._headers_json)
            r.raise_for_status()
            return r.json()

    # -------------------- Verify 1:1 --------------------
    def verify_faces(self, face_id1: str, face_id2: str) -> Dict[str, Any]:
        """
        Verifica si dos faceId pertenecen a la misma persona.
        Respuesta: {"isIdentical": bool, "confidence": float}
        """
        url = f"{self.endpoint}/face/v1.0/verify"
        payload = {"faceId1": face_id1, "faceId2": face_id2}
        with httpx.Client(timeout=self._timeout) as client:
            r = client.post(url, headers=self._headers_json, json=payload)
            r.raise_for_status()
            return r.json()

    # -------------------- Identify (grupo) --------------------
    def identify(
        self,
        face_ids: List[str],
        max_candidates: int = 1,
        confidence_threshold: float = 0.65,
    ) -> List[Dict[str, Any]]:
        """
        Identifica rostros contra el PersonGroup configurado.
        Devuelve lista de resultados (uno por faceId) con candidatos.
        """
        if not self.supports_identity:
            raise RuntimeError("IDENTIFICATION_UNAVAILABLE")

        url = f"{self.endpoint}/face/v1.0/identify"
        payload = {
            "personGroupId": self.person_group_id,
            "faceIds": face_ids,
            "maxNumOfCandidatesReturned": max_candidates,
            "confidenceThreshold": confidence_threshold,
        }
        with httpx.Client(timeout=self._timeout) as client:
            r = client.post(url, headers=self._headers_json, json=payload)
            r.raise_for_status()
            return r.json()

    def train_and_wait(self, poll_seconds: float = 1.0, timeout_seconds: float = 60.0) -> str:
        """
        Lanza entrenamiento del PersonGroup y espera a que termine.
        """
        if not self.supports_identity:
            raise RuntimeError("IDENTIFICATION_UNAVAILABLE")

        self.train_person_group()
        start = time.time()
        while time.time() - start < timeout_seconds:
            status = self.get_training_status()
            st = (status or {}).get("status")
            if st == "succeeded":
                return "succeeded"
            if st == "failed":
                raise RuntimeError(f"Entrenamiento falló: {status}")
            time.sleep(poll_seconds)
        raise TimeoutError("Timeout esperando entrenamiento.")
