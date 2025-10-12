# api/stt.py
from __future__ import annotations

import os
import io
import wave
import contextlib
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from google.cloud import speech
from google.oauth2 import service_account
from dotenv import load_dotenv

# Cargar .env si existe
load_dotenv()

APP_NAME = "Mariantonieta-IA STT"
DEFAULT_LANGUAGE = os.getenv("GCP_SPEECH_LANGUAGE", "es-CR")
DEFAULT_ENCODING = os.getenv("GCP_SPEECH_ENCODING", "LINEAR16").upper()

app = FastAPI(title=APP_NAME, version="0.1.0", docs_url="/docs", redoc_url="/redoc")


class STTResponse(BaseModel):
    transcript: str
    confidence: float
    duration_ms: int
    detected_sample_rate: int
    channels: int
    language_code: str


def _read_wav_info(raw: bytes) -> tuple[int, int, int]:
    """Devuelve (sample_rate_hz, channels, duration_ms)."""
    with contextlib.closing(wave.open(io.BytesIO(raw), "rb")) as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()  # bytes por muestra (2 == 16 bits)
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        if sampwidth != 2:
            raise ValueError(
                f"WAV no es LINEAR16 (16-bit PCM). sampwidth={sampwidth*8} bits."
            )
        duration_sec = nframes / float(framerate) if framerate else 0.0
        duration_ms = int(duration_sec * 1000)
        return framerate, channels, duration_ms


def _encoding_enum(enc_str: str) -> speech.RecognitionConfig.AudioEncoding:
    enc_str = enc_str.upper().strip()
    if enc_str == "LINEAR16":
        return speech.RecognitionConfig.AudioEncoding.LINEAR16
    if enc_str == "FLAC":
        return speech.RecognitionConfig.AudioEncoding.FLAC
    if enc_str == "MULAW":
        return speech.RecognitionConfig.AudioEncoding.MULAW
    if enc_str == "ALAW":
        return speech.RecognitionConfig.AudioEncoding.ALAW
    return speech.RecognitionConfig.AudioEncoding.LINEAR16


def _make_speech_client() -> speech.SpeechClient:
    """
    Crea el cliente usando el .env, sin depender de exports en la terminal.
    Reglas:
      1) GOOGLE_APPLICATION_CREDENTIALS (ruta al JSON) -> requerido que exista.
      2) Si no está, intenta ADC por defecto (solo GCP). 
    """
    sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if sa_path:
        if not os.path.isabs(sa_path):
            # Normaliza la ruta relativa al cwd de ejecución
            sa_path = os.path.abspath(sa_path)
        if not os.path.exists(sa_path):
            raise HTTPException(
                status_code=500,
                detail=f"Credencial no encontrada en ruta '{sa_path}'. "
                       f"Revisa GOOGLE_APPLICATION_CREDENTIALS en tu .env."
            )
        creds = service_account.Credentials.from_service_account_file(sa_path)
        return speech.SpeechClient(credentials=creds)

    # Fallback (solo funcionará dentro de GCP con ADC configurado)
    return speech.SpeechClient()


@app.get("/health")
def health():
    return {"status": "ok", "app": APP_NAME}


@app.post("/stt", response_model=STTResponse)
async def stt(file: UploadFile = File(...)):
    """Recibe un WAV (LINEAR16) y devuelve el transcript."""
    if not file:
        raise HTTPException(status_code=400, detail="Falta el archivo de audio.")

    ct = (file.content_type or "").lower()
    if not (ct.endswith("wav") or "octet-stream" in ct or "audio" in ct):
        raise HTTPException(
            status_code=415,
            detail=f"Content-Type no soportado ({file.content_type}). Sube un WAV PCM (LINEAR16).",
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="El archivo está vacío.")

    try:
        sample_rate, channels, duration_ms = _read_wav_info(raw)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"El archivo no parece ser WAV PCM válido (LINEAR16). Detalle: {e}",
        )

    # Cliente con credenciales del .env
    client = _make_speech_client()

    config = speech.RecognitionConfig(
        encoding=_encoding_enum(DEFAULT_ENCODING),
        sample_rate_hertz=sample_rate,
        audio_channel_count=channels,
        language_code=DEFAULT_LANGUAGE,
        enable_automatic_punctuation=True,
    )
    audio = speech.RecognitionAudio(content=raw)

    try:
        response = client.recognize(config=config, audio=audio)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error de Speech-to-Text: {e}")

    transcripts: List[str] = []
    confidences: List[float] = []

    for result in response.results:
        if not result.alternatives:
            continue
        best = result.alternatives[0]
        transcripts.append(best.transcript.strip())
        if best.confidence:
            confidences.append(float(best.confidence))

    if not transcripts:
        raise HTTPException(status_code=204, detail="Sin hipótesis de transcripción.")

    final_transcript = " ".join(t for t in transcripts if t).strip()
    avg_confidence = float(sum(confidences) / len(confidences)) if confidences else 0.0

    return STTResponse(
        transcript=final_transcript,
        confidence=avg_confidence,
        duration_ms=duration_ms,
        detected_sample_rate=sample_rate,
        channels=channels,
        language_code=DEFAULT_LANGUAGE,
    )