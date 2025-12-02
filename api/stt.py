# api/stt.py

from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, UploadFile, File, HTTPException

from dl_models.asr import asr_model

router = APIRouter(prefix="/stt", tags=["speech-to-text"])

# Carpeta donde guardaremos temporalmente los audios subidos
AUDIO_UPLOAD_DIR = Path("audio/uploads")
AUDIO_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/", summary="Transcribir audio WAV a texto")
async def speech_to_text(file: UploadFile = File(...)):
    # Validar tipo de archivo (básico)
    if file.content_type not in (
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/vnd.wave",
    ):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos WAV")

    # Guardar archivo temporalmente
    tmp_name = f"{uuid4()}.wav"
    tmp_path = AUDIO_UPLOAD_DIR / tmp_name

    data = await file.read()
    tmp_path.write_bytes(data)

    # Llamar al modelo
    try:
        transcript = asr_model.transcribe_wav(str(tmp_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al transcribir audio: {e}")

    return {
        "filename": file.filename,
        "transcript": transcript,
    }