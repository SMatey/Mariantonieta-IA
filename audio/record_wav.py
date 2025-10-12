# audio/record_wav.py
from __future__ import annotations
import argparse
import datetime as dt
import os
import wave
import sounddevice as sd

def record_wav(
    out_path: str,
    dur_seconds: float = 5.0,
    sample_rate: int = 16000,
    channels: int = 1,
):
    """
    Graba audio del micrófono y lo guarda como WAV PCM 16-bit (LINEAR16), mono, 16 kHz.
    """
    if channels != 1:
        raise ValueError("Para STT usaremos mono (channels=1).")
    if sample_rate != 16000:
        print(f"[Aviso] Recomendado 16000 Hz. Usando {sample_rate} Hz igualmente.")

    print(f"Grabando {dur_seconds} s a {sample_rate} Hz, {channels} canal...")
    audio = sd.rec(
        int(dur_seconds * sample_rate),
        samplerate=sample_rate,
        channels=channels,
        dtype='int16'  # 16-bit PCM
    )
    sd.wait()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with wave.open(out_path, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)          # 16 bits = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())

    print(f"Listo: {out_path}")

def list_devices():
    print(sd.query_devices())

def main():
    parser = argparse.ArgumentParser(
        description="Graba WAV PCM 16-bit mono 16 kHz para STT."
    )
    parser.add_argument("--dur", type=float, default=5.0, help="Duración en segundos (ej. 5.0).")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate (recomendado 16000).")
    parser.add_argument("--name", type=str, default="", help="Nombre del archivo (ej. sample.wav).")
    parser.add_argument("--list-devices", action="store_true", help="Lista dispositivos de audio y sale.")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    # Nombre por defecto con timestamp si no se indica
    filename = (
        f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        if args.name.strip() == ""
        else (args.name if args.name.lower().endswith(".wav") else f"{args.name}.wav")
    )

    out_path = os.path.join(os.path.dirname(__file__), filename)
    record_wav(out_path, dur_seconds=args.dur, sample_rate=args.sr, channels=1)

if __name__ == "__main__":
    main()