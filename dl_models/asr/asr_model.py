# dl_models/asr/asr_model.py

import json
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Ruta base = carpeta donde está este archivo
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "mariantonieta_asr_ctc.h5"
VOCAB_PATH = BASE_DIR / "vocab_characters.json"

# Parámetros de STFT (los mismos del notebook)
FRAME_LENGTH = 256
FRAME_STEP = 160
FFT_LENGTH = 384

# Variables globales para cargar una sola vez
_model = None
_char_to_num = None
_num_to_char = None


def _load_vocab():
    """Carga el vocabulario y crea las capas de mapeo carácter <-> índice."""
    global _char_to_num, _num_to_char

    if _char_to_num is not None and _num_to_char is not None:
        return

    if not VOCAB_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo de vocabulario: {VOCAB_PATH}")

    with VOCAB_PATH.open("r", encoding="utf-8") as f:
        characters = json.load(f)

    _char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
    _num_to_char = keras.layers.StringLookup(
        vocabulary=_char_to_num.get_vocabulary(),
        oov_token="",
        invert=True,
    )


def get_model():
    """Devuelve el modelo cargado en memoria (lo carga la primera vez)."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"No se encontró el archivo del modelo: {MODEL_PATH}")

        _load_vocab()
        # Para inferencia no necesitamos compilar con la pérdida CTC
        _model = keras.models.load_model(MODEL_PATH, compile=False)

    return _model


def preprocess_wav(file_path: str) -> tf.Tensor:
    """
    Convierte un archivo .wav en el espectrograma normalizado que espera el modelo.
    Devuelve un tensor con shape (1, time, freq_bins).
    """
    # Leer archivo
    file_bytes = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(file_bytes)  # [num_muestras, 1]

    # Quitar dimensión de canal y convertir a float32
    audio = tf.squeeze(audio, axis=-1)
    audio = tf.cast(audio, tf.float32)

    # STFT
    spectrogram = tf.signal.stft(
        audio,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        fft_length=FFT_LENGTH,
    )
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)

    # Normalización
    means = tf.math.reduce_mean(spectrogram, axis=1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, axis=1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)

    # Añadir dimensión batch
    spectrogram = tf.expand_dims(spectrogram, axis=0)
    return spectrogram


def _decode_batch_predictions(pred: np.ndarray) -> List[str]:
    """Convierte las predicciones CTC del modelo en texto legible."""
    _load_vocab()  # por si acaso
    input_len = np.ones(pred.shape[0]) * pred.shape[1]

    results = keras.backend.ctc_decode(
        pred,
        input_length=input_len,
        greedy=True,
    )[0][0]

    textos: List[str] = []
    for r in results:
        texto = tf.strings.reduce_join(_num_to_char(r)).numpy().decode("utf-8")
        textos.append(texto)

    return textos


def transcribe_wav(file_path: str) -> str:
    """
    Transcribe un archivo .wav a texto usando el modelo ASR.
    Devuelve un string con la transcripción (sin limpiar).
    """
    model = get_model()
    spec = preprocess_wav(file_path)
    pred = model.predict(spec)
    textos = _decode_batch_predictions(pred)

    if not textos:
        return ""

    return textos[0].strip()