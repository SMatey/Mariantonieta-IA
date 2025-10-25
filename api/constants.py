"""
Constantes para las APIs - Rutas de archivos y configuraciones
"""
from pathlib import Path

# Directorio base (la carpeta raíz del proyecto)
BASE_DIR = Path(__file__).resolve().parent.parent

# Directorio donde guardar/leer datos
DATA_DIR = BASE_DIR / "data" / "raw"

# Nombres de archivos CSV
BITCOIN_CSV_FILENAME = "bitcoin_price_Training.csv"
FLIGHTS_CSV_FILENAME = "DelayedFlights.csv"

# Rutas completas a los archivos CSV
CSV_BITCOIN_PATH = DATA_DIR / BITCOIN_CSV_FILENAME
CSV_FLIGHTS_PATH = DATA_DIR / FLIGHTS_CSV_FILENAME

# Directorio de modelos
MODELS_DIR = BASE_DIR / "models"
ML_MODELS_PATH = BASE_DIR / "ml_models"