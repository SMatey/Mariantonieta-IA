from pathlib import Path

# Directorio base (la carpeta donde está este archivo)
BASE_DIR = Path(__file__).resolve().parent.parent


# Directorio donde guardar/leer datos
DATA_DIR = BASE_DIR / "data/raw"
# Nombre del archivo CSV con precios de bitcoin
BITCOIN_CSV_FILENAME = "bitcoin_price_Training - Training.csv"

FLIGHTS_CSV_FILENAME = "DelayedFlights.csv"
# Ruta completa al CSV (Path object)
CSV_BITCOIN_PATH = DATA_DIR / BITCOIN_CSV_FILENAME
CSV_FLIGHTS_PATH = DATA_DIR / FLIGHTS_CSV_FILENAME

