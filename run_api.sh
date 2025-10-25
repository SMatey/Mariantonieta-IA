#!/bin/bash
# Activa el entorno virtual si existe
if [ -d "venv" ]; then
  source venv/bin/activate
fi

# Inicia la aplicación FastAPI con Uvicorn
echo "Iniciando servidor de API en http://0.0.0.0:8000"
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
