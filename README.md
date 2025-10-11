# 🧠 Primer Proyecto de Inteligencia Artificial — ITCR (Sede San Carlos)

Un sistema distribuido compuesto por microservicios de Machine Learning y un coordinador LLM local. El objetivo es combinar modelos predictivos tradicionales con capacidades conversacionales para que los usuarios exploren y consulten resultados mediante lenguaje natural.

Destacado
- Arquitectura basada en microservicios (API REST para modelos).
- Un coordinador que integra un LLM local (p. ej. Ollama / LLaMA) para diálogo y explicación.
- Conjuntos de modelos para regresión, clasificación y recomendación listos para desplegar.

## Modelos incluidos

| Archivo                  | Tipo          | Propósito                                         |
|-------------------------:|:-------------:|:--------------------------------------------------|
| `bitcoin_model.pkl`      | Regresión     | Predicción del precio del Bitcoin                  |
| `movies_model.pkl`       | Recomendación | Sugerencia de películas según preferencias         |
| `house_model.pkl`        | Regresión     | Predicción del precio de viviendas                 |
| `stroke_model.pkl`       | Clasificación | Detección de riesgo de accidente cerebrovascular   |
| `flight_delay_model.pkl` | Regresión     | Predicción de retrasos en vuelos                   |

> Nota: Los modelos listados suelen estar en la carpeta `models/` (ver estructura del proyecto).

## Requisitos

- Python 3.10+ (recomendado)
- pip, virtualenv (o venv)
- Node.js + npm (para el frontend)
- macOS: Homebrew (para instalar Ollama si se usa)

Instala dependencias Python:

```bash
python3 -m venv venv
source venv/bin/activate   # macOS / Linux (zsh compatible)
pip install -r requirements.txt
```

## Variables de entorno (recomendadas)

Exporta estas variables en tu shell o crea un archivo `.env` (no incluir en Git):

```bash
export ENV=development
export AZURE_FACE_KEY="tu_api_key"
export AZURE_FACE_ENDPOINT="https://<endpoint>.cognitiveservices.azure.com/"
export LLM_HOST="http://localhost:8001"
export API_BASE_URL="http://localhost:8080"
```

## Configuración del LLM (opcional: Ollama)

Si quieres correr un LLM local con Ollama (opcional):

```bash
# macOS (Homebrew)
brew install ollama
ollama pull llama3:3b

# Ejecutar el modelo para pruebas
ollama run llama3:3b
```

Dependiendo de tu arquitectura y recursos, puedes elegir otro modelo o servicio. El coordinador LLM del repo asume que hay un endpoint local en `LLM_HOST`.

## Ejecución — servicios individuales

1) Backend (API de modelos)

```bash
source venv/bin/activate
uvicorn api.main:app --reload --port 8000
```

2) Coordinador LLM

```bash
python3 llm/coordinator.py
```

3) Frontend (interfaz)

```bash
cd frontend || cd interface  # revisar el nombre correcto de la carpeta en tu repo
npm install
npm run dev
```

Si la carpeta del frontend se llama `interface` en tu repositorio original, usa esa en lugar de `frontend`.

## Script unificado (run_all.sh)

Hay un script de conveniencia `run_all.sh` que arranca los componentes en segundo plano. Antes de usarlo, asegúrate de que los comandos existentes (uvicorn, ollama, python) funcionen desde tu shell.

Contenido de ejemplo (ya incluido en el repo):

```bash
#!/bin/bash
source venv/bin/activate
echo "🚀 Iniciando API de Machine Learning..."
uvicorn api.main:app --port 8000 &

echo "🧠 Iniciando LLM (Ollama) si está instalado..."
ollama serve &

echo "🔗 Iniciando coordinador LLM..."
python3 llm/coordinator.py &

echo "💻 Iniciando frontend..."
cd frontend || cd interface
npm run dev
```

Haz el script ejecutable y ejecútalo:

```bash
chmod +x run_all.sh
./run_all.sh
```

## Estructura del proyecto (resumen)

- `api/` — microservicio(s) y endpoints para los modelos ML
- `llm/` — coordinador/puente entre los modelos y el LLM
- `frontend/` o `interface/` — interfaz React (UI)
- `models/` — modelos preentrenados (.pkl u otros)
- `data/` — datos raw/processed/examples
- `notebooks/` — notebooks exploratorios
- `tests/` — pruebas unitarias y de integración

## Solución de problemas común

- "ModuleNotFoundError" — activa el venv y reinstala dependencias: `pip install -r requirements.txt`.
- Problemas con Ollama — verificar versión y que el servicio esté corriendo: `ollama ps` / `ollama logs`.
- Frontend no arranca — revisa `node` y `npm` instalados, luego `npm install` y `npm run dev`.

## Buenas prácticas

- Mantén credenciales fuera del repositorio (.env en .gitignore).
- Versiona modelos con nombres que incluyan versión y fecha cuando sean reentrenados.


## Contacto y créditos

- Autor/es: Sebastian Matey, Liz Salazar, Roosevelt Pérez — Instituto Tecnológico de Costa Rica (Sede San Carlos)
- Repo: ProyectoIA



