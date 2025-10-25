# 🧠 Primer Proyecto de Inteligencia Artificial — ITCR (Sede San Carlos)

Un sistema distribuido compuesto por microservicios de Machine Learning y un coordinador LLM local. El objetivo es combinar modelos predictivos tradicionales con capacidades conversacionales para que los usuarios exploren y consulten resultados mediante lenguaje natural.

## Destacado
- Arquitectura basada en microservicios (API REST para modelos)
- Coordinador inteligente que integra un LLM local (Ollama / LLaMA) para diálogo y explicación
- Modelos especializados para regresión, clasificación, recomendación y **series de tiempo**
- Nuevo modelo Prophet para predicciones temporales de Bitcoin
- Interfaz conversacional para consultas en lenguaje natural

## Modelos incluidos

| Modelo                           | Tipo                    | Propósito                                         | Estado    |
|:--------------------------------:|:-----------------------:|:--------------------------------------------------|:----------|
| `prophet_bitcoin_v2_*.pkl`      | **Series de Tiempo**   | **Predicción temporal del precio del Bitcoin**   | ✅ Activo |
| `knn_movie_recommendation_model.pkl` | Recomendación      | Sugerencia de películas según preferencias       | ✅ Activo |
| `random_forest_properties_*.pkl` | Regresión             | Predicción del precio de propiedades             | ✅ Activo |
| `ACV_decision_tree_model.pkl`    | Clasificación          | Detección de riesgo de accidente cerebrovascular | ✅ Activo |
| `bitcoin_random_forest_*.pkl`    | Regresión              | Predicción Bitcoin (modelo anterior)             | 📦 Legacy |

> **Nuevo**: El modelo de Bitcoin ahora usa **Prophet** para análisis de series temporales, permitiendo predicciones más precisas con tendencias estacionales y intervalos de confianza.

## Requisitos

- Python 3.9+ (recomendado)
- pip, virtualenv (o venv)
- Node.js + npm (para el frontend)
- macOS: Homebrew (para instalar Ollama si se usa)

### Dependencias principales nuevas:
- **Prophet**: Para modelos de series temporales
- **joblib**: Para carga optimizada de modelos ML
- **FastAPI**: APIs REST modernas y eficientes

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

### 1) Backend (API de modelos) — **Recomendado**

```bash
source venv/bin/activate
./run_api.sh  # Script optimizado que usa uvicorn correctamente
```

O manualmente:
```bash
source venv/bin/activate
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 2) Coordinador LLM

```bash
source venv/bin/activate
python llm/coordinator.py
```

### 3) Frontend (interfaz)

```bash
cd frontend
npm install
npm run dev
```

## Nuevas funcionalidades

### 🔮 Predicciones temporales de Bitcoin
El nuevo modelo Prophet permite consultas como:
- "¿Cuál será el precio de Bitcoin mañana?"
- "Predice Bitcoin para la próxima semana"
- "¿Qué precio tendrá Bitcoin el 1 de enero de 2025?"

### 🤖 Coordinador inteligente mejorado
- Extracción automática de fechas y parámetros
- Respuestas contextuales según el tipo de modelo
- Manejo de errores y respaldos automáticos

## Script unificado (run_all.sh)

Hay un script de conveniencia `run_all.sh` que arranca los componentes en segundo plano:

```bash
chmod +x run_all.sh
./run_all.sh
```

El script incluye:
- ✅ Activación automática del entorno virtual
- ✅ Inicio de la API optimizada con `uvicorn`
- ✅ Configuración correcta de puertos y hosts
- ✅ Manejo de errores y dependencias

## Estructura del proyecto

```
proyecto/
├── api/                     # 🔥 APIs REST para modelos ML
│   ├── main.py             # Entrada principal de la API
│   ├── constants.py        # 🆕 Constantes centralizadas
│   ├── routes/             # Rutas específicas por modelo
│   │   ├── bitcoin_api.py  # 🔮 API Prophet para Bitcoin
│   │   ├── movies_api.py   # 🎬 API recomendaciones
│   │   └── properties_api.py # 🏠 API predicción propiedades
│   └── core/               # Configuración central
├── llm/                    # 🧠 Coordinador LLM mejorado
│   └── coordinator.py      # 🆕 Coordinador con extracción inteligente
├── frontend/               # 💻 Interfaz React
├── models/                 # 🤖 Modelos ML entrenados
│   ├── prophet_bitcoin_v2_*.pkl  # 🆕 Modelo Prophet
│   ├── knn_movie_*.pkl           # Recomendaciones
│   └── random_forest_*.pkl       # Otros modelos
├── data/                   # 📊 Datasets
├── notebooks/              # 📚 Análisis exploratorio
└── tests/                  # 🧪 Pruebas automatizadas
```

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



