# 🧠 Primer Proyecto de Inteligencia Artificial — ITCR (Sede San Carlos)

Un sistema distribuido de microservicios que integra modelos de Machine Learning con capacidades conversacionales mediante un coordinador LLM local. La arquitectura permite consultar y explorar resultados predictivos usando lenguaje natural.

## Características Técnicas

- **Arquitectura de Microservicios**: APIs REST independientes para cada modelo ML
- **Coordinador LLM Inteligente**: Integración con Ollama/LLaMA para procesamiento de lenguaje natural
- **Sistema de Logging Centralizado**: Monitoreo y debugging profesional con rotación automática
- **Extracción Inteligente de Parámetros**: Parsing automático de consultas en lenguaje natural
- **Interfaz Conversacional**: Comunicación natural entre usuario y modelos predictivos
- **Manejo Robusto de Errores**: Fallbacks automáticos y validación de entrada

## Modelos de Machine Learning

| Modelo                              | Algoritmo               | Dominio de Aplicación                            | Estado    |
|:-----------------------------------:|:-----------------------:|:--------------------------------------------------|:----------|
| `prophet_bitcoin_v2_*.pkl`         | Prophet (Facebook)      | Predicción temporal de criptomonedas            | ✅ Activo |
| `catboost_avocado_*.pkl`            | CatBoost               | Predicción de precios de commodities agrícolas   | ✅ Activo |
| `knn_movie_recommendation_*.pkl`    | K-Nearest Neighbors    | Sistema de recomendación por similitud          | ✅ Activo |
| `random_forest_flights_*.pkl`       | Random Forest          | Predicción de retrasos en transporte aéreo      | ✅ Activo |
| `decision_tree_acv_*.pkl`           | Decision Tree          | Evaluación de riesgo médico                      | ✅ Activo |
| `bitcoin_random_forest_*.pkl`       | Random Forest          | Predicción de criptomonedas (versión anterior)  | 📦 Legacy |

### Tecnologías de ML Implementadas
- **Series Temporales**: Prophet para análisis de tendencias y estacionalidad
- **Gradient Boosting**: CatBoost para manejo de features categóricas
- **Ensemble Methods**: Random Forest para robustez predictiva
- **Sistemas de Recomendación**: KNN con métricas de similitud personalizadas
- **Árboles de Decisión**: Interpretabilidad para dominio médico

### Modelos de Deep Learning (Nuevos)
Se han añadido varios modelos basados en Deep Learning para ampliar las capacidades del sistema, especialmente en reconocimiento facial, detección de emociones y transcripción de voz.

| Modelo / Archivo                         | Framework / Formato | Dominio de Aplicación                        | Estado    |
|:----------------------------------------:|:-------------------:|:--------------------------------------------:|:---------:|
| `dl_models/emotion_model.keras`          | Keras / TensorFlow  | Detección de emoción facial (imagen/video)   | ✅ Activo |
| `dl_models/asr/mariantonieta_asr_ctc.h5` | Keras (CTC)         | Reconocimiento automático de voz (ASR)       | ✅ Activo |
| Modelo de vehículos (YOLOv8/Ultralytics) | YOLO / PyTorch     | Detección y clasificación de vehículos       | ✅ Activo |

Estas incorporaciones permiten:
- Clasificación de emociones faciales en tiempo real mediante un modelo Keras.
- Transcripción de audio a texto con un modelo CTC (ASR) para integrarlo con el coordinador LLM.
- Detección y clasificación automática de vehículos en imágenes usando YOLOv8/Ultralytics.

Nota: Algunos modelos (ONNX) requieren `onnxruntime` para ejecución eficiente. Ver sección de instalación para más detalles.

## Stack Tecnológico

### Backend & APIs
- **Python 3.9+**: Lenguaje principal del sistema
- **FastAPI**: Framework moderno para APIs REST con validación automática
- **Uvicorn**: Servidor ASGI de alto rendimiento
- **Pydantic**: Validación de datos y serialización tipo-segura

### Machine Learning
- **Prophet**: Análisis de series temporales con componentes estacionales
- **CatBoost**: Gradient boosting con manejo nativo de features categóricas  
- **Scikit-learn**: Biblioteca estándar para algoritmos ML clásicos
- **Joblib**: Persistencia optimizada de modelos ML
- **Pandas & NumPy**: Manipulación y procesamiento de datos

- **TensorFlow & Keras**: Entrenamiento e inferencia de modelos deep learning (p.ej. clasificación de emociones, ASR CTC)
- **Ultralytics (YOLO)**: Detección de objetos en imágenes y video (inferencia rápida)
- **Tools para ASR**: Librerías y utilidades para evaluación de ASR (e.g., `jiwer` para WER)

### LLM & Procesamiento de Lenguaje
- **Ollama**: Runtime local para modelos de lenguaje
- **LLaMA**: Arquitectura de transformer para comprensión del lenguaje
- **Extracción de entidades**: Parsing inteligente de parámetros temporales y numéricos

### Monitoreo & Logging
- **Sistema de logging centralizado**: Configuración unificada con rotación automática
- **Métricas de rendimiento**: Tracking de timing y throughput por endpoint
- **Health checks**: Monitoreo automático del estado de modelos
- **Error handling**: Manejo robusto de excepciones con fallbacks

### Frontend & UI
- **React**: Framework de interfaz de usuario moderna
- **TypeScript**: Tipado estático para JavaScript
- **Vite**: Build tool optimizado para desarrollo
- **Node.js + npm**: Runtime y gestión de dependencias

## Instalación y Configuración

### Requisitos del Sistema
- **Python 3.9+** (recomendado 3.11+)
- **Node.js 16+** y npm
- **Git** para control de versiones
- **4GB+ RAM** para modelos ML
- **macOS/Linux**: Homebrew para dependencias adicionales

### Configuración del Entorno Python

```bash
# Clonar repositorio
git clone https://github.com/SMatey/Mariantonieta-IA.git
cd Mariantonieta-IA

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate   # macOS/Linux (zsh compatible)

# Instalar dependencias
pip install -r requirements.txt
```

### Variables de Entorno

Configura estas variables para personalizar el comportamiento del sistema:

```bash
# Archivo .env (no incluir en Git)
ENV=development
AZURE_FACE_KEY="your_api_key"
AZURE_FACE_ENDPOINT="https://<endpoint>.cognitiveservices.azure.com/"
LLM_HOST="http://localhost:11434"
API_BASE_URL="http://localhost:8000"
LOG_LEVEL="INFO"
MAX_LOG_SIZE_MB=5
LOG_BACKUP_COUNT=5
```

### Configuración del LLM Local (Ollama)

```bash
# macOS (Homebrew)
brew install ollama

# Iniciar servicio
ollama serve

# Descargar modelo recomendado
ollama pull llama3.1:8b
```

## Ejecución del Sistema

### Método Unificado (Recomendado)

```bash
# Ejecutar todos los servicios
chmod +x run_all.sh
./run_all.sh
```

### Servicios Individuales

#### 1. API Backend (Puerto 8000)
```bash
source venv/bin/activate
./run_api.sh  # Script optimizado con uvicorn
```

### 2) Coordinador LLM

```bash
source venv/bin/activate
python llm/coordinator.py
```

#### 3. Frontend (Puerto 5173)
```bash
cd frontend
npm install
npm run dev
```

### Verificación del Sistema

```bash
# Health checks de las APIs
curl http://localhost:8000/health      # API principal
curl http://localhost:8000/models/bitcoin/health
curl http://localhost:8000/models/avocado/health

# Estado del LLM
curl http://localhost:11434/api/tags   # Ollama models
```

## Capacidades del Sistema

### Procesamiento de Lenguaje Natural
- **Extracción temporal**: Reconocimiento automático de fechas relativas y absolutas
- **Parsing de parámetros**: Identificación inteligente de valores numéricos y categorías
- **Contextualización**: Interpretación de consultas ambiguas con context-awareness
- **Respuestas explicativas**: Generación de interpretaciones detalladas de resultados

### APIs de Predicción Disponibles
- **Análisis temporal**: Predicciones de series de tiempo con intervalos de confianza
- **Commodities agrícolas**: Predicción de precios con features de mercado
- **Sistemas de recomendación**: Filtrado colaborativo y por contenido
- **Análisis de riesgo**: Evaluación probabilística en dominios médicos
- **Transporte aéreo**: Predicción de retrasos con factores meteorológicos

### Arquitectura de Microservicios
- **Escalabilidad horizontal**: Cada modelo puede escalarse independientemente
- **Tolerancia a fallos**: Fallbacks automáticos y circuit breakers
- **Load balancing**: Distribución de carga entre instancias
- **Versionado**: Soporte para múltiples versiones de modelos simultáneamente

## Arquitectura del Sistema

```
proyecto/
├── api/                           # 🔥 Microservicios REST
│   ├── main.py                   # Coordinador principal de APIs
│   ├── config_logger.py          # Sistema de logging centralizado
│   ├── constants.py              # Configuración centralizada
│   ├── routes/                   # Endpoints por dominio
│   │   ├── bitcoin_api.py        # API de criptomonedas
│   │   ├── avocado_api.py        # API de commodities agrícolas
│   │   ├── movies_api.py         # API de recomendaciones
│   │   ├── flights_api.py        # API de transporte aéreo
│   │   └── acv_api.py            # API médica de riesgo
│   └── models/                   # Esquemas Pydantic
├── llm/                          # 🧠 Coordinador LLM
│   ├── coordinator.py            # Orquestador inteligente
│   └── extract_params.py        # Extracción de parámetros NLP
├── ml_models/                    # 🤖 Modelos ML serializados
│   ├── prophet_bitcoin_v2_*.pkl  # Series temporales
│   ├── catboost_avocado_*.pkl    # Gradient boosting
│   └── *.pkl                     # Otros modelos entrenados
├── logs/                         # 📋 Sistema de logging
│   ├── main_api.log              # Log del coordinador
│   ├── *_api.log                 # Logs por microservicio
│   └── README.md                 # Documentación de logs
├── frontend/                     # 💻 Interfaz React
│   ├── src/components/           # Componentes UI
│   └── package.json              # Dependencias frontend
├── data/                         # 📊 Datasets y ejemplos
├── notebooks/                    # 📚 Análisis exploratorio
└── tests/                        # 🧪 Suite de pruebas
```

### Flujo de Datos
1. **Usuario** → Frontend React
2. **Frontend** → Coordinador LLM (puerto 8001)
3. **Coordinador** → Extracción de parámetros NLP
4. **Coordinador** → API específica (puerto 8000)
5. **API** → Modelo ML + Logging
6. **Respuesta** → Usuario con interpretación

## Monitoreo y Debugging

### Sistema de Logging
- **Logging centralizado**: Configuración unificada en `api/config_logger.py`
- **Rotación automática**: Archivos de máximo 5MB con 5 backups
- **Niveles configurables**: INFO, ERROR, WARNING, DEBUG
- **Sin output en consola**: Logs exclusivamente en archivos para interfaces limpias

### Métricas de Rendimiento
```bash
# Monitoreo en tiempo real
tail -f logs/main_api.log logs/bitcoin_api.log

# Análisis de errores
grep "ERROR" logs/*.log

# Estadísticas de predicciones
grep "Prediction" logs/*_api.log | wc -l
```

### Health Checks Automatizados
- **Estado de modelos**: Verificación de carga exitosa
- **Conectividad LLM**: Pruebas de comunicación con Ollama
- **Métricas de memoria**: Uso de recursos por modelo
- **Endpoints de diagnóstico**: `/health` en cada microservicio

## Troubleshooting

### Problemas Comunes

#### Errores de Dependencias
```bash
# ModuleNotFoundError
source venv/bin/activate
pip install -r requirements.txt --upgrade

# Verificar instalación
python -c "import fastapi, prophet, catboost; print('✅ Dependencies OK')"
```

#### Problemas con Ollama
```bash
# Verificar estado del servicio
ollama ps
ollama list

# Logs de Ollama
tail -f ~/.ollama/logs/server.log

# Reiniciar servicio
pkill ollama && ollama serve
```

#### Issues del Frontend
```bash
# Limpiar cache y reinstalar
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### Configuración de Desarrollo

#### Variables de Debug
```bash
export LOG_LEVEL="DEBUG"
export FASTAPI_DEBUG="true"
export LLM_TIMEOUT="30"
```

#### Pruebas de Conectividad
```bash
# Test API principal
curl -X GET http://localhost:8000/health | jq

# Test coordinador LLM
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"query": "¿Cuál es el precio de Bitcoin?"}'
```


## Información del Proyecto

### Equipo de Desarrollo
- **Sebastian Matey** 
- **Liz Salazar** 
- **Roosevelt Pérez** 

### Institución
**Instituto Tecnológico de Costa Rica (TEC)**  
Sede San Carlos - Escuela de Ingeniería en Computación

### Tecnologías y Licencias
- **Repositorio**: Mariantonieta-IA (GitHub)
- **Licencia**: MIT License
- **Stack principal**: Python 3.11, FastAPI, React 18, TypeScript
- **ML Stack**: Prophet, CatBoost, Scikit-learn

**Última actualización**: 1 de diciembre de 2025  




