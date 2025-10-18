from fastapi import APIRouter, HTTPException
from api.responses import query_models, health_models

from llm.coordinator import interpretar_y_ejecutar


router = APIRouter(tags=["Coordinator"])


@router.post("/ask", response_model=query_models.QueryResponse)
def ask_user(request: query_models.QueryRequest):
    """Usa el LLM coordinator para procesar la consulta."""
    try:
        respuesta = interpretar_y_ejecutar(request.query)
        return query_models.QueryResponse(respuesta=respuesta)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en coordinador: {str(e)}")