# storage/cosmos_store.py
from __future__ import annotations
import os
from functools import lru_cache
from typing import List, Dict, Any

from azure.cosmos import CosmosClient, PartitionKey
from azure.cosmos.exceptions import CosmosHttpResponseError

# --- ENV obligatorias ---
def _require_env():
    needed = [
        "AZURE_COSMOS_ENDPOINT",
        "AZURE_COSMOS_KEY",
        "AZURE_COSMOS_DB",
        "AZURE_COSMOS_CONTAINER",
    ]
    missing = [k for k in needed if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Cosmos no configurado. Faltan: {', '.join(missing)}")

# --- Políticas vectoriales e indexación (dicts, no clases) ---
def _vector_embedding_policy() -> Dict[str, Any]:
    dim = int(os.getenv("FACE_EMBED_DIM", "128"))         # SFace => 128
    distance = os.getenv("COSMOS_VECTOR_DISTANCE", "cosine").lower()
    return {
        "vectorEmbeddings": [{
            "path": "/embedding",
            "dataType": "float32",
            "dimensions": dim,
            "distanceFunction": distance
        }]
    }

def _indexing_policy() -> Dict[str, Any]:
    return {
        "automatic": True,
        "indexingMode": "consistent",
        "includedPaths": [{"path": "/*"}],
        "excludedPaths": [
            {"path": "/\"_etag\"/?"},
            {"path": "/embedding/*"}  # evita indexación “normal” del vector
        ],
        "vectorIndexes": [
            {"path": "/embedding", "type": "quantizedFlat"}  # "flat" o "diskANN" también válido
        ],
    }

@lru_cache(maxsize=1)
def _get_client_tuple():
    _require_env()
    endpoint  = os.getenv("AZURE_COSMOS_ENDPOINT")
    key       = os.getenv("AZURE_COSMOS_KEY")
    db_name   = os.getenv("AZURE_COSMOS_DB")
    coll_name = os.getenv("AZURE_COSMOS_CONTAINER")

    client = CosmosClient(endpoint, credential=key)
    db = client.create_database_if_not_exists(id=db_name)

    try:
        container = db.create_container_if_not_exists(
            id=coll_name,
            partition_key=PartitionKey(path="/userId"),
            indexing_policy=_indexing_policy(),
            vector_embedding_policy=_vector_embedding_policy(),
        )
    except TypeError as te:
        raise RuntimeError(
            "Fallo creando el contenedor con vector search. Revisa que estés pasando "
            "`indexing_policy` y `vector_embedding_policy` (como dict) exactamente con esos nombres."
        ) from te
    except CosmosHttpResponseError as ce:
        # 409 existe; 400/BadRequest si la cuenta no tiene habilitado vector search
        raise

    # Verifica que realmente tenga la policy de vectores
    props = container.read()
    if "vectorEmbeddingPolicy" not in props:
        raise RuntimeError(
            f"El contenedor '{coll_name}' existe pero SIN vector search. "
            "Bórralo o usa otro nombre en AZURE_COSMOS_CONTAINER para crearlo con vector search."
        )

    return client, db, container

def upsert_face_vector(user_id: str, blob_path: str, embedding: List[float]) -> str:
    _, _, container = _get_client_tuple()
    # id único por usuario + blob
    from os.path import basename
    doc_id = f"{user_id}:{basename(blob_path)}"
    doc = {
        "id": doc_id,
        "userId": user_id,
        "blobPath": blob_path,
        "embedding": embedding,  # list[float] (idealmente float32)
    }
    container.upsert_item(doc, partition_key=user_id)
    return doc_id

def topk_by_vector(vector: List[float], k: int = 5, user_id: str | None = None) -> List[Dict[str, Any]]:
    """
    k-NN en Cosmos.
    - Si user_id se pasa: busca dentro de esa partición (recomendado).
    - Intenta 2 variantes de ORDER BY:
        A) ORDER BY VectorDistance(...)
        B) ORDER BY distance (alias en la misma SELECT)
    - Si fallan, hace fallback a rankeo local.
    """
    _, _, container = _get_client_tuple()
    dim = int(os.getenv("FACE_EMBED_DIM", "128"))

    # --- A) Consulta por partición (si hay user_id) con ORDER BY VectorDistance(...) ---
    if user_id:
        query_a = """
        SELECT TOP @k
            c.id, c.userId, c.blobPath,
            VectorDistance(c.embedding, @q) AS distance
        FROM c
        WHERE c.userId = @userId
          AND IS_DEFINED(c.embedding)
          AND IS_ARRAY(c.embedding)
          AND ARRAY_LENGTH(c.embedding) = @dim
        ORDER BY VectorDistance(c.embedding, @q)
        """
        params_a = [
            {"name": "@k", "value": int(k)},
            {"name": "@q", "value": [float(x) for x in vector]},  # asegurar float nativo
            {"name": "@userId", "value": user_id},
            {"name": "@dim", "value": dim},
        ]
        try:
            return list(container.query_items(
                query=query_a,
                parameters=params_a,
                partition_key=user_id,
                enable_cross_partition_query=False
            ))
        except Exception:
            # B) Mismo filtro, pero ordenando por el alias 'distance' desde una subselect
            query_b = """
            SELECT TOP @k r.id, r.userId, r.blobPath, r.distance
            FROM (
                SELECT
                    c.id, c.userId, c.blobPath,
                    VectorDistance(c.embedding, @q) AS distance
                FROM c
                WHERE c.userId = @userId
                  AND IS_DEFINED(c.embedding)
                  AND IS_ARRAY(c.embedding)
                  AND ARRAY_LENGTH(c.embedding) = @dim
            ) AS r
            ORDER BY r.distance
            """
            try:
                return list(container.query_items(
                    query=query_b,
                    parameters=params_a,
                    partition_key=user_id,
                    enable_cross_partition_query=False
                ))
            except Exception:
                pass  # caerá al fallback local

    # --- (Opcional) Global (todas las particiones): puede fallar con 2206. Probamos A y B. ---
    query_a_glob = """
    SELECT TOP @k
        c.id, c.userId, c.blobPath,
        VectorDistance(c.embedding, @q) AS distance
    FROM c
    WHERE IS_DEFINED(c.embedding)
      AND IS_ARRAY(c.embedding)
      AND ARRAY_LENGTH(c.embedding) = @dim
    ORDER BY VectorDistance(c.embedding, @q)
    """
    params_glob = [
        {"name": "@k", "value": int(k)},
        {"name": "@q", "value": [float(x) for x in vector]},
        {"name": "@dim", "value": dim},
    ]
    try:
        return list(container.query_items(
            query=query_a_glob,
            parameters=params_glob,
            enable_cross_partition_query=True
        ))
    except Exception:
        # B) Variante con alias en subselect
        query_b_glob = """
        SELECT TOP @k r.id, r.userId, r.blobPath, r.distance
        FROM (
            SELECT
                c.id, c.userId, c.blobPath,
                VectorDistance(c.embedding, @q) AS distance
            FROM c
            WHERE IS_DEFINED(c.embedding)
              AND IS_ARRAY(c.embedding)
              AND ARRAY_LENGTH(c.embedding) = @dim
        ) AS r
        ORDER BY r.distance
        """
        try:
            return list(container.query_items(
                query=query_b_glob,
                parameters=params_glob,
                enable_cross_partition_query=True
            ))
        except Exception:
            pass  # caerá al fallback local

    # --- Fallback: traer muestra y rankear localmente (cosine) ---
    q = """
    SELECT TOP @n c.id, c.userId, c.blobPath, c.embedding
    FROM c
    WHERE IS_DEFINED(c.embedding)
      AND IS_ARRAY(c.embedding)
      AND ARRAY_LENGTH(c.embedding) = @dim
    """
    rows = list(container.query_items(
        query=q,
        parameters=[{"name": "@n", "value": max(200, k)}],
        enable_cross_partition_query=True
    ))
    import numpy as np
    v = np.asarray([float(x) for x in vector], dtype="float32")
    v = v / (np.linalg.norm(v) + 1e-12)

    def score(e):
        u = np.asarray(e["embedding"], dtype="float32")
        u = u / (np.linalg.norm(u) + 1e-12)
        return float(1.0 - (u @ v))  # cosine distance

    rows.sort(key=score)
    return [
        {
            "id": r["id"],
            "userId": r["userId"],
            "blobPath": r.get("blobPath"),
            "distance": score(r)
        }
        for r in rows[:k]
    ]
