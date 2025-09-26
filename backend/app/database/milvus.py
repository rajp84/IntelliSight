from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusException,
    connections,
    utility,
)


def connect(alias: str = "default", host: Optional[str] = None, port: Optional[str] = None) -> None:
    """
    Establish a connection to Milvus.

    Uses MILVUS_HOST and MILVUS_PORT env vars if not provided, defaulting to localhost:19530.
    """
    host = host or os.getenv("MILVUS_HOST", "127.0.0.1")
    port = port or os.getenv("MILVUS_PORT", "19530")
    if not connections.has_connection(alias):
        connections.connect(alias=alias, host=host, port=port)


def is_connected(alias: str = "default") -> bool:
    return connections.has_connection(alias)


def disconnect(alias: str = "default") -> None:
    if connections.has_connection(alias):
        connections.disconnect(alias)


def list_collections() -> List[str]:
    return utility.list_collections()


def drop_collection(collection_name: str) -> None:
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)


def create_collection(
    collection_name: str,
    dim: int,
    index_params: Optional[Dict[str, Any]] = None,
    description: str = "",
    auto_id: bool = True,
    metric_type: str = "L2",
    index_type: str = "IVF_FLAT",
    shards_num: int = 2,
) -> Collection:
    """
    Create a simple collection with an auto-increment primary key `id` (int64),
    a vector field `embedding` with the provided dimension, and an optional
    string `payload` field for arbitrary metadata.
    """
    if utility.has_collection(collection_name):
        return Collection(collection_name)

    id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=auto_id)
    vector_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    payload_field = FieldSchema(name="payload", dtype=DataType.VARCHAR, max_length=2048)

    schema = CollectionSchema(
        fields=[id_field, vector_field, payload_field],
        description=description,
    )

    collection = Collection(name=collection_name, schema=schema, shards_num=shards_num)

    # Create index on the vector field for efficient search
    index_params = index_params or {
        "index_type": index_type,
        "metric_type": metric_type,
        "params": {"nlist": 1024},
    }
    collection.create_index(field_name="embedding", index_params=index_params)

    # Load the collection into memory for search/insert
    collection.load()
    return collection


def insert_entries(
    collection_name: str,
    embeddings: Sequence[Sequence[float]],
    payloads: Optional[Sequence[str]] = None,
) -> List[int]:
    """
    Insert entries into a collection.

    If the primary key is auto_id=True, omit ids and Milvus assigns them.
    Returns the list of assigned ids.
    """
    if not connections.has_connection("default"):
        raise RuntimeError("Milvus connection not initialized. Ensure connection_manager.init_connections() has run.")
    collection = Collection(collection_name)
    num = len(embeddings)
    if payloads is None:
        payloads = [""] * num
    if len(payloads) != num:
        raise ValueError("payloads length must match embeddings length")

    # Data order must match fields in schema excluding auto-id
    mr = collection.insert([embeddings, payloads])
    collection.flush()
    return list(mr.primary_keys)


def upsert_entries(
    collection_name: str,
    ids: Sequence[int],
    embeddings: Sequence[Sequence[float]],
    payloads: Optional[Sequence[str]] = None,
) -> List[int]:
    """
    Upsert entries with explicit ids (requires auto_id=False in schema).
    """
    if not connections.has_connection("default"):
        raise RuntimeError("Milvus connection not initialized. Ensure connection_manager.init_connections() has run.")
    collection = Collection(collection_name)
    num = len(embeddings)
    if len(ids) != num:
        raise ValueError("ids length must match embeddings length")
    if payloads is None:
        payloads = [""] * num
    if len(payloads) != num:
        raise ValueError("payloads length must match embeddings length")

    mr = collection.upsert([list(ids), embeddings, payloads])
    collection.flush()
    return list(mr.primary_keys)


def delete_by_ids(collection_name: str, ids: Sequence[int]) -> int:
    if not connections.has_connection("default"):
        raise RuntimeError("Milvus connection not initialized. Ensure connection_manager.init_connections() has run.")
    collection = Collection(collection_name)
    expr = f"id in {list(ids)}"
    res = collection.delete(expr)
    collection.flush()
    return res.delete_count


def query_by_ids(collection_name: str, ids: Sequence[int], output_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    if not connections.has_connection("default"):
        raise RuntimeError("Milvus connection not initialized. Ensure connection_manager.init_connections() has run.")
    collection = Collection(collection_name)
    expr = f"id in {list(ids)}"
    return collection.query(expr=expr, output_fields=output_fields or ["id", "payload"])


def search_embeddings(
    collection_name: str,
    query_embeddings: Sequence[Sequence[float]],
    top_k: int = 5,
    metric_type: str = "L2",
    nprobe: int = 16,
    output_fields: Optional[List[str]] = None,
):
    if not connections.has_connection("default"):
        raise RuntimeError("Milvus connection not initialized. Ensure connection_manager.init_connections() has run.")
    collection = Collection(collection_name)
    search_params = {"metric_type": metric_type, "params": {"nprobe": nprobe}}
    return collection.search(
        data=query_embeddings,
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=output_fields or ["id", "payload"],
    )


__all__ = [
    "connect",
    "disconnect",
    "list_collections",
    "create_collection",
    "drop_collection",
    "insert_entries",
    "upsert_entries",
    "delete_by_ids",
    "query_by_ids",
    "search_embeddings",
]


