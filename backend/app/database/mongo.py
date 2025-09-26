from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from pymongo import ASCENDING, DESCENDING, MongoClient, errors


_clients: Dict[str, MongoClient] = {}
_default_db_names: Dict[str, Optional[str]] = {}


def is_connected(alias: str = "default") -> bool:
    return alias in _clients and _clients[alias] is not None


def connect(
    alias: str = "default",
    uri: Optional[str] = None,
    *,
    host: Optional[str] = None,
    port: Optional[int] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    auth_source: Optional[str] = None,
    tls: Optional[bool] = None,
    db_name: Optional[str] = None,
) -> MongoClient:
    """
    Establish a connection to MongoDB.

    Environment fallbacks when arguments are not provided:
    - MONGO_URI
    - MONGO_HOST (default 127.0.0.1), MONGO_PORT (default 27017)
    - MONGO_USERNAME, MONGO_PASSWORD, MONGO_AUTH_SOURCE
    - MONGO_TLS ("true"/"false")
    - MONGO_DB (default database name for helpers)
    """
    if alias in _clients and _clients[alias] is not None:
        return _clients[alias]

    if uri is None:
        host = host or os.getenv("MONGO_HOST", "127.0.0.1")
        port = int(port or int(os.getenv("MONGO_PORT", "27017")))
        # Defaults for secured local dev instance
        username = username or os.getenv("MONGO_USERNAME", "root")
        password = password or os.getenv("MONGO_PASSWORD", "password")
        auth_source = auth_source or os.getenv("MONGO_AUTH_SOURCE", "admin")
        tls_env = os.getenv("MONGO_TLS")
        tls = tls if tls is not None else (tls_env.lower() == "true" if tls_env else None)

        # Build URI manually only when credentials provided; otherwise use host/port form
        if username and password:
            uri = f"mongodb://{username}:{password}@{host}:{port}"
            query: List[str] = []
            if auth_source:
                query.append(f"authSource={auth_source}")
            if tls is True:
                query.append("tls=true")
            if query:
                uri += "/?" + "&".join(query)
        else:
            # No credentials
            uri = f"mongodb://{host}:{port}"

    client = MongoClient(uri)
    _clients[alias] = client
    # Default DB fallback if env not set
    _default_db_names[alias] = db_name or os.getenv("MONGO_DB") or "intellisight"
    return client


def disconnect(alias: str = "default") -> None:
    client = _clients.get(alias)
    if client is not None:
        client.close()
        _clients.pop(alias, None)
        _default_db_names.pop(alias, None)


def get_client(alias: str = "default") -> MongoClient:
    client = _clients.get(alias)
    if client is None:
        raise RuntimeError("MongoDB client not initialized. Ensure connection_manager.init_connections() has run.")
    return client


def get_db(db_name: Optional[str] = None, *, alias: str = "default"):
    client = get_client(alias)
    # Use stored default, env, or a sensible fallback
    final_db = db_name or _default_db_names.get(alias) or os.getenv("MONGO_DB") or "intellisight"
    return client[final_db]


def list_collections(db_name: Optional[str] = None, *, alias: str = "default") -> List[str]:
    db = get_db(db_name, alias=alias)
    return db.list_collection_names()


def create_collection(
    name: str,
    *,
    db_name: Optional[str] = None,
    alias: str = "default",
    options: Optional[Dict[str, Any]] = None,
):
    db = get_db(db_name, alias=alias)
    if name in db.list_collection_names():
        return db.get_collection(name)
    options = options or {}
    return db.create_collection(name, **options)


def drop_collection(name: str, *, db_name: Optional[str] = None, alias: str = "default") -> None:
    db = get_db(db_name, alias=alias)
    if name in db.list_collection_names():
        db.drop_collection(name)


def get_collection(name: str, *, db_name: Optional[str] = None, alias: str = "default"):
    db = get_db(db_name, alias=alias)
    return db.get_collection(name)


def insert_one(
    collection_name: str,
    document: Dict[str, Any],
    *,
    db_name: Optional[str] = None,
    alias: str = "default",
) -> Any:
    coll = get_collection(collection_name, db_name=db_name, alias=alias)
    result = coll.insert_one(document)
    return result.inserted_id


def insert_many(
    collection_name: str,
    documents: Sequence[Dict[str, Any]],
    *,
    db_name: Optional[str] = None,
    alias: str = "default",
) -> List[Any]:
    coll = get_collection(collection_name, db_name=db_name, alias=alias)
    result = coll.insert_many(list(documents))
    return list(result.inserted_ids)


def find_one(
    collection_name: str,
    filter: Optional[Dict[str, Any]] = None,
    *,
    db_name: Optional[str] = None,
    alias: str = "default",
    projection: Optional[Dict[str, int]] = None,
    sort: Optional[List[Tuple[str, int]]] = None,
) -> Optional[Dict[str, Any]]:
    coll = get_collection(collection_name, db_name=db_name, alias=alias)
    if sort:
        return coll.find_one(filter or {}, projection=projection, sort=sort)
    return coll.find_one(filter or {}, projection=projection)


def find_many(
    collection_name: str,
    filter: Optional[Dict[str, Any]] = None,
    *,
    db_name: Optional[str] = None,
    alias: str = "default",
    projection: Optional[Dict[str, int]] = None,
    sort: Optional[List[Tuple[str, int]]] = None,
    limit: Optional[int] = None,
    skip: Optional[int] = None,
) -> List[Dict[str, Any]]:
    coll = get_collection(collection_name, db_name=db_name, alias=alias)
    cursor = coll.find(filter or {}, projection=projection)
    if sort:
        cursor = cursor.sort(sort)
    if skip:
        cursor = cursor.skip(skip)
    if limit:
        cursor = cursor.limit(limit)
    return list(cursor)


def update_one(
    collection_name: str,
    filter: Dict[str, Any],
    update: Dict[str, Any],
    *,
    db_name: Optional[str] = None,
    alias: str = "default",
    upsert: bool = False,
) -> Dict[str, Any]:
    coll = get_collection(collection_name, db_name=db_name, alias=alias)
    res = coll.update_one(filter, update, upsert=upsert)
    return {"matched": res.matched_count, "modified": res.modified_count, "upserted_id": res.upserted_id}


def update_many(
    collection_name: str,
    filter: Dict[str, Any],
    update: Dict[str, Any],
    *,
    db_name: Optional[str] = None,
    alias: str = "default",
    upsert: bool = False,
) -> Dict[str, Any]:
    coll = get_collection(collection_name, db_name=db_name, alias=alias)
    res = coll.update_many(filter, update, upsert=upsert)
    return {"matched": res.matched_count, "modified": res.modified_count, "upserted_id": res.upserted_id}


def delete_one(
    collection_name: str,
    filter: Dict[str, Any],
    *,
    db_name: Optional[str] = None,
    alias: str = "default",
) -> int:
    coll = get_collection(collection_name, db_name=db_name, alias=alias)
    res = coll.delete_one(filter)
    return res.deleted_count


def delete_many(
    collection_name: str,
    filter: Dict[str, Any],
    *,
    db_name: Optional[str] = None,
    alias: str = "default",
) -> int:
    coll = get_collection(collection_name, db_name=db_name, alias=alias)
    res = coll.delete_many(filter)
    return res.deleted_count


def create_index(
    collection_name: str,
    keys: Sequence[Tuple[str, Union[int, str]]],
    *,
    db_name: Optional[str] = None,
    alias: str = "default",
    unique: bool = False,
    sparse: bool = False,
    ttl_seconds: Optional[int] = None,
) -> str:
    coll = get_collection(collection_name, db_name=db_name, alias=alias)
    normalized: List[Tuple[str, int]] = []
    for field, direction in keys:
        if isinstance(direction, str):
            dir_val = ASCENDING if direction.lower() in ("asc", "ascending") else DESCENDING
        else:
            dir_val = int(direction)
        normalized.append((field, dir_val))
    options: Dict[str, Any] = {"unique": unique, "sparse": sparse}
    if ttl_seconds is not None:
        options["expireAfterSeconds"] = int(ttl_seconds)
    return coll.create_index(normalized, **options)


__all__ = [
    "connect",
    "disconnect",
    "is_connected",
    "get_client",
    "get_db",
    "list_collections",
    "create_collection",
    "drop_collection",
    "get_collection",
    "insert_one",
    "insert_many",
    "find_one",
    "find_many",
    "update_one",
    "update_many",
    "delete_one",
    "delete_many",
    "create_index",
]


