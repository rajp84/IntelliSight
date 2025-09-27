from __future__ import annotations

import time
from typing import Any, Dict, Optional

from .mongo import (
    create_collection,
    create_index,
    find_one,
    find_many,
    insert_one,
    update_one,
    delete_one,
)


COLLECTION_NAME = "training"


def _ensure_collection() -> None:
    # Idempotent create and indexes helpful for querying history
    create_collection(COLLECTION_NAME)
    try:
        # Recent first by started time; filter by status commonly
        create_index(COLLECTION_NAME, [("started_at", -1)])
        create_index(COLLECTION_NAME, [("status", 1), ("started_at", -1)])
    except Exception:
        # Index creation best-effort only
        pass


def create_training_record(
    *,
    file_path: str,
    file_name: str,
    fps: Optional[float],
    width: int,
    height: int,
    total_frames: int,
    training_params: Dict[str, Any],
    status: str = "running",
) -> Any:
    """Create a new training record and return its inserted id."""
    _ensure_collection()
    started_at = time.time()
    doc: Dict[str, Any] = {
        "status": status,
        "file_path": file_path,
        "file_name": file_name,
        "fps": float(fps) if fps is not None else None,
        "width": int(width),
        "height": int(height),
        "total_frames": int(total_frames),
        "frames_processed": 0,
        "started_at": started_at,
        "completed_at": None,
        "elapsed_s": 0.0,
        "training_params": dict(training_params or {}),
    }
    return insert_one(COLLECTION_NAME, doc)


def update_training_record(record_id: Any, *, set_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Generic update helper for a training record ($set)."""
    return update_one(COLLECTION_NAME, {"_id": record_id}, {"$set": dict(set_fields or {})}, upsert=False)


def mark_training_status(
    record_id: Any,
    status: str,
    *,
    frames_processed: Optional[int] = None,
    fps_effective: Optional[float] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience helper to finalize/update status with computed elapsed time."""
    now = time.time()
    started = None
    try:
        doc = find_one(COLLECTION_NAME, {"_id": record_id}, projection={"started_at": 1})
        if doc:
            started = doc.get("started_at")
    except Exception:
        started = None
    elapsed = (now - float(started)) if started is not None else None
    set_fields: Dict[str, Any] = {
        "status": status,
        "completed_at": now,
        "elapsed_s": elapsed if elapsed is not None else None,
    }
    if frames_processed is not None:
        set_fields["frames_processed"] = int(frames_processed)
    if fps_effective is not None:
        set_fields["fps_effective"] = float(fps_effective)
    if error is not None:
        set_fields["error"] = error
    # Remove None values to keep document clean
    set_fields = {k: v for k, v in set_fields.items() if v is not None}
    return update_training_record(record_id, set_fields=set_fields)


__all__ = [
    "create_training_record",
    "update_training_record",
    "mark_training_status",
    "delete_training_record",
]


def error_out_running_records(reason: Optional[str] = None) -> int:
    """Mark any records with status 'running' as 'error' before starting a new run.

    Returns the number of records updated.
    """
    _ensure_collection()
    updated = 0
    try:
        running_docs = find_many(COLLECTION_NAME, {"status": "running"}, projection={"_id": 1}, limit=None)
        for doc in running_docs:
            try:
                mark_training_status(doc.get("_id"), "error", error=reason or "Superseded by new training run")
                updated += 1
            except Exception:
                # best effort per-doc
                pass
    except Exception:
        return 0
    return int(updated)


def list_training_runs(page: int = 1, page_size: int = 20) -> Dict[str, Any]:
    """Return { total, items } for training runs, newest first.

    page is 1-based.
    """
    _ensure_collection()
    page = max(1, int(page))
    page_size = max(1, min(200, int(page_size)))
    skip = (page - 1) * page_size
    try:
        # Count total
        from .mongo import get_collection
        coll = get_collection(COLLECTION_NAME)
        total = coll.count_documents({})
        # Fetch items sorted by started_at desc
        items = coll.find({}, projection={
            "file_path": 1,
            "file_name": 1,
            "status": 1,
            "started_at": 1,
            "completed_at": 1,
            "frames_processed": 1,
            "total_frames": 1,
            "elapsed_s": 1,
        }).sort([("started_at", -1)]).skip(skip).limit(page_size)
        out = []
        for d in items:
            # Normalize ObjectId to string
            _id = str(d.get("_id")) if d.get("_id") is not None else None
            out.append({
                "_id": _id,
                "status": d.get("status"),
                "file_name": d.get("file_name"),
                "file_path": d.get("file_path"),
                "started_at": d.get("started_at"),
                "completed_at": d.get("completed_at"),
                "frames_processed": d.get("frames_processed"),
                "total_frames": d.get("total_frames"),
                "elapsed_s": d.get("elapsed_s"),
            })
        return {"total": int(total), "items": out}
    except Exception:
        return {"total": 0, "items": []}


def delete_training_record(record_id: Any) -> bool:
    """Delete a training record by id (string or ObjectId). Returns True if removed."""
    try:
        # Accept string id and convert
        try:
            from bson import ObjectId  # type: ignore
            rid = ObjectId(record_id) if not hasattr(record_id, "binary") else record_id
        except Exception:
            rid = record_id
        deleted = delete_one(COLLECTION_NAME, {"_id": rid})
        return bool(deleted and deleted > 0)
    except Exception:
        return False


