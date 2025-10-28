from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Iterable

import os
import mimetypes
import asyncio
import cv2
import json
import uuid
import io

from PIL import Image

from ..database.mongo import (
    create_collection,
    insert_one,
    find_many,
    find_one,
    delete_one,
    delete_many,
    get_collection,
)
from ..database.milvus import ensure_things_collection
from ..storage.minio_client import (
    put_negative_image_bytes,
    remove_negative_image,
    put_image_bytes as put_minio_image,
    ensure_things_bucket,
)
from ..service.embedder_service import embed_image
from ..service.system_service import get_configuration
from ..socket.socket_manager import broadcast


def _probe_media_metadata(media_path: Path) -> Tuple[int, int, int, float]:
    """Return (total_frames, frame_width, frame_height, fps)."""
    total_frames = 0
    frame_width = 0
    frame_height = 0
    fps_val = 0.0
    try:
        cap = cv2.VideoCapture(str(media_path))
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps_val = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            cap.release()
        else:
            with Image.open(str(media_path)) as im:
                frame_width, frame_height = im.size
                total_frames = 1
    except Exception:
        pass
    return total_frames, frame_width, frame_height, fps_val


def accept_job(job_id: str, media_path: Path, callback_url: str, job_type: str) -> Dict[str, Any]:
    create_collection("jobs")
    total_frames, frame_width, frame_height, fps_val = _probe_media_metadata(media_path)

    # Resolve model for detection jobs at acceptance time
    model_path_str: str | None = None
    try:
        if job_type == "detection":
            cfg = get_configuration()
            lib_path = str(cfg.get("library_path") or "/").strip()
            default_model = str(cfg.get("default_model") or "").strip()
            if default_model:
                mp = Path(lib_path) / "model" / default_model
                if mp.exists() and mp.is_file():
                    model_path_str = str(mp)
    except Exception:
        model_path_str = None

    job_doc: Dict[str, Any] = {
        "job_id": job_id,
        "media_file": str(media_path),
        "status": "accepted",
        "type": job_type,
        "conf": 0.6,
        "total_frames": total_frames,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "fps": fps_val,
        "callback_url": callback_url,
        "created_at": datetime.utcnow(),
        **({"model_path": model_path_str, "model_name": os.path.basename(model_path_str) if model_path_str else None} if model_path_str else {}),
    }
    insert_one("jobs", job_doc)

    # Fire-and-forget broadcast; do not block the request
    try:
        payload = dict(job_doc)
        ca = payload.get("created_at")
        if isinstance(ca, datetime):
            payload["created_at"] = ca.isoformat() + "Z"
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                broadcast(
                    "job_update",
                    {
                        "job_id": payload.get("job_id"),
                        "status": payload.get("status"),
                        "type": payload.get("type"),
                        "media_file": payload.get("media_file"),
                        "created_at": payload.get("created_at"),
                        "progress": 0.0,
                    },
                )
            )
        except RuntimeError:
            asyncio.run(
                broadcast(
                    "job_update",
                    {
                        "job_id": payload.get("job_id"),
                        "status": payload.get("status"),
                        "type": payload.get("type"),
                        "media_file": payload.get("media_file"),
                        "created_at": payload.get("created_at"),
                        "progress": 0.0,
                    },
                )
            )
    except Exception:
        pass

    # Return a copy without Mongo internal id
    out_doc = dict(job_doc)
    return out_doc


def list_jobs(limit: int) -> List[Dict[str, Any]]:
    create_collection("jobs")
    items = find_many("jobs", sort=[("created_at", -1)], limit=max(1, min(limit, 1000)))
    for it in items:
        it.pop("_id", None)
        ca = it.get("created_at")
        if isinstance(ca, datetime):
            it["created_at"] = ca.isoformat() + "Z"
    return items


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    create_collection("jobs")
    doc = find_one("jobs", {"job_id": job_id})
    if not doc:
        return None
    doc.pop("_id", None)
    ca = doc.get("created_at")
    if isinstance(ca, datetime):
        doc["created_at"] = ca.isoformat() + "Z"
    sa = doc.get("started_at")
    if isinstance(sa, datetime):
        doc["started_at"] = sa.isoformat() + "Z"
    ea = doc.get("completed_at")
    if isinstance(ea, datetime):
        doc["completed_at"] = ea.isoformat() + "Z"
    return doc


def get_job_frames(job_id: str, after_idx: int, blocks: int) -> Dict[str, Any]:
    coll = get_collection("frame_buckets")
    cur = (
        coll.find(
            {"job_id": job_id, "block_end_idx": {"$gt": int(after_idx)}},
            {"frames": 1, "block_start_idx": 1, "block_end_idx": 1, "imgw": 1, "imgh": 1, "_id": 0},
        )
        .sort([("block_start_idx", 1)])
        .limit(max(1, min(10, int(blocks))))
    )
    blocks_list = list(cur)
    out_frames: List[Dict[str, Any]] = []
    imgw = None
    imgh = None
    for b in blocks_list:
        imgw = b.get("imgw", imgw)
        imgh = b.get("imgh", imgh)
        for fr in b.get("frames", []) or []:
            if int(fr.get("i", -1)) <= int(after_idx):
                continue
            out_frames.append(fr)
            if len(out_frames) >= 500:
                break
        if len(out_frames) >= 500:
            break
    last_idx = out_frames[-1]["i"] if out_frames else after_idx
    return {"frames": out_frames, "last_idx": last_idx, "imgw": imgw, "imgh": imgh}


def _resolve_job_media_path(job_id: str) -> str:
    job = find_one("jobs", {"job_id": job_id})
    if not job:
        return ""
    media_path = str(job.get("media_file") or "").strip()
    return media_path


def open_media_stream(job_id: str, range_header: Optional[str]) -> Tuple[Iterable[bytes], Dict[str, str], int, str]:
    """
    Return (body_iter, headers, status_code, content_type) for the media stream.
    Parsing range happens here to keep routes focused on response construction.
    """
    media_path = _resolve_job_media_path(job_id)
    if not media_path or not os.path.isfile(media_path):
        raise FileNotFoundError("Media file not found")

    file_size = os.path.getsize(media_path)
    content_type, _ = mimetypes.guess_type(media_path)
    if media_path.lower().endswith(".mov"):
        content_type = "video/quicktime"
    content_type = content_type or "application/octet-stream"

    if range_header is None:
        def iter_file() -> Iterable[bytes]:
            with open(media_path, "rb") as f:
                while True:
                    data = f.read(1024 * 1024)
                    if not data:
                        break
                    yield data

        headers = {"Accept-Ranges": "bytes", "Content-Length": str(file_size)}
        return iter_file(), headers, 200, content_type

    # Parse Range header
    try:
        units, rng = range_header.split("=", 1)
        if units != "bytes":
            raise ValueError("Unsupported range unit")
        start_str, end_str = (rng.split("-", 1) + [""])[:2]
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else (file_size - 1)
        start = max(0, start)
        end = min(file_size - 1, end)
        if start > end:
            raise ValueError("Invalid range")
    except Exception:
        start, end = 0, file_size - 1

    chunk_size = (end - start) + 1

    def file_range(start_pos: int, end_pos: int) -> Iterable[bytes]:
        with open(media_path, "rb") as f:
            f.seek(start_pos)
            remaining = (end_pos - start_pos) + 1
            while remaining > 0:
                read_size = min(1024 * 1024, remaining)
                data = f.read(read_size)
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(chunk_size),
    }
    return file_range(start, end), headers, 206, content_type


def list_frame_blocks(job_id: str) -> Dict[str, Any]:
    coll = get_collection("frame_buckets")
    docs = list(
        coll.find(
            {"job_id": job_id}, {"_id": 0, "block_start_idx": 1, "block_end_idx": 1, "frames": 1}
        ).sort([("block_start_idx", 1)])
    )
    blocks: List[Dict[str, Any]] = []
    min_start: Optional[int] = None
    max_end: Optional[int] = None
    total_frames = 0
    for d in docs:
        bs = int(d.get("block_start_idx", -1))
        be = int(d.get("block_end_idx", -1))
        cnt = len(d.get("frames", []) or [])
        blocks.append({"start": bs, "end": be, "count": cnt})
        min_start = bs if min_start is None else min(min_start, bs)
        max_end = be if max_end is None else max(max_end, be)
        total_frames += cnt
    return {"blocks": blocks, "min_start": min_start, "max_end": max_end, "total_frames": total_frames}


def delete_job_and_frames(job_id: str) -> Dict[str, Any]:
    """Delete the job document and all corresponding frame_buckets for that job_id."""
    # Delete frame buckets
    try:
        fb = get_collection("frame_buckets")
        res_fb = fb.delete_many({"job_id": job_id})
        fb_deleted = int(getattr(res_fb, 'deleted_count', 0))
    except Exception:
        fb_deleted = 0
    # Delete job doc
    try:
        res_job = delete_one("jobs", {"job_id": job_id})
        job_deleted = bool(res_job)
    except Exception:
        job_deleted = False
    return {"deleted": job_deleted, "frame_buckets_deleted": fb_deleted}

def set_frame_invalid(job_id: str, frame_idx: int, invalid: bool = True) -> Dict[str, Any]:
    media_path = _resolve_job_media_path(job_id)
    if not media_path or not os.path.exists(media_path):
        raise FileNotFoundError("Media file not found")

    neg_object_name: Optional[str] = None
    if invalid:
        try:
            img_bytes: bytes = b""
            ct = "image/jpeg"
            cap = cv2.VideoCapture(media_path)
            if cap.isOpened():
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                    ok, fbgr = cap.read()
                    if ok:
                        ok2, buf = cv2.imencode(".jpg", fbgr)
                        if ok2:
                            img_bytes = buf.tobytes()
                finally:
                    cap.release()
            else:
                with open(media_path, "rb") as f:
                    img_bytes = f.read()
            if img_bytes:
                neg_object_name = f"{job_id}_{int(frame_idx)}.jpg"
                put_negative_image_bytes(neg_object_name, img_bytes, ct)
        except Exception:
            neg_object_name = None
    else:
        try:
            coll = get_collection("frame_buckets")
            doc = coll.find_one({"job_id": job_id, "frames.i": int(frame_idx)}, {"frames": 1})
            if doc:
                for fr in doc.get("frames", []) or []:
                    if int(fr.get("i", -1)) == int(frame_idx):
                        key = fr.get("negative_object")
                        if key:
                            try:
                                remove_negative_image(key)
                            except Exception:
                                pass
                        break
        except Exception:
            pass

    try:
        coll = get_collection("frame_buckets")
        block = coll.find_one({"job_id": job_id, "frames.i": int(frame_idx)})
        if block:
            frames = block.get("frames", []) or []
            updated_frames = []
            for fr in frames:
                if int(fr.get("i", -1)) == int(frame_idx):
                    fr = dict(fr)
                    if invalid:
                        fr["invalid"] = True
                        if neg_object_name:
                            fr["negative_object"] = neg_object_name
                    else:
                        fr.pop("invalid", None)
                        fr.pop("negative_object", None)
                updated_frames.append(fr)
            coll.update_one({"_id": block.get("_id")}, {"$set": {"frames": updated_frames}})
    except Exception:
        pass

    return {
        "job_id": job_id,
        "frame_idx": int(frame_idx),
        "invalid": bool(invalid),
        "negative_object": neg_object_name,
    }


def add_frame_crop_to_things(
    job_id: str,
    frame_idx: int,
    label: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> Dict[str, Any]:
    media_path = _resolve_job_media_path(job_id)
    if not media_path or not os.path.exists(media_path):
        raise FileNotFoundError("Media file not found")

    full_bytes: bytes = b""
    try:
        cap = cv2.VideoCapture(media_path)
        if cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ok, fbgr = cap.read()
                if not ok:
                    raise RuntimeError("Failed to read frame")
                ok2, buf = cv2.imencode(".jpg", fbgr)
                if not ok2:
                    raise RuntimeError("Failed to encode frame")
                full_bytes = buf.tobytes()
                pil = Image.fromarray(cv2.cvtColor(fbgr, cv2.COLOR_BGR2RGB))
            finally:
                cap.release()
        else:
            with open(media_path, "rb") as f:
                full_bytes = f.read()
            pil = Image.open(media_path).convert("RGB")
    except Exception as ex:
        raise RuntimeError(f"Failed to load frame: {ex}")

    try:
        _x1 = max(0, int(round(x1)))
        _y1 = max(0, int(round(y1)))
        _x2 = max(_x1 + 1, int(round(x2)))
        _y2 = max(_y1 + 1, int(round(y2)))
        crop = pil.crop((_x1, _y1, _x2, _y2))
    except Exception as ex:
        raise ValueError(f"Invalid crop: {ex}")

    emb = embed_image(crop)
    emb_list = emb.reshape(1, -1).tolist()
    dim = len(emb_list[0]) if emb_list and emb_list[0] else 0
    if dim <= 0:
        raise RuntimeError("Invalid embedding vector")

    coll = ensure_things_collection(dim=dim)
    coll.load()

    

    image_id = str(uuid.uuid4())
    payload = {
        "label": label,
        "bbox": [int(_x1), int(_y1), int(_x2), int(_y2)],
        "image_id": image_id,
        "source": "process",
        "source_job_id": job_id,
        "source_frame_idx": int(frame_idx),
    }
    payload_json = json.dumps(payload)

    try:
        has_group = any(getattr(f, "name", "") == "group_id" for f in coll.schema.fields)
    except Exception:
        has_group = False
    if has_group:
        coll.insert([emb_list, [payload_json], ["unknown"]])
    else:
        coll.insert([emb_list, [payload_json]])
    coll.flush()

    ensure_things_bucket()

    bio = io.BytesIO()
    crop.save(bio, format="JPEG", quality=90)
    put_minio_image("things", f"{image_id}.jpg", bio.getvalue(), "image/jpeg")
    put_minio_image("things", f"{image_id}_full.jpg", full_bytes, "image/jpeg")

    return {"status": "ok", "image_id": image_id}


