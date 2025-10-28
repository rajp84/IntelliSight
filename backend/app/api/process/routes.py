from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, HttpUrl

from ...service.process_service import (
    accept_job as svc_accept_job,
    list_jobs as svc_list_jobs,
    get_job as svc_get_job,
    get_job_frames as svc_get_job_frames,
    open_media_stream as svc_open_media_stream,
    list_frame_blocks as svc_list_frame_blocks,
    set_frame_invalid as svc_set_frame_invalid,
    add_frame_crop_to_things as svc_add_frame_crop_to_things,
    delete_job_and_frames as svc_delete_job_and_frames,
)


router = APIRouter()


class MediaFileRequest(BaseModel):
    job_id: str
    media_file: str
    callback_url: HttpUrl
    type: str | None = "detection"  # "detection" | "similarity_search"


@router.post("")
async def process_media(request: MediaFileRequest):
    media_path = Path(request.media_file)
    if not media_path.exists():
        raise HTTPException(status_code=404, detail="Media file not found")

    job_type = (request.type or "detection").strip() if isinstance(request.type, str) else "detection"
    if job_type not in {"detection", "similarity_search", "find_anything"}:
        job_type = "detection"

    try:
        svc_accept_job(request.job_id, media_path, str(request.callback_url), job_type)
    except Exception:
        # Keep API responsive per existing behavior
        pass

    return JSONResponse(status_code=202, content={"status": "Accepted", "job_id": request.job_id})


@router.get("/jobs")
async def list_jobs(limit: int = 100) -> Dict[str, List[Dict[str, Any]]]:
    try:
        items = svc_list_jobs(limit)
        return {"items": items}
    except Exception:
        return {"items": []}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str) -> Dict[str, Any]:
    try:
        doc = svc_get_job(job_id)
        if not doc:
            return {"error": "not_found"}
        return doc
    except Exception:
        return {"error": "unknown"}


@router.get("/jobs/{job_id}/frames")
async def get_job_frames(job_id: str, after_idx: int = -1, blocks: int = 3) -> Dict[str, Any]:
    try:
        return svc_get_job_frames(job_id, after_idx, blocks)
    except Exception:
        return {"frames": [], "last_idx": after_idx}


@router.get("/jobs/{job_id}/media")
async def stream_job_media(job_id: str, request: Request):
    try:
        body_iter, headers, status_code, content_type = svc_open_media_stream(
            job_id, request.headers.get("range")
        )
    except FileNotFoundError as ex:
        raise HTTPException(status_code=404, detail=str(ex))

    return StreamingResponse(body_iter, status_code=status_code, media_type=content_type, headers=headers)


@router.get("/jobs/{job_id}/frame-blocks")
async def list_frame_blocks(job_id: str) -> Dict[str, Any]:
    try:
        return svc_list_frame_blocks(job_id)
    except Exception:
        return {"blocks": []}


@router.post("/jobs/{job_id}/frames/{frame_idx}/invalid")
async def set_frame_invalid(job_id: str, frame_idx: int, invalid: bool = True) -> Dict[str, Any]:
    try:
        return svc_set_frame_invalid(job_id, frame_idx, invalid)
    except FileNotFoundError as ex:
        raise HTTPException(status_code=404, detail=str(ex))


class AddToThingsFromJobRequest(BaseModel):
    label: str
    x1: float
    y1: float
    x2: float
    y2: float


@router.post("/jobs/{job_id}/frames/{frame_idx}/add-to-things")
async def add_frame_crop_to_things(job_id: str, frame_idx: int, req: AddToThingsFromJobRequest) -> Dict[str, Any]:
    try:
        res = svc_add_frame_crop_to_things(
            job_id,
            frame_idx,
            req.label,
            req.x1,
            req.y1,
            req.x2,
            req.y2,
        )
        return res
    except FileNotFoundError as ex:
        raise HTTPException(status_code=404, detail=str(ex))
    except ValueError as ex:
        raise HTTPException(status_code=400, detail=str(ex))
    except RuntimeError as ex:
        raise HTTPException(status_code=500, detail=str(ex))
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to add to things: {ex}")


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str) -> Dict[str, Any]:
    try:
        return svc_delete_job_and_frames(job_id)
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to delete job: {ex}")

