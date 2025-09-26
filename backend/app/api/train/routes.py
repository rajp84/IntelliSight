from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional

from ...service.training_service import start_od_training, training_status, _STATE

router = APIRouter()


class StartTrainingRequest(BaseModel):
    mosaic: bool = False
    extra_args: Optional[Dict[str, Any]] = None
    path: str


@router.post("")
def start_training(req: StartTrainingRequest) -> Dict[str, Any]:
    extra = dict(req.extra_args or {})
    if req.path:
        extra.update({"input": req.path})
    ok = start_od_training(mosaic=req.mosaic, extra_args=extra)
    if not ok:
        raise HTTPException(status_code=409, detail="A training job is already running")
    return {"status": "started"}


@router.get("/status")
def get_training_status() -> Dict[str, Any]:
    return training_status()


@router.post("/terminate")
def terminate_training() -> Dict[str, Any]:
    _STATE.request_stop()
    return {"status": "terminating"}


