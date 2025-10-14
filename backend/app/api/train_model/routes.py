from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional

from ...service.model_trainer_service import start_training as mt_start, trainer_status as mt_status, terminate_training as mt_terminate


router = APIRouter()


class StartModelTrainRequest(BaseModel):
    epochs: int = 100
    yolo_model: str = "yolov8n.pt"
    synth_per_image: Optional[int] = None


@router.post("")
def start(req: StartModelTrainRequest) -> Dict[str, Any]:
    ok = mt_start(epochs=req.epochs, yolo_model=req.yolo_model, synth_per_image=req.synth_per_image)
    if not ok:
        raise HTTPException(status_code=409, detail="A model training job is already running")
    return {"status": "started"}


@router.get("/status")
def status() -> Dict[str, Any]:
    return mt_status()


@router.post("/terminate")
def terminate() -> Dict[str, Any]:
    mt_terminate()
    return {"status": "terminating"}


