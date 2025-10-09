from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict

from ...service.test_service import start_test, testing_status, terminate_test


router = APIRouter()


class StartTestRequest(BaseModel):
    path: str
    threshold: float = 0.7
    batch_size: int = 32


@router.post("")
def start(req: StartTestRequest) -> Dict[str, Any]:
    ok = start_test(path=req.path, threshold=req.threshold, batch_size=req.batch_size)
    if not ok:
        raise HTTPException(status_code=409, detail="A test is already running or path invalid")
    return {"status": "started", "batching": True, "batch_size": req.batch_size}


@router.get("/status")
def status() -> Dict[str, Any]:
    return testing_status()


@router.post("/terminate")
def terminate() -> Dict[str, Any]:
    terminate_test()
    return {"status": "terminating"}