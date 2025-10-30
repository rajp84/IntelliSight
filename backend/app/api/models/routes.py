from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List
import logging

from ...service.system_service import get_configuration
from ...service.roboflow_service import (
    list_local_models as service_list_local_models,
    set_default_model as service_set_default_model,
    set_roboflow_api_key as service_set_roboflow_api_key,
    download_from_roboflow as service_download_from_roboflow,
    list_roboflow_workspaces as service_list_workspaces,
    list_roboflow_projects as service_list_projects,
    list_roboflow_versions as service_list_versions,
    list_local_datasets as service_list_local_datasets,
    download_roboflow_dataset as service_download_dataset,
    delete_local_model as service_delete_local_model,
    delete_local_dataset as service_delete_local_dataset,
)
from ...service.dataset_import_service import import_dataset_as_training

logger = logging.getLogger(__name__)


logger = logging.getLogger(__name__)
router = APIRouter()
 


class SetDefaultRequest(BaseModel):
    name: str


class SetRoboflowKeyRequest(BaseModel):
    api_key: str


class DownloadRoboflowRequest(BaseModel):
    workspace: str
    project: str
    version: int
    format: str = "yolov8"  # roboflow model format


class DownloadRoboflowDatasetRequest(BaseModel):
    workspace: str
    project: str
    version: int
    format: str = "yolov8"  # e.g., yolov8, coco, voc, etc.


class ImportDatasetRequest(BaseModel):
    path: str
    split: str = "train"
    format: str = "yolov8"


class DeleteModelRequest(BaseModel):
    name: str


class DeleteDatasetRequest(BaseModel):
    name: str


@router.get("")
def list_models() -> Dict[str, Any]:
    cfg = get_configuration()
    items = service_list_local_models()
    default_model = cfg.get("default_model") or ""
    rf_key = cfg.get("roboflow_api_key") or ""
    return {"items": items, "default_model": default_model, "roboflow_api_key": rf_key}


@router.post("/default")
def set_default(req: SetDefaultRequest) -> Dict[str, Any]:
    name = service_set_default_model(req.name)
    return {"status": "ok", "default_model": name}


@router.post("/roboflow/key")
def set_roboflow_key(req: SetRoboflowKeyRequest) -> Dict[str, Any]:
    service_set_roboflow_api_key(req.api_key)
    return {"status": "ok"}


@router.post("/delete")
def delete_model(req: DeleteModelRequest) -> Dict[str, Any]:
    service_delete_local_model(req.name)
    return {"status": "ok"}


@router.post("/roboflow/download")
def download_from_roboflow(req: DownloadRoboflowRequest) -> Dict[str, Any]:
    _ = service_download_from_roboflow(req.workspace, req.project, req.version)
    return {"status": "ok"}


@router.get("/roboflow/workspaces")
def list_roboflow_workspaces() -> Dict[str, Any]:
    return service_list_workspaces()


@router.get("/roboflow/projects")
def list_roboflow_projects(workspace: str) -> Dict[str, Any]:
    return service_list_projects(workspace)


@router.get("/roboflow/versions")
def list_roboflow_versions(workspace: str, project: str) -> Dict[str, Any]:
    return service_list_versions(workspace, project)


@router.get("/roboflow/datasets")
def list_roboflow_datasets() -> Dict[str, Any]:
    items = service_list_local_datasets()
    return {"items": items}


@router.post("/roboflow/dataset/download")
def download_roboflow_dataset(req: DownloadRoboflowDatasetRequest) -> Dict[str, Any]:
    res = service_download_dataset(req.workspace, req.project, req.version, req.format)
    path = res.get("path") if isinstance(res, dict) else res
    return {"status": "ok", "path": path}


@router.post("/roboflow/dataset/import")
def import_roboflow_dataset(req: ImportDatasetRequest) -> Dict[str, Any]:
    res = import_dataset_as_training(req.path, split=req.split, label_format=req.format)
    return {"status": "ok", **res}


@router.post("/roboflow/dataset/delete")
def delete_roboflow_dataset(req: DeleteDatasetRequest) -> Dict[str, Any]:
    service_delete_local_dataset(req.name)
    return {"status": "ok"}

