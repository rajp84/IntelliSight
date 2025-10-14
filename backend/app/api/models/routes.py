from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List
from pathlib import Path
import shutil
import os
import logging
    

from ...service.system_service import get_configuration, save_configuration
from ...workers.process_worker import get_worker
from ...jobs.object_detection import ObjectDetection
import logging
logger = logging.getLogger(__name__)


logger = logging.getLogger(__name__)
router = APIRouter()


def _models_dir() -> Path:
    # Use absolute path inside container
    p = Path("/app/media/model")
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return p


def _list_local_models() -> List[Dict[str, Any]]:
    root = _models_dir()
    items: List[Dict[str, Any]] = []
    exts = {".pt", ".onnx", ".engine", ".tflite", ".xml", ".bin"}
    try:
        for child in sorted(root.glob("**/*")):
            if child.is_file() and child.suffix.lower() in exts:
                rel = child.relative_to(root)
                items.append({
                    "name": str(rel).replace("\\", "/"),
                    "size": child.stat().st_size,
                    "modified": int(child.stat().st_mtime * 1000),
                })
    except Exception:
        pass
    return items


class SetDefaultRequest(BaseModel):
    name: str


class SetRoboflowKeyRequest(BaseModel):
    api_key: str


class DownloadRoboflowRequest(BaseModel):
    workspace: str
    project: str
    version: int
    format: str = "yolov8"  # roboflow model format


@router.get("")
def list_models() -> Dict[str, Any]:
    cfg = get_configuration()
    items = _list_local_models()
    default_model = cfg.get("default_model") or ""
    rf_key = cfg.get("roboflow_api_key") or ""
    return {"items": items, "default_model": default_model, "roboflow_api_key": rf_key}


@router.post("/default")
def set_default(req: SetDefaultRequest) -> Dict[str, Any]:
    # Validate exists
    target = _models_dir() / req.name
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Model file not found")
    cfg = get_configuration()
    cfg["default_model"] = req.name
    save_configuration(cfg)
    return {"status": "ok", "default_model": req.name}


@router.post("/roboflow/key")
def set_roboflow_key(req: SetRoboflowKeyRequest) -> Dict[str, Any]:
    key = (req.api_key or "").strip()
    cfg = get_configuration()
    cfg["roboflow_api_key"] = key
    save_configuration(cfg)
    return {"status": "ok"}


@router.post("/roboflow/download")
def download_from_roboflow(req: DownloadRoboflowRequest) -> Dict[str, Any]:
    logger.info(f"Downloading model from Roboflow: {req.workspace} {req.project} {req.version}")
    cfg = get_configuration()
    api_key = (cfg.get("roboflow_api_key") or "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="Roboflow API key not set")

    try:
        # Lazy import to avoid hard dependency when not used
        from roboflow import Roboflow  # type: ignore
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Roboflow package not available: {ex}")

    try:
        import roboflow  # type: ignore
        rf = roboflow.Roboflow(api_key=api_key)  # type: ignore[attr-defined]
        workspace = None
        try:
            workspace = rf.workspace(req.workspace)
            print("workspace", workspace)
        except Exception:
            try:
                # Fallback to default workspace bound to key
                workspace = rf.workspace()
                print("workspace", workspace)
            except Exception as e:
                print("Error listing workspaces", e)
                workspace = None

        proj = workspace.project(req.project)
        ver = proj.version(req.version)
        
        model = ver.model

        logger.info(f"Downloading model from Roboflow: {model}")

        lib_path = cfg.get("library_path", "")
        models_dir = lib_path + "/model"

        out_dir = model.download(location=models_dir, format="pt")
        
        final_path = os.path.join(models_dir, req.project + "_" + str(req.version) + ".pt")        
        os.rename(models_dir + "/weights.pt", final_path)
        logger.info(f"Model downloaded to: {final_path}")
        
        return {"status": "ok"}
    except HTTPException:
        raise
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to download from Roboflow: {ex}")


@router.get("/roboflow/workspaces")
def list_roboflow_workspaces() -> Dict[str, Any]:
    cfg = get_configuration()
    api_key = (cfg.get("roboflow_api_key") or "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="Roboflow API key not set")
    try:
        import roboflow  # type: ignore
        rf = roboflow.Roboflow(api_key=api_key)  # type: ignore[attr-defined]
        ws = rf.workspace()  # default workspace bound to this key
        slug = getattr(ws, 'slug', None)
        name = getattr(ws, 'name', None)
        wid = getattr(ws, 'id', None)
        ident = slug or wid or (name or "workspace")
        item = {"id": str(wid or ident), "name": str(name or ident), "slug": str(slug or ident)}
        return {"items": [item]}
    except HTTPException:
        raise
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to list Roboflow workspaces: {ex}")


@router.get("/roboflow/projects")
def list_roboflow_projects(workspace: str) -> Dict[str, Any]:
    cfg = get_configuration()
    api_key = (cfg.get("roboflow_api_key") or "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="Roboflow API key not set")
    try:
        import roboflow  # type: ignore
        rf = roboflow.Roboflow(api_key=api_key)  # type: ignore[attr-defined]
        items: list[dict] = []
        # Try explicit workspace first
        ws_obj = None
        try:
            ws_obj = rf.workspace(workspace)
            print("ws_obj", ws_obj)
        except Exception:
            try:
                # Fallback to default workspace bound to key
                ws_obj = rf.workspace()
                print("ws_obj", ws_obj)
            except Exception as e:
                print("Error listing workspaces", e)
                ws_obj = None
        if ws_obj is not None:
            projs = None
            try:
                projs = ws_obj.projects()  # type: ignore[attr-defined]
                print("projs", projs)
            except Exception as e:
                print("Error listing projects", e)
                projs = None
            if projs is not None:
                for p in projs or []:
                    # SDK may return strings like "workspaceSlug/projectSlug"
                    if isinstance(p, str):
                        seg = p.split("/", 1)
                        proj_slug = seg[1] if len(seg) > 1 else seg[0]
                        items.append({
                            "id": proj_slug,
                            "name": proj_slug,
                            "slug": proj_slug,
                        })
                    elif isinstance(p, dict):
                        slug = p.get("slug") or p.get("id")
                        name = p.get("name") or slug
                        if slug:
                            items.append({"id": str(slug), "name": str(name), "slug": str(slug)})
                    else:
                        slug = getattr(p, 'slug', None) or getattr(p, 'id', None)
                        name = getattr(p, 'name', None) or slug
                        if slug:
                            items.append({"id": str(slug), "name": str(name), "slug": str(slug)})
        
        return {"items": items}
    except HTTPException:
        raise
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to list Roboflow projects: {ex}")


@router.get("/roboflow/versions")
def list_roboflow_versions(workspace: str, project: str) -> Dict[str, Any]:
    cfg = get_configuration()
    api_key = (cfg.get("roboflow_api_key") or "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="Roboflow API key not set")
    try:
        import roboflow  # type: ignore
        rf = roboflow.Roboflow(api_key=api_key)  # type: ignore[attr-defined]
        items: list[dict] = []
        proj_obj = None
        try:
            workspace_obj = rf.workspace(workspace)            
            proj_obj = workspace_obj.project(project)
            print("proj_obj", proj_obj)
        except Exception as e:
            print("Error listing projects", e)
            try:
                proj_obj = rf.workspace().project(project)
                print("proj_obj", proj_obj)
            except Exception as e:
                print("Error listing projects", e)
                proj_obj = None
        if proj_obj is not None:
            vers = None
            try:
                vers = proj_obj.versions()  # type: ignore[attr-defined]
            except Exception:
                vers = None
            if vers is not None:
                for v in vers or []:
                    raw_version = (
                        getattr(v, 'version', None)
                        or (v.get('version') if isinstance(v, dict) else None)
                        or getattr(v, 'id', None)
                        or (v.get('id') if isinstance(v, dict) else None)
                    )
                    ver_str = str(raw_version) if raw_version is not None else ''
                    try:
                        ver_num = int(ver_str.split('/')[-1])
                    except Exception:
                        try:
                            ver_num = int(ver_str)
                        except Exception:
                            continue
                    name = getattr(v, 'name', None) or (v.get('name') if isinstance(v, dict) else None) or str(ver_num)
                    items.append({"id": ver_num, "name": str(name), "version": ver_num})
        # Fallback REST
        if not items:
            import requests
            tried = [(workspace, project)]
            wl = (workspace or "").lower(); pl = (project or "").lower()
            if wl and pl and (wl, pl) not in tried:
                tried.append((wl, pl))
            for w, p in tried:
                r = requests.get(f"https://api.roboflow.com/{w}/{p}/versions", params={"api_key": api_key}, timeout=10)
                if r.ok:
                    data = r.json() if hasattr(r, 'json') else {}
                for v in (data or {}).get('versions') or (data or {}).get('data') or []:
                    raw_version = v.get('version') or v.get('id')
                    ver_str = str(raw_version) if raw_version is not None else ''
                    try:
                        ver_num = int(ver_str.split('/')[-1])
                    except Exception:
                        try:
                            ver_num = int(ver_str)
                        except Exception:
                            continue
                    name = v.get('name') or str(ver_num)
                    items.append({"id": ver_num, "name": str(name), "version": ver_num})
                    if items:
                        break
        return {"items": items}
    except HTTPException:
        raise
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to list Roboflow versions: {ex}")


