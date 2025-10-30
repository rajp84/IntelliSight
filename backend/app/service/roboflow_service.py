from __future__ import annotations

from typing import Any, Dict, List
from pathlib import Path
import os
import logging
import requests
import roboflow
from fastapi import HTTPException
import shutil

from .system_service import get_configuration, save_configuration


logger = logging.getLogger(__name__)


def _models_dir() -> Path:
    # Use absolute path inside container
    p = Path("/app/media/model")
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return p


def list_local_models() -> List[Dict[str, Any]]:
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


def set_default_model(name: str) -> str:
    target = _models_dir() / name
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Model file not found")
    cfg = get_configuration()
    cfg["default_model"] = name
    save_configuration(cfg)
    return name


def set_roboflow_api_key(api_key: str) -> None:
    key = (api_key or "").strip()
    cfg = get_configuration()
    cfg["roboflow_api_key"] = key
    save_configuration(cfg)


def delete_local_model(name: str) -> None:
    root = _models_dir()
    try:
        target = (root / name).resolve()
        root_resolved = root.resolve()
        if not str(target).startswith(str(root_resolved)):
            raise HTTPException(status_code=400, detail="Invalid model path")
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="Model file not found")
        # Prevent deleting default model
        cfg = get_configuration()
        if (cfg.get("default_model") or "") == name:
            raise HTTPException(status_code=400, detail="Cannot delete the default model")
        target.unlink()
        try:
            # Clean up empty parent dirs under models_dir
            parent = target.parent
            while parent != root_resolved and parent.is_dir():
                if any(parent.iterdir()):
                    break
                parent.rmdir()
                parent = parent.parent
        except Exception:
            pass
    except HTTPException:
        raise
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {ex}")


def download_from_roboflow(workspace: str, project: str, version: int) -> str:
    logger.info(f"Downloading model from Roboflow: {workspace} {project} {version}")
    cfg = get_configuration()
    api_key = (cfg.get("roboflow_api_key") or "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="Roboflow API key not set")

    try:
        rf = roboflow.Roboflow(api_key=api_key)  # type: ignore[attr-defined]

        # Resolve workspace (explicit or default bound to key)
        try:
            ws_obj = rf.workspace(workspace)
        except Exception:
            try:
                ws_obj = rf.workspace()
            except Exception as e:
                logger.error("Error resolving workspace: %s", e)
                ws_obj = None

        if ws_obj is None:
            raise HTTPException(status_code=400, detail="Unable to resolve Roboflow workspace")

        proj = ws_obj.project(project)
        ver = proj.version(version)
        model = ver.model

        logger.info(f"Downloading model from Roboflow: {model}")

        lib_path = cfg.get("library_path", "")
        models_dir = lib_path + "/model"

        out_dir = model.download(location=models_dir, format="pt")
        # Normalize and rename output weights to a stable filename
        final_path = os.path.join(models_dir, project + "_" + str(version) + ".pt")
        src_path = os.path.join(models_dir, "weights.pt")
        if not os.path.exists(src_path):
            # Some SDK versions may place the file inside a subdirectory
            candidate = os.path.join(out_dir or models_dir, "weights.pt")
            if os.path.exists(candidate):
                src_path = candidate
        os.rename(src_path, final_path)
        logger.info(f"Model downloaded to: {final_path}")

        return final_path
    except HTTPException:
        raise
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to download from Roboflow: {ex}")


def list_roboflow_workspaces() -> Dict[str, Any]:
    cfg = get_configuration()
    api_key = (cfg.get("roboflow_api_key") or "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="Roboflow API key not set")
    try:
        
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


def list_roboflow_projects(workspace: str) -> Dict[str, Any]:
    cfg = get_configuration()
    api_key = (cfg.get("roboflow_api_key") or "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="Roboflow API key not set")
    try:
        rf = roboflow.Roboflow(api_key=api_key)  # type: ignore[attr-defined]
        items: List[Dict[str, Any]] = []
        # Try explicit workspace first, then default
        ws_obj = None
        try:
            ws_obj = rf.workspace(workspace)
        except Exception:
            try:
                ws_obj = rf.workspace()
            except Exception as e:
                logger.error("Error listing workspaces: %s", e)
                ws_obj = None
        if ws_obj is not None:
            projs = None
            try:
                projs = ws_obj.projects()  # type: ignore[attr-defined]
            except Exception as e:
                logger.error("Error listing projects: %s", e)
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


def list_roboflow_versions(workspace: str, project: str) -> Dict[str, Any]:
    cfg = get_configuration()
    api_key = (cfg.get("roboflow_api_key") or "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="Roboflow API key not set")
    try:
        rf = roboflow.Roboflow(api_key=api_key)  # type: ignore[attr-defined]
        items: List[Dict[str, Any]] = []
        proj_obj = None
        try:
            workspace_obj = rf.workspace(workspace)
            proj_obj = workspace_obj.project(project)
        except Exception as e:
            logger.error("Error resolving project: %s", e)
            try:
                proj_obj = rf.workspace().project(project)
            except Exception as e2:
                logger.error("Error resolving default project: %s", e2)
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
            tried = [(workspace, project)]
            wl = (workspace or "").lower(); pl = (project or "").lower()
            if wl and pl and (wl, pl) not in tried:
                tried.append((wl, pl))
            for w, p in tried:
                r = requests.get(f"https://api.roboflow.com/{w}/{p}/versions", params={"api_key": api_key}, timeout=10)
                if r.ok:
                    data = r.json() if hasattr(r, 'json') else {}
                else:
                    data = {}
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



# Dataset helpers and APIs
def _datasets_dir() -> Path:
    # Use absolute path inside container for dataset storage
    p = Path("/app/media/dataset")
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return p


def list_local_datasets() -> List[Dict[str, Any]]:
    root = _datasets_dir()
    items: List[Dict[str, Any]] = []
    try:
        # List only immediate children (datasets are usually directories)
        for child in sorted(root.iterdir()):
            if child.is_dir():
                # Best-effort size calculation (may be large)
                size_bytes = 0
                try:
                    for p in child.rglob("*"):
                        if p.is_file():
                            size_bytes += p.stat().st_size
                except Exception:
                    pass
                items.append({
                    "name": child.name,
                    "size": size_bytes,
                    "modified": int(child.stat().st_mtime * 1000),
                    "path": str(child),
                })
    except Exception:
        pass
    return items


def download_roboflow_dataset(workspace: str, project: str, version: int, fmt: str = "yolov8") -> str:
    logger.info(f"Downloading dataset from Roboflow: {workspace} {project} {version} format={fmt}")
    cfg = get_configuration()
    api_key = (cfg.get("roboflow_api_key") or "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="Roboflow API key not set")

    try:
        rf = roboflow.Roboflow(api_key=api_key)  # type: ignore[attr-defined]

        # Resolve workspace (explicit or default bound to key)
        try:
            ws_obj = rf.workspace(workspace)
        except Exception:
            try:
                ws_obj = rf.workspace()
            except Exception as e:
                logger.error("Error resolving workspace for dataset: %s", e)
                ws_obj = None

        if ws_obj is None:
            raise HTTPException(status_code=400, detail="Unable to resolve Roboflow workspace")

        proj = ws_obj.project(project)
        ver = proj.version(version)

        lib_path = cfg.get("library_path", "")
        datasets_dir = lib_path + "/dataset" if lib_path else str(_datasets_dir())
        out_dir = f"{datasets_dir}/{project}_{version}"
        # Roboflow SDK returns the directory path where dataset is downloaded
        dataset = ver.download(model_format=fmt, location=out_dir)
        logger.info(f"Dataset downloaded to: {out_dir}")
        return {"path": out_dir}
    except HTTPException:
        raise
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to download Roboflow dataset: {ex}")


def delete_local_dataset(name: str) -> None:
    """Delete a local dataset directory under the datasets root by name."""
    root = _datasets_dir()
    try:
        # Resolve and ensure the target path is within the datasets root
        target = (root / name).resolve()
        root_resolved = root.resolve()
        if not str(target).startswith(str(root_resolved)):
            raise HTTPException(status_code=400, detail="Invalid dataset path")
        if not target.exists() or not target.is_dir():
            raise HTTPException(status_code=404, detail="Dataset not found")
        # Remove directory recursively
        shutil.rmtree(target)
        # Best-effort cleanup of empty parent directories within the root
        try:
            parent = target.parent
            while parent != root_resolved and parent.is_dir():
                if any(parent.iterdir()):
                    break
                parent.rmdir()
                parent = parent.parent
        except Exception:
            pass
    except HTTPException:
        raise
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {ex}")

