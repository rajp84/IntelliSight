from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Dict, List, Optional
from pathlib import Path
import os

from ...service.system_service import get_configuration as svc_get_configuration, save_configuration as svc_save_configuration


router = APIRouter()


class TrainingParamsModel(BaseModel):
    model_config = ConfigDict(extra='allow')
    batch_size: int | None = Field(default=None)
    detect_every: int | None = Field(default=None)
    frame_stride: int | None = Field(default=None)
    max_new_tokens: int | None = Field(default=None)
    mosaic_cols: int | None = Field(default=None)
    mosaic_tile_scale: float | None = Field(default=None)
    resize_width: int | None = Field(default=None)
    detection_debug: bool | None = Field(default=None)
    tf32: bool | None = Field(default=None)
    score_threshold: float | None = Field(default=None)
    interpolate_boxes: bool | None = Field(default=None)
    dtype_mode: str | None = Field(default=None)


class ConfigurationModel(BaseModel):
    # Allow forward-compatible extra fields
    model_config = ConfigDict(extra='allow')

    library_path: str = Field(..., description="Filesystem path to library root")
    training_params: TrainingParamsModel | None = Field(default=None)


@router.get("/system/config")
def get_configuration() -> Dict[str, Any]:
    return svc_get_configuration()


@router.post("/system/config")
def save_configuration(config: ConfigurationModel) -> Dict[str, Any]:
    # Persist exactly what was provided (including optional fields)
    payload = config.model_dump()
    svc_save_configuration(payload)
    return {"status": "ok"}


# -------- Library listing --------
class LibraryListItem(BaseModel):
    name: str
    type: str  # 'file' | 'dir'


class LibraryListResponse(BaseModel):
    root: str
    path: str
    items: List[LibraryListItem]


def _ensure_within_root(root: Path, target: Path) -> None:
    try:
        target.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=400, detail="Path is outside of library root")


@router.get("/system/library/list", response_model=LibraryListResponse)
def list_library(path: Optional[str] = Query(default="", description="Subpath relative to library root")) -> Any:
    cfg = svc_get_configuration()
    root_str = cfg.get("library_path")
    if not root_str:
        raise HTTPException(status_code=400, detail="library_path is not configured")

    root = Path(root_str).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=400, detail="library_path does not exist or is not a directory")

    target = (root / path).resolve()
    _ensure_within_root(root, target)

    try:
        entries: List[LibraryListItem] = []
        with os.scandir(target) as it:
            for entry in it:
                # Skip hidden files/folders
                if entry.name.startswith('.'):
                    continue
                entry_type = 'dir' if entry.is_dir() else 'file'
                entries.append(LibraryListItem(name=entry.name, type=entry_type))
        # Sort: dirs first, then files, both alphabetically
        entries.sort(key=lambda e: (0 if e.type == 'dir' else 1, e.name.lower()))
        rel_path = str(target.relative_to(root)) if target != root else ""
        return LibraryListResponse(root=str(root), path=rel_path, items=entries)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Path not found")
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")


@router.get("/system/library/file")
def get_library_file(path: str = Query(..., description="File path relative to library root")):
    cfg = svc_get_configuration()
    root_str = cfg.get("library_path")
    if not root_str:
        raise HTTPException(status_code=400, detail="library_path is not configured")

    root = Path(root_str).expanduser().resolve()
    target = (root / path).resolve()
    _ensure_within_root(root, target)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(target))


# -------- Training routes --------

