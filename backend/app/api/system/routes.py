from fastapi import APIRouter, HTTPException, Query, UploadFile, File
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
    embedding_batch_size: int | None = Field(default=None)


class ConfigurationModel(BaseModel):
    # Allow forward-compatible extra fields
    model_config = ConfigDict(extra='allow')

    library_path: str = Field(..., description="Filesystem path to library root")
    training_params: TrainingParamsModel | None = Field(default=None)
    library_file_extensions: str = Field(default="jpg,jpeg,png,gif,webp,svg,bmp,tiff,mp4,avi,mov,wmv,flv,webm,mkv,mp3,wav,flac,aac,ogg,m4a", description="Comma-separated list of file extensions to show in library browser")
    hf_token: str | None = Field(default=None, description="Hugging Face access token for gated/private models")
    florence_model: str = Field(default="microsoft/Florence-2-large", description="Florence-2 model to use for object detection")
    dinov3_model: str = Field(default="facebook/dinov3-vitb16-pretrain-lvd1689m", description="DINOv3 model to use for embeddings")
    dinov3_dimension: int = Field(default=768, description="DINOv3 embedding dimension (auto-determined from model)")


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

    # Get file extension filter
    allowed_extensions = cfg.get("library_file_extensions", "").lower().split(",")
    allowed_extensions = [ext.strip() for ext in allowed_extensions if ext.strip()]

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
                
                if entry.is_dir():
                    # Always include directories
                    entries.append(LibraryListItem(name=entry.name, type='dir'))
                else:
                    # Filter files by extension
                    if allowed_extensions:
                        file_ext = entry.name.split('.')[-1].lower() if '.' in entry.name else ''
                        if file_ext in allowed_extensions:
                            entries.append(LibraryListItem(name=entry.name, type='file'))
                    else:
                        # If no extensions specified, include all files
                        entries.append(LibraryListItem(name=entry.name, type='file'))
        
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


@router.post("/system/library/upload")
def upload_library_file(
    file: UploadFile = File(...),
    path: str = Query(default="", description="Target directory relative to library root")
) -> Dict[str, Any]:
    """Upload a file to the library directory."""
    cfg = svc_get_configuration()
    root_str = cfg.get("library_path")
    if not root_str:
        raise HTTPException(status_code=400, detail="library_path is not configured")

    root = Path(root_str).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=400, detail="library_path does not exist or is not a directory")

    # Validate filename
    if not file.filename or file.filename.strip() == "":
        raise HTTPException(status_code=400, detail="Filename is required")
    
    # Sanitize filename
    filename = file.filename.strip()
    if "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Filename cannot contain path separators")
    
    # Create target directory if it doesn't exist
    target_dir = (root / path).resolve()
    _ensure_within_root(root, target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Create target file path
    target_file = target_dir / filename
    
    # Check if file already exists
    if target_file.exists():
        raise HTTPException(status_code=409, detail=f"File '{filename}' already exists")
    
    try:
        # Write file to disk
        with open(target_file, "wb") as f:
            content = file.file.read()
            f.write(content)
        
        return {
            "status": "success",
            "filename": filename,
            "path": str(target_file.relative_to(root)),
            "size": len(content)
        }
    except Exception as e:
        # Clean up file if it was created
        if target_file.exists():
            target_file.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


@router.post("/system/library/folder")
def create_library_folder(
    name: str = Query(..., description="Folder name"),
    path: str = Query(default="", description="Parent directory relative to library root")
) -> Dict[str, Any]:
    """Create a new folder in the library directory."""
    cfg = svc_get_configuration()
    root_str = cfg.get("library_path")
    if not root_str:
        raise HTTPException(status_code=400, detail="library_path is not configured")

    root = Path(root_str).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=400, detail="library_path does not exist or is not a directory")

    # Validate folder name
    if not name or name.strip() == "":
        raise HTTPException(status_code=400, detail="Folder name is required")
    
    # Sanitize folder name
    name = name.strip()
    if "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Folder name cannot contain path separators")
    
    if name in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid folder name")

    target_dir = (root / path).resolve()
    _ensure_within_root(root, target_dir)
    
    new_folder = target_dir / name
    
    try:
        if new_folder.exists():
            raise HTTPException(status_code=400, detail="Folder already exists")
        
        new_folder.mkdir(parents=True, exist_ok=False)
        return {
            "status": "success",
            "message": f"Folder '{name}' created successfully",
            "path": str(new_folder.relative_to(root))
        }
    except FileExistsError:
        raise HTTPException(status_code=400, detail="Folder already exists")
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create folder: {str(e)}")


# -------- Training routes --------

