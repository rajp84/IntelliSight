from __future__ import annotations

from typing import Any, Dict, List
import asyncio
import psutil
try:
    import GPUtil
except Exception: 
    GPUtil = None 

from ..socket.socket_manager import broadcast

from ..database.mongo import connect as mongo_connect, create_collection, find_one, update_one


COLLECTION_NAME = "configuration"
DOCUMENT_ID = {"_id": "default"}

_DEFAULT_TRAINING_PARAMS: Dict[str, Any] = {
    "batch_size": 1,
    "detect_every": 10,
    "frame_stride": 1,
    "max_new_tokens": 256,
    "mosaic_cols": 3,
    "mosaic_tile_scale": 1,
    "resize_width": 1024,
    "detection_debug": True,
    "tf32": True,
    "score_threshold": 0.15,
    "interpolate_boxes": True,
    "dtype_mode": "float16",
    "embedding_batch_size": 256,
}

_DEFAULT_CONFIG: Dict[str, Any] = {
    "library_path": "/",
    "training_params": _DEFAULT_TRAINING_PARAMS,
    "library_file_extensions": "jpg,jpeg,png,gif,webp,svg,bmp,tiff,mp4,avi,mov,wmv,flv,webm,mkv",
    "hf_token": "",
    "florence_model": "microsoft/Florence-2-large",
    "dinov3_model": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3_dimension": 768,
}


def get_dinov3_dimension(model_id: str) -> int:
    """Get the embedding dimension for a given DINOv3 model ID."""
    # ViT backbones
    if "vits16" in model_id:
        return 384
    elif "vitb16" in model_id:
        return 768
    elif "vitl16" in model_id:
        return 1024
    elif "vith16plus" in model_id:
        return 1280
    elif "vit7b16" in model_id:
        return 4096
    # ConvNeXt backbones
    elif "convnext-tiny" in model_id:
        return 384
    elif "convnext-small" in model_id:
        return 512
    elif "convnext-base" in model_id:
        return 1024
    elif "convnext-large" in model_id:
        return 1536
    else:
        # Default fallback
        return 768


def _ensure_collection() -> None:
    mongo_connect()
    # Ensure the configuration collection exists
    create_collection(COLLECTION_NAME)


def get_configuration() -> Dict[str, Any]:
    _ensure_collection()
    doc = find_one(COLLECTION_NAME, DOCUMENT_ID)
    if not doc:
        # No config yet; return defaults with environment variable fallback
        config = dict(_DEFAULT_CONFIG)
        # Check for Hugging Face token in environment variables
        import os
        if os.environ.get("HUGGINGFACE_HUB_TOKEN"):
            config["hf_token"] = os.environ.get("HUGGINGFACE_HUB_TOKEN")
        return config
    doc.pop("_id", None)
    # Backward compatibility: if flat training fields exist, fold them into training_params
    if "training_params" not in doc:
        tp: Dict[str, Any] = {}
        for k, dv in _DEFAULT_TRAINING_PARAMS.items():
            tp[k] = doc.get(k, dv)
        # Keep library_path at top-level
        # Auto-determine DINOv3 dimension based on selected model
        dinov3_model = doc.get("dinov3_model", _DEFAULT_CONFIG["dinov3_model"])
        dinov3_dimension = get_dinov3_dimension(dinov3_model)
        
        # Check for Hugging Face token in environment variables as fallback
        import os
        hf_token = doc.get("hf_token", _DEFAULT_CONFIG["hf_token"])
        if not hf_token and os.environ.get("HUGGINGFACE_HUB_TOKEN"):
            hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
        
        return {
            "library_path": doc.get("library_path", "/"),
            "library_file_extensions": doc.get("library_file_extensions", _DEFAULT_CONFIG["library_file_extensions"]),
            "hf_token": hf_token,
            "florence_model": doc.get("florence_model", _DEFAULT_CONFIG["florence_model"]),
            "dinov3_model": dinov3_model,
            "dinov3_dimension": dinov3_dimension,
            "training_params": tp,
        }
    # Ensure defaults for missing nested fields
    merged_tp: Dict[str, Any] = dict(_DEFAULT_TRAINING_PARAMS)
    merged_tp.update(doc.get("training_params", {}))
    # Auto-determine DINOv3 dimension based on selected model
    dinov3_model = doc.get("dinov3_model", _DEFAULT_CONFIG["dinov3_model"])
    dinov3_dimension = get_dinov3_dimension(dinov3_model)
    
    # Check for Hugging Face token in environment variables as fallback
    import os
    hf_token = doc.get("hf_token", _DEFAULT_CONFIG["hf_token"])
    if not hf_token and os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    
    return {
        "library_path": doc.get("library_path", "/"),
        "library_file_extensions": doc.get("library_file_extensions", _DEFAULT_CONFIG["library_file_extensions"]),
        "hf_token": hf_token,
        "florence_model": doc.get("florence_model", _DEFAULT_CONFIG["florence_model"]),
        "dinov3_model": dinov3_model,
        "dinov3_dimension": dinov3_dimension,
        "training_params": merged_tp,
    }


def save_configuration(config: Dict[str, Any]) -> None:
    _ensure_collection()
    # Create collection explicitly on write, to be explicit about intent
    create_collection(COLLECTION_NAME)
    update = {"$set": dict(config)}
    update_one(COLLECTION_NAME, DOCUMENT_ID, update, upsert=True)


__all__ = ["get_configuration", "save_configuration"]


async def _collect_system_stats() -> Dict[str, Any]:
    cpu_percent = psutil.cpu_percent(interval=None)
    virt = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    gpus: List[Dict[str, Any]] = []
    if GPUtil is not None:
        try:
            for g in GPUtil.getGPUs():
                gpus.append({
                    "name": getattr(g, 'name', 'GPU'),
                    "memory_total": getattr(g, 'memoryTotal', 0),
                    "memory_used": getattr(g, 'memoryUsed', 0),
                    "memory_util": getattr(g, 'memoryUtil', 0),
                    "load": getattr(g, 'load', 0),
                })
        except Exception:
            gpus = []
    return {
        "cpu": {"percent": cpu_percent},
        "memory": {"total": virt.total, "available": virt.available, "percent": virt.percent, "used": virt.used},
        "disk": {"total": disk.total, "percent": disk.percent, "used": disk.used, "free": disk.free},
        "gpus": gpus,
    }


_stats_task: asyncio.Task | None = None


async def _stats_loop(interval_seconds: float = 2.0) -> None:
    while True:
        data = await _collect_system_stats()
        await broadcast("system_stats", data)
        await asyncio.sleep(interval_seconds)


def start_system_stats_broadcast(loop: asyncio.AbstractEventLoop | None = None) -> None:
    global _stats_task
    if _stats_task and not _stats_task.done():
        return
    loop = loop or asyncio.get_event_loop()
    _stats_task = loop.create_task(_stats_loop())


