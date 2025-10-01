from __future__ import annotations

import asyncio
import threading
import time
from queue import Queue
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image


_LOCK = threading.Lock()
_EMBEDDER: Optional[dict] = None


def _to_rgb_image(img: Union[Image.Image, np.ndarray]) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    arr = np.asarray(img)
    if arr.ndim == 2:  # grayscale
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:  # RGBA -> RGB
        arr = arr[..., :3]
    # Heuristic: if appears BGR (OpenCV), swap
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 3 and np.mean(arr[..., 2]) > np.mean(arr[..., 0]):
        # Likely RGB already; keep as-is
        rgb = arr
    else:
        # Swap BGR->RGB (safe for grayscale replicated)
        rgb = arr[..., ::-1]
    return Image.fromarray(rgb)


def load_dino(
    model_id: str = "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    *,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> Tuple[Any, Any, str]:
    """Load the specified DINOv3 model from Hugging Face and return (processor, model, device_str).

    - Only uses the official DINOv3 weights: facebook/dinov3-vit7b16-pretrain-lvd1689m
    - device: "cuda", "cpu" (auto by default).
    - dtype: None/"auto"/"float16"/"bfloat16"/"float32".
    """
    global _EMBEDDER
    with _LOCK:
        try:
            import torch
        except Exception as ex:
            raise RuntimeError(f"PyTorch required for DINO embedder: {ex}")

        want_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype_map = {
            None: None,
            "auto": None,
            "float16": torch.float16,
            "bfloat16": getattr(torch, "bfloat16", torch.float16),
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype if dtype is None or isinstance(dtype, str) else str(dtype), None)

        # Reuse cache when identical
        if _EMBEDDER is not None:
            if (
                _EMBEDDER.get("model_id") == model_id
                and _EMBEDDER.get("device") == want_device
                and _EMBEDDER.get("dtype") == (torch_dtype or "auto")
            ):
                return _EMBEDDER["transform"], _EMBEDDER["model"], want_device

        # Load via Hugging Face
        from transformers import AutoImageProcessor, AutoProcessor, AutoModel

        # Configure HF token if provided
        import os
        if hf_token and not os.environ.get("HUGGINGFACE_HUB_TOKEN"):
            os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
        elif not hf_token and os.environ.get("HUGGINGFACE_HUB_TOKEN"):
            # Use environment variable as fallback if no token provided
            hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")

        # Load processor (try AutoImageProcessor first, then AutoProcessor)
        processor = None
        last_err: Optional[Exception] = None
        for proc_loader in (
            lambda: AutoImageProcessor.from_pretrained(
                model_id,
                token=hf_token if hf_token else None,
            ),
            lambda: AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
                token=hf_token if hf_token else None,
            ),
        ):
            try:
                processor = proc_loader()
                break
            except Exception as e:
                last_err = e
                continue
        if processor is None:
            raise RuntimeError(
                f"Failed to load processor for {model_id}. Ensure your Hugging Face token is set in config and has access. Last error: {last_err}"
            )

        # Load model
        try:
            model = AutoModel.from_pretrained(
                model_id,
                trust_remote_code=True,
                token=hf_token if hf_token else None,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model {model_id}. Ensure your Hugging Face token is set in config and has access. Error: {e}"
            )
        model.eval()
        model.to(want_device)
        if torch_dtype is not None:
            try:
                model = model.to(dtype=torch_dtype)
            except Exception:
                pass

        _EMBEDDER = {
            "transform": processor,  # handled specially downstream
            "model": model,
            "device": want_device,
            "dtype": torch_dtype or "auto",
            "model_id": model_id,
        }
        return processor, model, want_device


def embed_image(
    image: Union[Image.Image, np.ndarray],
    *,
    model_id: str = "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    normalize: bool = True,
) -> np.ndarray:
    """Compute a single-image embedding using Dinov2 and return 1D np.ndarray.

    Accepts PIL Image or numpy array (H,W,3) in RGB or BGR (auto-detected).
    """
    emb = embed_images([image], model_id=model_id, device=device, dtype=dtype, batch_size=1, normalize=normalize)
    return emb[0]


def embed_images(
    images: Sequence[Union[Image.Image, np.ndarray]],
    *,
    model_id: str = "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    batch_size: int = 16,
    normalize: bool = True,
) -> np.ndarray:
    """Compute embeddings for a batch of images. Returns (N, D) float32 array.

    - Uses CLS token / pooled output if provided by the model; otherwise mean-pools patch tokens.
    - Batches inputs for memory efficiency; set batch_size based on your GPU.
    """
    # Fetch HF token from configuration if available
    try:
        from app.service.system_service import get_configuration as _get_cfg  # lazy import to avoid cycles
        _cfg = _get_cfg()
        _hf_token = _cfg.get("hf_token") or None
    except Exception:
        _hf_token = None

    # Use the configured model from system config to ensure cache hit
    # This ensures we use the same model that was preloaded
    try:
        from app.service.system_service import get_configuration as _get_cfg
        _cfg = _get_cfg()
        configured_dinov3_model = _cfg.get("dinov3_model", "facebook/dinov3-vitb16-pretrain-lvd1689m")
        if model_id == "facebook/dinov3-vit7b16-pretrain-lvd1689m":  # Default model
            model_id = configured_dinov3_model  # Use the configured model
    except Exception:
        # Fallback to original behavior if config unavailable
        if model_id == "facebook/dinov3-vit7b16-pretrain-lvd1689m":  # Default model
            model_id = "facebook/dinov3-vitb16-pretrain-lvd1689m"  # Use the same model we preloaded
    
    transform_or_processor, model, dev = load_dino(model_id=model_id, device=device, dtype=dtype, hf_token=_hf_token)

    import torch
    feats: List[np.ndarray] = []
    imgs_rgb: List[Image.Image] = [
        _to_rgb_image(img) for img in images
    ]

    with torch.no_grad():
        # HF processor path only
        is_timm = False
        for i in range(0, len(imgs_rgb), max(1, int(batch_size))):
            chunk = imgs_rgb[i : i + batch_size]
            if is_timm:
                tensors = [transform_or_processor(im).unsqueeze(0) for im in chunk]
                inputs = torch.cat(tensors, dim=0).to(dev)
                outputs = model.forward_features(inputs)
                # try multiple keys / shapes
                if isinstance(outputs, dict):
                    x = outputs.get("x_norm_clstoken") or outputs.get("cls_token") or outputs.get("pooled") or outputs.get("x")
                    if x is None:
                        # fall back to any first tensor-like value
                        for v in outputs.values():
                            if isinstance(v, torch.Tensor):
                                x = v
                                break
                    if x is None:
                        raise RuntimeError("Dinov3 features missing from forward_features dict")
                    # if token sequence, mean-pool
                    if x.dim() == 3:
                        x = x.mean(dim=1)
                else:
                    x = outputs
                    if x.dim() == 3:
                        x = x.mean(dim=1)
            else:
                # HF processor path (dinov2 fallback)
                processor = transform_or_processor
                inputs = processor(images=chunk, return_tensors="pt")
                inputs = {k: v.to(dev) for k, v in inputs.items()}
                outputs = model(**inputs)
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    x = outputs.pooler_output
                elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                    x = outputs.last_hidden_state.mean(dim=1)
                else:
                    x = getattr(outputs, "embeddings", None)
                    if x is None:
                        raise RuntimeError("Unexpected model outputs; cannot derive embeddings")

            x = x.float()
            if normalize:
                x = torch.nn.functional.normalize(x, p=2, dim=1)
            feats.append(x.detach().cpu().numpy())

    if not feats:
        return np.zeros((0, 1), dtype=np.float32)
    return np.concatenate(feats, axis=0).astype(np.float32, copy=False)


def is_model_loaded() -> bool:
    """Check if a model is currently loaded and cached."""
    return _EMBEDDER is not None


def get_loaded_model_info() -> Optional[dict]:
    """Get information about the currently loaded model."""
    if _EMBEDDER is None:
        return None
    return {
        "model_id": _EMBEDDER.get("model_id"),
        "device": _EMBEDDER.get("device"),
        "dtype": _EMBEDDER.get("dtype"),
    }


def preload_model(
    model_id: Optional[str] = None,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> bool:
    """Preload the model to avoid loading on first embedding call.
    
    Returns True if model was loaded successfully, False otherwise.
    """
    try:
        # Get token and model_id from configuration if not provided
        if hf_token is None or model_id is None:
            from app.service.system_service import get_configuration
            config = get_configuration()
            if hf_token is None:
                hf_token = config.get("hf_token") or None
            if model_id is None:
                model_id = config.get("dinov3_model", "facebook/dinov3-vit7b16-pretrain-lvd1689m")
        
        print(f"[DEBUG] embedder_service: Preloading model: {model_id}")
        
        # Pre-download checkpoint shards for large models
        try:
            from huggingface_hub import snapshot_download
            import os
            
            # Set HF token in environment if provided
            if hf_token and not os.environ.get("HUGGINGFACE_HUB_TOKEN"):
                os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
            elif not hf_token and os.environ.get("HUGGINGFACE_HUB_TOKEN"):
                # Use environment variable as fallback if no token provided
                hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
            
            # Download model files including shards
            print(f"[DEBUG] embedder_service: Downloading model files and shards for {model_id}")
            cache_dir = snapshot_download(
                repo_id=model_id,
                token=hf_token,
                allow_patterns=["*.bin", "*.safetensors", "*.json", "*.txt", "*.onnx"],
                local_files_only=False,
            )
            print(f"[DEBUG] embedder_service: Model files downloaded to: {cache_dir}")
        except Exception as e:
            print(f"[DEBUG] embedder_service: Warning - Could not pre-download shards: {e}")
            # Continue anyway, model loading might still work
        
        # Load the model (this will use cached files if available)
        print(f"[DEBUG] embedder_service: Loading model into memory...")
        transform, model, dev = load_dino(model_id=model_id, device=device, dtype=dtype, hf_token=hf_token)
        
        # Verify model is actually loaded and ready
        print(f"[DEBUG] embedder_service: Verifying model is ready...")
        if model is None:
            raise RuntimeError("Model failed to load")
        
        # Test the model with a dummy input to ensure it's fully ready
        try:
            import torch
            import numpy as np
            from PIL import Image
            
            # Create a small test image
            test_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            
            # Test embedding (this will fail if model isn't ready)
            test_emb = embed_image(test_img, model_id=model_id, device=device, dtype=dtype, normalize=False)
            print(f"[DEBUG] embedder_service: Model test successful - embedding shape: {test_emb.shape}")
        except Exception as e:
            print(f"[DEBUG] embedder_service: Model test failed: {e}")
            # Don't fail the preload, but warn
            print(f"[DEBUG] embedder_service: Warning - Model may not be fully ready")
        
        print(f"[DEBUG] embedder_service: Model preloaded successfully and verified")
        return True
    except Exception as e:
        print(f"[DEBUG] embedder_service: Failed to preload model: {e}")
        return False


def enqueue_embeddings(
    frame_index: int,
    frame_bgr: np.ndarray,
    boxes: List[Tuple[float, float, float, float]],
    labels: List[str],
    scores: List[float],
    embed_queue: Queue,
    score_threshold: float,
    set_status_callback: Callable,
    emit_log_callback: Callable,
    tracks: Optional[List] = None
) -> None:
    """
    Enqueue individual detections for embedding processing.
    
    Args:
        frame_index: Index of the current frame
        frame_bgr: BGR image array from OpenCV
        boxes: List of bounding boxes (x1, y1, x2, y2)
        labels: List of detection labels
        scores: List of detection scores
        embed_queue: Queue to put embedding items into
        score_threshold: Minimum score threshold for detections
        set_status_callback: Callback to update training status
        emit_log_callback: Callback to emit log messages
        tracks: Optional tracking information
    """
    try:
        h, w = frame_bgr.shape[:2]
        
        # Process each detection individually to get correct crops
        for i, b in enumerate(boxes or []):
            # Apply score threshold to filter out low confidence detections
            if scores and i < len(scores) and scores[i] < score_threshold:
                continue
            
            # Extract bounding box coordinates (raw Florence-2 detections only)
            # should be 4-element tuples: (x1, y1, x2, y2)
            if isinstance(b, (list, tuple)) and len(b) >= 4:
                x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
            else:
                # Skip bad boxes
                continue
            
            # Clamp coordinates to image bounds
            xi1 = max(0, min(w - 1, int(x1)))
            yi1 = max(0, min(h - 1, int(y1)))
            xi2 = max(0, min(w - 1, int(x2)))
            yi2 = max(0, min(h - 1, int(y2)))
            
            # Skip invalid boxes
            if xi2 <= xi1 or yi2 <= yi1:
                continue
            
            # Extract individual crop for this detection
            crop = frame_bgr[yi1:yi2, xi1:xi2]
            
            # Get label and score for this detection
            label_txt = (labels[i] if labels and i < len(labels) else "object")
            scr = (scores[i] if scores and i < len(scores) else 1.0)
            
            # Create item for this individual detection
            unique_id = f"{frame_index}_{i}_{int(time.time() * 1000000)}"
            item = {
                "crop": crop, 
                "label": label_txt, 
                "score": float(scr), 
                "bbox": (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))), 
                "track_id": None,  # No tracking for raw detections (we'll add it later... not working correctly yet)
                "frame_index": frame_index,
                "unique_id": unique_id
            }
            
            # Enqueue this individual detection
            try:
                embed_queue.put_nowait(item)
            except Exception as e:
                emit_log_callback(f"Failed to enqueue embedding item: {e}")
                break

        # Update queue stats
        try:
            qe_now = embed_queue.qsize()
            qemax_now = getattr(embed_queue, 'maxsize', 0) or 0
            set_status_callback(queue_embed=qe_now, queue_embed_max=qemax_now)
        except Exception:
            pass
    except Exception:
        pass


__all__ = [
    "load_dino",
    "embed_image",
    "embed_images",
    "is_model_loaded",
    "get_loaded_model_info",
    "preload_model",
    "enqueue_embeddings",
]


