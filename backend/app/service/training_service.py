from __future__ import annotations

import cv2
import asyncio
import threading
import logging
import os
import time
import traceback
import numpy as np
import time
import torch
import json
import uuid
from pathlib import Path
import json
from io import BytesIO
from typing import Any, Dict, Optional
from PIL import Image
from functools import partial

from queue import Queue, Empty, Full
from ..utils.media_utils import make_mosaic_grid, image_to_base64_jpeg, draw_boxes_pil, create_detection_mosaic, split_mosaic_detections
from ..workers.training_workers import embedding_worker, detection_worker, detection_prefetcher
from .detector_service import load_detector_model, is_detector_loaded, get_detector_info, enable_detector_trt, detect_batch, detect_single, parse_detection_output, detect_and_parse, detect_single_and_parse, unload_detector
from ..trackers.sort import Sort as BaseSort
from .system_service import get_configuration

from ..socket.socket_manager import broadcast
from ..database.training_repo import create_training_record, mark_training_status, update_training_record, error_out_running_records
from ..database.milvus import ensure_training_collection, insert_training_embeddings
from ..service.embedder_service import embed_images, is_model_loaded, get_loaded_model_info, preload_model, enqueue_embeddings
from ..storage.minio_client import put_training_image_bytes, ensure_training_bucket

from app.service.embedder_service import preload_model

class _TrainingState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._status: Dict[str, Any] = {
            "state": "idle",
            "progress": 0,
            "message": "",
            "started_at": None,
            "ended_at": None,
        }

    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def _set_status(self, **kwargs: Any) -> None:
        with self._lock:
            self._status.update(kwargs)

    def request_stop(self) -> None:
        logging.getLogger(__name__).info("Terminate requested for training job")
        self._stop_event.set()

    def _should_stop(self) -> bool:
        return self._stop_event.is_set()

    def start_od(self, mosaic: bool = False, extra_args: Optional[Dict[str, Any]] = None) -> bool:
        if self.is_running():
            return False
        logging.getLogger(__name__).info("Starting training job (mosaic=%s, extra=%s)", mosaic, extra_args)
        try:
            # Ensure no stale running records are present
            num_err = error_out_running_records("Superseded by new training run")
            if num_err:
                logging.getLogger(__name__).info("Marked %d stale running training records as error", num_err)
        except Exception:
            pass
        self._stop_event.clear()
        self._set_status(state="running", progress=0, message="Training started", started_at=time.time(), ended_at=None)
        try:
            asyncio.run(broadcast("training_status", self.status()))
        except RuntimeError:
            pass
        
        # Preload dino
        try:
            if not is_model_loaded():
                logging.getLogger(__name__).info("Preloading embedding model...")

                # Get HF token and model from configuration
                cfg = get_configuration()
                hf_token = cfg.get("hf_token") or None
                dinov3_model = cfg.get("dinov3_model", "facebook/dinov3-vit7b16-pretrain-lvd1689m")
                preload_success = preload_model(model_id=dinov3_model, hf_token=hf_token)
                if preload_success:
                    logging.getLogger(__name__).info("Embedding model preloaded successfully")
                else:
                    logging.getLogger(__name__).warning("Failed to preload embedding model")
            else:
                logging.getLogger(__name__).info("Embedding model already loaded")
        except Exception as e:
            logging.getLogger(__name__).warning(f"Error preloading embedding model: {e}")

        def _runner() -> None:
            try:
                def emit_log(message: str) -> None:
                    logging.getLogger(__name__).info(message)
                    try:
                        asyncio.run(broadcast("training_log", {"message": message}))
                    except RuntimeError:
                        pass
                
                
                input_path = (extra_args or {}).get("input") if extra_args else None

                # build absolute path
                cfg = get_configuration()
                library_root = cfg.get("library_path") or ""
                resolved_path: Optional[Path] = None
                if input_path:
                    try:
                        base = Path(library_root).expanduser().resolve() if library_root else None
                        candidate = Path(input_path)
                        if base is not None:
                            # Enforce relative within library root
                            candidate_resolved = (base / candidate.as_posix().lstrip("/\\")).resolve()
                            if str(candidate_resolved).startswith(str(base)):
                                resolved_path = candidate_resolved
                            else:
                                logging.getLogger(__name__).warning("Path is outside of library root: %s", candidate_resolved)
                        else:
                            resolved_path = candidate.resolve()
                    except Exception:
                        resolved_path = None
                emit_log(f"Received training request: input_path={input_path} resolved={resolved_path} mosaic={mosaic}")
                imgs: list[Image.Image] = []
                # if resolved_path is not None:
                #     p = resolved_path
                #     if p.is_dir():
                #         for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
                #             for fp in p.glob(ext):
                #                 try:
                #                     imgs.append(Image.open(fp).convert("RGB"))
                #                 except Exception:
                #                     pass
                #     elif p.is_file():
                #         # If it's a common video, sample frames; else try as image
                #         try:
                #             suffix = p.suffix.lower()
                #             if suffix in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
                #                 import cv2
                #                 cap = cv2.VideoCapture(str(p))
                #                 if not cap.isOpened():
                #                     raise RuntimeError(f"Failed to open video: {p}")
                #                 frame_stride = int((extra_args or {}).get("frame_stride", 10))
                #                 max_frames = int((extra_args or {}).get("max_frames", 12))
                #                 read_idx = 0
                #                 added = 0
                #                 while added < max_frames:
                #                     ok, frame_bgr = cap.read()
                #                     if not ok:
                #                         break
                #                     if (read_idx % max_frames) % frame_stride != 0:
                #                         read_idx += 1
                #                         continue
                #                     imgs.append(Image.fromarray(frame_bgr[:, :, ::-1]))
                #                     added += 1
                #                     read_idx += 1
                #                 cap.release()
                #                 logging.getLogger(__name__).info("Sampled %d frame(s) from video", added)
                #             else:
                #                 imgs.append(Image.open(p).convert("RGB"))
                #         except Exception:
                #             logging.getLogger(__name__).warning("Failed to open input as image/video: %s; falling back to blank image", p)

                # if not imgs:
                #     # Fallback to a blank image if nothing provided
                #     imgs = [Image.new("RGB", (640, 480), (30, 30, 30))]
                # self._set_status(message=f"Found {len(imgs)} image(s)")
                # emit_log(f"Found {len(imgs)} image(s) for processing")
                try:
                    asyncio.run(broadcast("training_status", self.status()))
                except RuntimeError:
                    pass

                # config stuff
                config = get_configuration()
                dtype_mode = config.get("dtype_mode")
                hf_token = config.get("hf_token") or None
                florence_model = config.get("florence_model", "microsoft/Florence-2-large")
                dinov3_model = config.get("dinov3_model", "facebook/dinov3-vitb16-pretrain-lvd1689m")
                dinov3_dimension = config.get("dinov3_dimension", 768)
                
                # Prepare Florence-2
                self._set_status(message=f"Loading {florence_model}...")
                emit_log(f"Loading model {florence_model} ...")
                try:
                    asyncio.run(broadcast("training_status", self.status()))
                except RuntimeError:
                    pass
                
                processor, model, device = load_detector_model(model_id=florence_model, dtype_mode=dtype_mode, hf_token=hf_token)
                emit_log(f"Model {florence_model} loaded on device {device}")
                self._set_status(message=f"{florence_model} loaded")
                try:
                    asyncio.run(broadcast("training_status", self.status()))
                except RuntimeError:
                    pass


                # video or image?
                if resolved_path is not None and resolved_path.is_file() and resolved_path.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
                    # Video pipeline closely following florence2 structure
                    cap = cv2.VideoCapture(str(resolved_path))
                    if not cap.isOpened():
                        raise RuntimeError(f"Failed to open video: {resolved_path}")
                    emit_log(f"Opened video {resolved_path}")
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    # config defaults
                    cfg_vals = get_configuration()
                    tp = dict(cfg_vals.get("training_params", {}))

                    if not tp:
                        tp = {k: cfg_vals.get(k) for k in (
                            "resize_width","batch_size","frame_stride","detect_every","max_new_tokens","mosaic_cols","mosaic_tile_scale","detection_debug","tf32","dtype_mode","score_threshold","interpolate_boxes","embedding_batch_size"
                        ) if cfg_vals.get(k) is not None}

                    resize_width = int((extra_args or {}).get("resize_width", tp.get("resize_width", 0)))
                    target_w = resize_width if resize_width > 0 else w
                    target_h = int(round(h * (target_w / float(w)))) if target_w != w else h

                    batch_size = max(1, int((extra_args or {}).get("batch_size", tp.get("batch_size", 1))))
                    frame_stride = max(1, int((extra_args or {}).get("frame_stride", tp.get("frame_stride", 1))))
                    detect_every = max(1, int((extra_args or {}).get("detect_every", tp.get("detect_every", 10))))
                    max_new_tokens = int((extra_args or {}).get("max_new_tokens", tp.get("max_new_tokens", 256)))
                    mosaic_cols_cfg = int(tp.get("mosaic_cols", 3))
                    
                    # Get task type and phrase for Florence-2 prompts
                    task_type = (extra_args or {}).get("task_type", "<OD>")
                    phrase = (extra_args or {}).get("phrase", "")
                    
                    # Build the prompt based on task type and phrase
                    if task_type == "<CAPTION_TO_PHRASE_GROUNDING>" and phrase:
                        florence_prompt = f"{task_type} {phrase}"
                    else:
                        florence_prompt = task_type
                    
                    # Log the task type and phrase being used
                    emit_log(f"Florence-2 Task: {task_type}")
                    if phrase:
                        emit_log(f"Grounding Phrase: '{phrase}'")
                    emit_log(f"Florence-2 Prompt: '{florence_prompt}'")
                    
                    # Log training start with task details
                    if task_type == "<CAPTION_TO_PHRASE_GROUNDING>" and phrase:
                        emit_log(f"Starting Phrase Grounding Detection training with phrase: '{phrase}'")
                    elif task_type == "<DENSE_REGION_CAPTION>":
                        emit_log(f"Starting Dense Region Caption training")
                    else:
                        emit_log(f"Starting Object Detection training with task: {task_type}")
                    default_scale_cfg = 0.5 if mosaic_cols_cfg == 2 else (1.0 / max(1, mosaic_cols_cfg))
                    mosaic_tile_scale_cfg = float(tp.get("mosaic_tile_scale", default_scale_cfg))
                    detection_debug = bool(tp.get("detection_debug", True))
                    tf32 = bool(tp.get("tf32", True))
                    dtype_mode = tp.get("dtype_mode", 'float32')
                    score_threshold = float(tp.get("score_threshold", 0.15))
                    interpolate_boxes = bool(tp.get("interpolate_boxes", True))
                    embedding_batch_size = max(1, int(tp.get("embedding_batch_size", 32)))

                    # Emit effective configuration for this run
                    effective_cfg = {
                        "resize_width": resize_width,
                        "batch_size": batch_size,
                        "frame_stride": frame_stride,
                        "detect_every": detect_every,
                        "max_new_tokens": max_new_tokens,
                        "mosaic_cols": mosaic_cols_cfg,
                        "mosaic_tile_scale": mosaic_tile_scale_cfg,
                        "detection_debug": detection_debug,
                        "tf32": tf32,
                        "score_threshold": score_threshold,
                        "interpolate_boxes": interpolate_boxes,
                        "embedding_batch_size": embedding_batch_size,
                    }
                    emit_log(f"Training config: {json.dumps(effective_cfg)}")

                    # Save training run metadata for results page
                    try:
                        run_started_ts = time.time()
                        training_doc_id = create_training_record(
                            file_path=str(resolved_path),
                            file_name=resolved_path.name,
                            fps=float(fps) if fps else None,
                            width=int(target_w),
                            height=int(target_h),
                            total_frames=int(total),
                            training_params={
                                "batch_size": batch_size,
                                "detect_every": detect_every,
                                "frame_stride": frame_stride,
                                "max_new_tokens": max_new_tokens,
                                "mosaic_cols": mosaic_cols_cfg,
                                "mosaic_tile_scale": mosaic_tile_scale_cfg,
                                "resize_width": resize_width,
                                "detection_debug": detection_debug,
                                "tf32": tf32,
                                "score_threshold": score_threshold,
                                "interpolate_boxes": interpolate_boxes,
                                "dtype_mode": dtype_mode,
                                "embedding_batch_size": embedding_batch_size,
                                "task_type": task_type,
                                "phrase": phrase,
                                "florence_prompt": florence_prompt,
                                "florence_model": florence_model,
                                "dinov3_model": dinov3_model,
                                "dinov3_dimension": dinov3_dimension,
                            },
                        )
                        
                        # UI updates
                        try:
                            self._set_status(
                                training_id=str(training_doc_id),
                                file_name=resolved_path.name,
                                started_at=run_started_ts,
                                total_frames=int(total),
                                frames_processed=0,
                                elapsed_s=0.0,
                            )

                            asyncio.run(broadcast("training_status", self.status()))
                        except Exception:
                            pass
                    except Exception as e:
                        training_doc_id = None
                        emit_log(f"Warning: failed to create training doc: {e}")

                    # Init Milvus collection
                    try:
                        if training_doc_id is not None:
                            ensure_training_collection(str(training_doc_id), dim=dinov3_dimension)
                            emit_log(f"Milvus: init training collection training_{training_doc_id} (dim {dinov3_dimension})")
                    except Exception as _ex:
                        emit_log(f"Milvus: init failed {_ex}")
                    detect_queue: Queue[tuple[int, Image.Image]] = Queue(maxsize=1024)

                    # Embedding queue for DinoV3                    
                    embed_queue: Queue[dict] = Queue(maxsize=2048) # TODO: add to config
                    embed_stop = threading.Event()
                    
                    # Update UI on embed queue
                    try:
                        self._set_status(queue_embed=embed_queue.qsize(), queue_embed_max=getattr(embed_queue, 'maxsize', 0) or 0)
                        asyncio.run(broadcast("training_status", self.status()))
                    except Exception:
                        pass
                    results_map: dict[int, tuple[list, list, list]] = {}
                    worker_stop = threading.Event()
                    eof_event = threading.Event()  # end-of-video signal. we need this to flush partial mosaics

                    


                    # Create worker function to handle embeddings
                    _embed_worker = partial(
                        embedding_worker,
                        embed_queue=embed_queue,
                        embed_stop=embed_stop,
                        training_doc_id=training_doc_id,
                        embedding_batch_size=embedding_batch_size,
                        set_status_callback=self._set_status,
                        emit_log_callback=emit_log
                    )

                    # Attempt to enable/build TRT engine using ONLY the first frame to derive Florence pixel size
                    try:
                        trt_cache_dir = Path.home() / ".florence_trt_cache"
                        trt_cache_dir.mkdir(parents=True, exist_ok=True)
                        if device.startswith("cuda"):

                            pos_backup = cap.get(1)  # CAP_PROP_POS_FRAMES
                            ok_probe, probe_bgr = cap.read()
                            if ok_probe:
                                if (target_w, target_h) != (w, h):
                                    probe_bgr = cv2.resize(probe_bgr, (target_w, target_h))

                                probe_pil = Image.fromarray(probe_bgr[:, :, ::-1])
                                _pv = processor(text="", images=probe_pil, return_tensors="pt")

                                vision_h, vision_w = int(_pv["pixel_values"].shape[-2]), int(_pv["pixel_values"].shape[-1])
                                emit_log(f"Using TRT vision size {vision_h}x{vision_w}")

                                onnx_path = (trt_cache_dir / f"vision_tower_{vision_h}x{vision_w}.onnx")
                                if not onnx_path.exists():
                                    self._set_status(message=f"Preparing vision tower TRT ONNX Runtime {vision_h}x{vision_w}... This may take a while")
                                    emit_log(f"Preparing ONNX for TRT vision tower {vision_h}x{vision_w} ...")
                                    try:
                                        asyncio.run(broadcast("training_status", self.status()))
                                    except RuntimeError:
                                        pass

                                ok_trt = enable_detector_trt(vision_h=vision_h, vision_w=vision_w, cache_dir=trt_cache_dir)
                                emit_log(f"TRT vision enabled: {ok_trt}")
                                if ok_trt:
                                    self._set_status(message="TensorRT vision enabled")
                                    try:
                                        asyncio.run(broadcast("training_status", self.status()))
                                    except RuntimeError:
                                        pass

                            # Reset back to frame 0 to avoid skipping content
                            try:
                                cap.set(1, 0)
                            except Exception:
                                pass

                    except Exception as e:
                        emit_log(f"TRT setup skipped/failed: {e}")

                    # Enable TF32 if requested and running on CUDA
                    try:
                        if tf32 and str(device).startswith("cuda"):
                            
                            torch.backends.cudnn.benchmark = True
                            try:
                                torch.backends.cuda.matmul.allow_tf32 = True
                            except Exception:
                                pass
                            try:
                                torch.set_float32_matmul_precision("high")
                            except Exception:
                                pass
                            emit_log("TF32 enabled for CUDA matmul")
                    except Exception as e:
                        emit_log(f"TF32 enablement failed or unavailable: {e}")

                            # Start detection worker thread
                    _det_worker = partial(
                        detection_worker,
                        detect_queue=detect_queue,
                        results_map=results_map,
                        worker_stop=worker_stop,
                        eof_event=eof_event,
                        florence_prompt=florence_prompt,
                        max_new_tokens=max_new_tokens,
                        detection_debug=detection_debug,
                        should_stop_callback=self._should_stop,
                        emit_log_callback=emit_log
                    )

                    # Preload DINOv3 embedder before starting workers - BLOCKING
                    self._set_status(message="Preparing DINOv3 embedder...")
                    emit_log("Preparing DINOv3 embedder (downloading shards)...")

                    try:
                        asyncio.run(broadcast("training_status", self.status()))
                    except RuntimeError:
                        pass
                    
                        try:
                            
                            
                            # Update status to show downloading phase
                            self._set_status(message="Downloading DINOv3 model files...")
                            emit_log(f"Downloading DINOv3 model files and checkpoint shards for {dinov3_model}...")
                            try:
                                asyncio.run(broadcast("training_status", self.status()))
                            except RuntimeError:
                                pass
                            
                            # CRITICAL: This must complete fully before workers start
                            emit_log(f"DINOv3 preload starting for {dinov3_model} - this will block until complete...")
                            dino_success = preload_model(
                                model_id=dinov3_model,
                                device=device,
                                dtype=dtype_mode,
                                hf_token=hf_token
                            )
                            
                            if dino_success:
                                emit_log(f"DINOv3 embedder {dinov3_model} loaded successfully - ready for workers")
                                self._set_status(message="DINOv3 embedder loaded")
                            else:
                                emit_log("DINOv3 embedder preload failed, will load on-demand")
                                self._set_status(message="DINOv3 embedder preload failed")
                        except Exception as e:
                            emit_log(f"DINOv3 embedder preload error: {e}")
                            self._set_status(message="DINOv3 embedder preload error")
                            
                        # Final status update before workers start
                        try:
                            asyncio.run(broadcast("training_status", self.status()))
                        except RuntimeError:
                            pass
                            
                    except Exception as e:
                        emit_log(f"DINOv3 embedder preload error: {e}")
                        self._set_status(message="DINOv3 embedder preload error")
                        # Still try to start workers, they'll load on-demand

                    worker_thread = threading.Thread(target=_det_worker, name="od-worker", daemon=True)
                    embed_thread = threading.Thread(target=_embed_worker, name="embed-worker", daemon=True)
                    worker_thread.start()
                    embed_thread.start()

                    # Prefetch detection frames by seeking directly to indices: 0, detect_every, 2*detect_every, ...
                    prefetch_thread = None
                    if total > 0:
                        # Create prefetcher
                        _det_prefetcher = partial(
                            detection_prefetcher,
                            video_path=str(resolved_path),
                            detect_queue=detect_queue,
                            worker_stop=worker_stop,
                            total_frames=total,
                            detect_every=detect_every,
                            target_size=(target_w, target_h),
                            original_size=(w, h),
                            emit_log_callback=emit_log
                        )

                        prefetch_thread = threading.Thread(target=_det_prefetcher, name="od-prefetch", daemon=True)
                        prefetch_thread.start()

                    try:
                        buffer_frames: dict[int, tuple] = {}
                        next_emit = 0
                        frame_index = 0
                        prev_gray_for_flow = None
                        last_boxes_for_flow: list | None = None
                        last_scores_for_flow: list | None = None
                        last_base_labels_for_flow: list | None = None
                        # Also persist raw detection boxes for interpolation
                        last_raw_boxes_for_flow: list | None = None
                        last_raw_labels_for_flow: list | None = None
                        last_raw_scores_for_flow: list | None = None
                        track_id_to_label: dict[int, str] = {}
                        # SORT tracker for ID assignment
                        try:
                            tracker = BaseSort(max_age=max(10, int(detect_every)), min_hits=1, iou_threshold=0.2)
                        except Exception:
                            tracker = BaseSort()
                        def _iou(bb1, bb2):
                            xA = max(bb1[0], bb2[0]); yA = max(bb1[1], bb2[1])
                            xB = min(bb1[2], bb2[2]); yB = min(bb1[3], bb2[3])
                            inter = max(0.0, xB - xA) * max(0.0, yB - yA)
                            area1 = max(0.0, bb1[2] - bb1[0]) * max(0.0, bb1[3] - bb1[1])
                            area2 = max(0.0, bb2[2] - bb2[0]) * max(0.0, bb2[3] - bb2[1])
                            union = area1 + area2 - inter
                            return (inter / union) if union > 0 else 0.0
                        def _update_track_labels(tracks_arr, det_boxes, det_labels):
                            try:
                                if tracks_arr is None or len(tracks_arr) == 0:
                                    return
                                if not det_boxes:
                                    return
                                for tr in tracks_arr:
                                    tx1, ty1, tx2, ty2, tid = tr
                                    # find det with max IoU
                                    best_iou = 0.0; best_idx = -1
                                    for j, db in enumerate(det_boxes):
                                        iou = _iou((tx1, ty1, tx2, ty2), db)
                                        if iou > best_iou:
                                            best_iou = iou; best_idx = j
                                    if best_idx >= 0 and best_iou >= 0.1:
                                        track_id_to_label[int(tid)] = str(det_labels[best_idx] if best_idx < len(det_labels) else 'object')
                            except Exception:
                                pass
                        # FPS tracking
                        fps_t0 = time.time()
                        fps_count = 0
                        fps_current = 0.0

                        def _lk_propagate(prev_gray, curr_gray, boxes):
                            if prev_gray is None or curr_gray is None or not boxes:
                                return boxes
                            lk_params = dict(winSize=(15, 15), maxLevel=2,
                                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                            def _sample_points(bb, step=8):
                                x1, y1, x2, y2 = bb
                                pts = []
                                ys = range(int(y1), int(y2), step)
                                xs = range(int(x1), int(x2), step)
                                for yy in ys:
                                    for xx in xs:
                                        pts.append([xx + 0.5, yy + 0.5])
                                if not pts:
                                    cx = 0.5 * (x1 + x2)
                                    cy = 0.5 * (y1 + y2)
                                    pts = [[cx, cy]]
                                return np.array(pts, dtype=np.float32)
                            new_boxes = []
                            h, w = curr_gray.shape[:2]
                            for bb in boxes:
                                pts = _lk_propagate._sample(bb) if hasattr(_lk_propagate, '_sample') else _sample_points(bb, step=max(4, int((bb[3]-bb[1]) / 6)))
                                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts, None, **lk_params)
                                if p1 is None or st is None:
                                    new_boxes.append(bb)
                                    continue
                                good_new = p1[st.flatten() == 1]
                                good_old = pts[st.flatten() == 1]
                                if good_new is None or len(good_new) == 0:
                                    new_boxes.append(bb)
                                    continue
                                shift = (good_new - good_old)
                                dx = float(np.median(shift[:, 0]))
                                dy = float(np.median(shift[:, 1]))
                                x1, y1, x2, y2 = bb
                                nx1 = max(0, min(w - 1, x1 + dx))
                                ny1 = max(0, min(h - 1, y1 + dy))
                                nx2 = max(0, min(w - 1, x2 + dx))
                                ny2 = max(0, min(h - 1, y2 + dy))
                                new_boxes.append((nx1, ny1, nx2, ny2))
                            return new_boxes
                        processed_total = 0
                        last_progress_update_ts = time.time()
                        while True:
                            if self._should_stop():
                                emit_log("Terminate received; purging queues and stopping video loop")
                                # Purge buffers and queues for immediate stop
                                try:
                                    buffer_frames.clear()
                                except Exception:
                                    pass
                                try:
                                    results_map.clear()
                                except Exception:
                                    pass
                                try:
                                    while True:
                                        try:
                                            _ = detect_queue.get_nowait()
                                            detect_queue.task_done()
                                        except Empty:
                                            break
                                except Exception:
                                    pass
                                worker_stop.set()
                                break
                            ok, frame_bgr = cap.read()
                            if not ok:
                                # Signal EOF to detection worker so it can flush partial mosaics
                                try:
                                    eof_event.set()
                                except Exception:
                                    pass
                                # Flush remaining frames in order
                                while next_emit in buffer_frames or any(k >= next_emit for k in results_map.keys()):
                                    if next_emit not in buffer_frames:
                                        # If waiting for a detection result that's pending, allow brief wait
                                        wait_start = time.time()
                                        while next_emit not in buffer_frames and any(k >= next_emit for k in results_map.keys()) and (time.time() - wait_start) < 0.2:
                                            time.sleep(0.01)
                                        if next_emit not in buffer_frames:
                                            break
                                    fb, need_detect = buffer_frames[next_emit]
                                    if need_detect and next_emit not in results_map:
                                        if self._should_stop():
                                            # On terminate, don't wait for detector; emit without boxes
                                            curr_gray = cv2.cvtColor(fb, cv2.COLOR_BGR2GRAY)
                                            bxs, lbls, scs = ([], [], [])
                                        else:
                                            # If EOF, don't block forever; give detector a short grace period then emit
                                            wait_start = time.time()
                                            while next_emit not in results_map and (time.time() - wait_start) < 0.5:
                                                time.sleep(0.01)
                                            if next_emit in results_map:
                                                pass
                                            else:
                                                bxs, lbls, scs = ([], [], [])
                                                curr_gray = cv2.cvtColor(fb, cv2.COLOR_BGR2GRAY)
                                    curr_gray = cv2.cvtColor(fb, cv2.COLOR_BGR2GRAY)
                                    if need_detect:
                                        if next_emit in results_map:
                                            bxs, lbls, scs = results_map.pop(next_emit)
                                        else:
                                            bxs, lbls, scs = ([], [], [])
                                    else:
                                        if last_boxes_for_flow is not None and prev_gray_for_flow is not None:
                                            bxs = _lk_propagate(prev_gray_for_flow, curr_gray, last_boxes_for_flow)
                                            scs = last_scores_for_flow or [1.0] * len(bxs)
                                            base_lbls = last_base_labels_for_flow or ["object"] * len(bxs)
                                            lbls = base_lbls
                                        else:
                                            bxs, lbls, scs = ([], [], [])
                                    # Update SORT on detection frames and draw tracked boxes
                                    try:
                                        dets_np = np.empty((0,5), dtype=np.float32)
                                        if bxs:
                                            dets_np = np.hstack([
                                                np.array(bxs, dtype=np.float32),
                                                np.array(scs if scs else [1.0]*len(bxs), dtype=np.float32).reshape(-1,1)
                                            ])
                                        tracks = tracker.update(dets_np)
                                        _update_track_labels(tracks, bxs or [], lbls or [])
                                        # Store raw Florence-2 detections only on detection frames
                                        if bxs and len(bxs) > 0:  # Only on detection frames with detections
                                            try:
                                                enqueue_embeddings(
                                                    frame_index=next_emit,
                                                    frame_bgr=fb,
                                                    boxes=bxs,
                                                    labels=lbls or [],
                                                    scores=scs or [],
                                                    embed_queue=embed_queue,
                                                    score_threshold=score_threshold,
                                                    set_status_callback=self._set_status,
                                                    emit_log_callback=emit_log,
                                                    tracks=None
                                                )
                                            except Exception:
                                                pass
                                        
                                        # Draw tracked boxes for preview (but don't store them as embeddings)
                                        if tracks is not None and len(tracks) > 0:
                                            draw_boxes_all = [(float(t[0]), float(t[1]), float(t[2]), float(t[3]), int(t[4])) for t in tracks]
                                            draw_labels_all = [f"{track_id_to_label.get(int(t[4]), 'object')} #{int(t[4])}" for t in tracks]
                                            draw_scores_all = [1.0] * len(draw_boxes_all)
                                        else:
                                            draw_boxes_all, draw_labels_all = (bxs or []), (lbls or [])
                                            draw_scores_all = [1.0] * len(draw_boxes_all)
                                        self._emit_frame_bgr(next_emit, fb, draw_boxes_all, draw_labels_all, draw_scores_all, score_threshold)
                                    except Exception:
                                        self._emit_frame_bgr(next_emit, fb, bxs or [], lbls or [], [1.0] * (len(bxs) if bxs else 0), score_threshold)
                                    processed_total += 1
                                    # Periodically persist frames_processed
                                    try:
                                        if 'training_doc_id' in locals() and training_doc_id is not None and (time.time() - last_progress_update_ts) >= 5.0:
                                            update_training_record(training_doc_id, set_fields={
                                                "frames_processed": int(processed_total),
                                                "elapsed_s": float(time.time() - run_started_ts),
                                            })
                                            last_progress_update_ts = time.time()
                                    except Exception:
                                        pass
                                    # Update FPS
                                    fps_count += 1
                                    now = time.time()
                                    if (now - fps_t0) >= 1.0:
                                        fps_current = fps_count / (now - fps_t0)
                                        fps_t0 = now
                                        fps_count = 0
                                    prev_gray_for_flow = curr_gray
                                    last_boxes_for_flow = bxs
                                    last_scores_for_flow = scs
                                    last_base_labels_for_flow = lbls if lbls else None
                                    # Update raw baseline as well
                                    last_raw_boxes_for_flow = bxs or []
                                    last_raw_labels_for_flow = lbls or []
                                    last_raw_scores_for_flow = scs or []
                                    pct = int(min(100, (next_emit / total) * 100)) if total > 0 else next_emit
                                    # Include queue sizes and fps in status
                                    try:
                                        qd = detect_queue.qsize()
                                        qmax = getattr(detect_queue, 'maxsize', 0) or 0
                                    except Exception:
                                        qd = 0
                                        qmax = 0
                                    try:
                                        qe = embed_queue.qsize()
                                        qemax = getattr(embed_queue, 'maxsize', 0) or 0
                                    except Exception:
                                        qe = 0
                                        qemax = 0
                                    qr = len(results_map)
                                    self._set_status(
                                        progress=pct,
                                        message=f"Processing frame {next_emit}{'/' + str(total) if total>0 else ''}",
                                        queue_detect=qd,
                                        queue_detect_max=qmax,
                                        queue_results=qr,
                                        queue_embed=qe,
                                        queue_embed_max=qemax,
                                        fps=round(fps_current, 2),
                                        frames_processed=int(processed_total),
                                        total_frames=int(total),
                                        elapsed_s=float(time.time() - run_started_ts),
                                    )
                                    try:
                                        asyncio.run(broadcast("training_status", self.status()))
                                    except RuntimeError:
                                        pass
                                    del buffer_frames[next_emit]
                                    next_emit += 1
                                break
                            if frame_stride > 1 and ((frame_index) % frame_stride != 0):
                                frame_index += 1
                                continue
                            if (target_w, target_h) != (w, h):
                                frame_bgr = cv2.resize(frame_bgr, (target_w, target_h))
                            need_detect = ((frame_index) % detect_every) == 0
                            # If prefetcher is active (known total), it will enqueue detection frames.
                            # Fallback: if total is unknown, enqueue on-the-fly here.
                            if need_detect and total <= 0:
                                try:
                                    pil = Image.fromarray(frame_bgr[:, :, ::-1])
                                    detect_queue.put_nowait((frame_index, pil))
                                except Exception:
                                    pass
                            buffer_frames[frame_index] = (frame_bgr, need_detect)
                            while next_emit in buffer_frames:
                                fb, nd = buffer_frames[next_emit]
                                if nd and next_emit not in results_map:
                                    if self._should_stop():
                                        # On terminate, proceed without waiting for detector
                                        pass
                                    else:
                                        break
                                curr_gray = cv2.cvtColor(fb, cv2.COLOR_BGR2GRAY)
                                if nd:
                                    bxs, lbls, scs = results_map.pop(next_emit, ([], [], []))
                                    # Update raw baseline for interpolation on subsequent frames
                                    last_raw_boxes_for_flow = bxs or []
                                    last_raw_labels_for_flow = lbls or []
                                    last_raw_scores_for_flow = scs or []
                                else:
                                    if interpolate_boxes and last_boxes_for_flow is not None and prev_gray_for_flow is not None:
                                        bxs = _lk_propagate(prev_gray_for_flow, curr_gray, last_boxes_for_flow)
                                        scs = last_scores_for_flow or [1.0] * len(bxs)
                                        base_lbls = last_base_labels_for_flow or ["object"] * len(bxs)
                                        lbls = base_lbls
                                    else:
                                        bxs, lbls, scs = ([], [], [])
                                # Use SORT for tracking boxes
                                try:
                                    if nd:
                                        dets_np = np.empty((0,5), dtype=np.float32)
                                        if bxs:
                                            dets_np = np.hstack([
                                                np.array(bxs, dtype=np.float32),
                                                np.array(scs if scs else [1.0]*len(bxs), dtype=np.float32).reshape(-1,1)
                                            ])
                                        tracks = tracker.update(dets_np)
                                        _update_track_labels(tracks, bxs or [], lbls or [])
                                    else:
                                        # advance tracker without detections
 
                                        tracks = tracker.update(np.empty((0,5), dtype=np.float32))
                                    # Store raw Florence-2 detections only on detection frames
                                    if nd and bxs and len(bxs) > 0:  # Only on detection frames with detections
                                        try:
                                            enqueue_embeddings(
                                                frame_index=next_emit,
                                                frame_bgr=fb,
                                                boxes=bxs,
                                                labels=lbls or [],
                                                scores=scs or [],
                                                embed_queue=embed_queue,
                                                score_threshold=score_threshold,
                                                set_status_callback=self._set_status,
                                                emit_log_callback=emit_log,
                                                tracks=None
                                            )
                                        except Exception:
                                            pass
                                    
                                    # Draw tracked boxes for preview (but don't store them as embeddings)
                                    if tracks is not None and len(tracks) > 0:
                                        # suppress per-frame step debug to reduce noise
                                        tboxes = [(float(t[0]), float(t[1]), float(t[2]), float(t[3]), int(t[4])) for t in tracks]
                                        tlabs = [f"{track_id_to_label.get(int(t[4]), 'object')} #{int(t[4])}" for t in tracks]
                                        tscores = [1.0] * len(tboxes)
                                        self._emit_frame_bgr(next_emit, fb, tboxes, tlabs, tscores, score_threshold)
                                    else:
                                        # suppress per-frame step debug to reduce noise
                                        self._emit_frame_bgr(next_emit, fb, bxs or [], lbls or [], [1.0] * (len(bxs) if bxs else 0), score_threshold)
                                except Exception:
                                    self._emit_frame_bgr(next_emit, fb, bxs or [], lbls or [], [1.0] * (len(bxs) if bxs else 0), score_threshold)
                                processed_total += 1
                                # Periodically persist frames_processed
                                try:
                                    if 'training_doc_id' in locals() and training_doc_id is not None and (time.time() - last_progress_update_ts) >= 5.0:
                                        update_training_record(training_doc_id, set_fields={
                                            "frames_processed": int(processed_total),
                                            "elapsed_s": float(time.time() - run_started_ts),
                                        })
                                        last_progress_update_ts = time.time()
                                except Exception:
                                    pass
                                # Update FPS
                                fps_count += 1
                                now = time.time()
                                if (now - fps_t0) >= 1.0:
                                    fps_current = fps_count / (now - fps_t0)
                                    fps_t0 = now
                                    fps_count = 0
                                prev_gray_for_flow = curr_gray
                                last_boxes_for_flow = bxs
                                last_scores_for_flow = scs
                                last_base_labels_for_flow = lbls if lbls else None
                                pct = int(min(100, (next_emit / total) * 100)) if total > 0 else next_emit
                                try:
                                    qd = detect_queue.qsize()
                                    qmax = getattr(detect_queue, 'maxsize', 0) or 0
                                except Exception:
                                    qd = 0
                                    qmax = 0
                                qr = len(results_map)
                                self._set_status(
                                    progress=pct,
                                    message=f"Processing frame {next_emit}{'/' + str(total) if total>0 else ''}",
                                    queue_detect=qd,
                                    queue_detect_max=qmax,
                                    queue_results=qr,
                                    fps=round(fps_current, 2),
                                    frames_processed=int(processed_total),
                                    total_frames=int(total),
                                    elapsed_s=float(time.time() - run_started_ts),
                                )
                                try:
                                    asyncio.run(broadcast("training_status", self.status()))
                                except RuntimeError:
                                    pass
                                del buffer_frames[next_emit]
                                next_emit += 1
                            if len(buffer_frames) > 300:
                                time.sleep(0.01)
                            frame_index += 1
                    finally:
                        worker_stop.set()
                        # Signal embedding worker to stop but don't set embed_stop yet
                        # We need to wait for the embedding queue to be empty first
                        try:
                            # clear any remaining items
                            while True:
                                try:
                                    _ = detect_queue.get_nowait()
                                    detect_queue.task_done()
                                except Empty:
                                    break
                            detect_queue.join()
                        except Exception:
                            pass
                        cap.release()
                        
                        # Wait for embedding queue to be empty before completing training
                        emit_log("Waiting for embedding queue to finish processing...")
                        embed_wait_start = time.time()
                        embed_wait_timeout = 30.0  # 30 second timeout
                        while not embed_queue.empty() and (time.time() - embed_wait_start) < embed_wait_timeout:
                            try:
                                qsize = embed_queue.qsize()
                                if qsize > 0:
                                    self._set_status(message=f"Waiting for embedding queue to empty ({qsize} items remaining)...")
                                    try:
                                        asyncio.run(broadcast("training_status", self.status()))
                                    except RuntimeError:
                                        pass
                                time.sleep(0.1)
                            except Exception:
                                break
                        
                        if not embed_queue.empty():
                            emit_log(f"Warning: Embedding queue still has {embed_queue.qsize()} items after timeout")
                        else:
                            emit_log("Embedding queue emptied successfully")
                        
                        # Now stop the embedding worker
                        embed_stop.set()
                        try:
                            if embed_thread.is_alive():
                                embed_thread.join(timeout=5)
                        except Exception:
                            pass
                        try:
                            if prefetch_thread is not None and prefetch_thread.is_alive():
                                prefetch_thread.join(timeout=2)
                        except Exception:
                            pass
                else:
                    # Image list or single image path: simple sequential processing with optional mosaic
                    batches: list[list[Image.Image]] = []
                    if mosaic and len(imgs) > 1:
                        m = make_mosaic_grid(imgs[:4], cols=2)
                        batches.append([m])
                    else:
                        batches = [[im] for im in imgs]

                    total_steps = max(1, len(batches))
                    for i, batch in enumerate(batches):
                        if self._should_stop():
                            logging.getLogger(__name__).info("Terminate received; stopping image loop")
                            break
                        self._set_status(message=f"Running OD batch {i+1}/{total_steps}")
                        try:
                            asyncio.run(broadcast("training_status", self.status()))
                        except RuntimeError:
                            pass
                        outs = detect_batch(batch, [florence_prompt], max_new_tokens=96)
                        bxs, lbls, scs = parse_detection_output(outs[0])
                        # Emit preview frame with boxes
                        try:
                            
                            pil_drawn = draw_boxes_pil(batch[0], bxs, lbls, scs, color=(0, 255, 0), score_thr=0.15)
                            frame_b64 = image_to_base64_jpeg(pil_drawn)
                            asyncio.run(broadcast("training_frame", {"index": i, "image": frame_b64}))
                        except Exception:
                            pass
                        pct = int((i + 1) / total_steps * 100)
                        self._set_status(progress=pct, message=f"Florence-2 OD {i+1}/{total_steps}...")
                        try:
                            asyncio.run(broadcast("training_status", self.status()))
                        except RuntimeError:
                            pass
                        time.sleep(0.05)

                # Wait for embedding queue to be empty before completing training (for image processing)
                if 'embed_queue' in locals() and 'embed_stop' in locals():
                    emit_log("Waiting for embedding queue to finish processing...")
                    embed_wait_start = time.time()
                    embed_wait_timeout = 30.0  # 30 second timeout
                    while not embed_queue.empty() and (time.time() - embed_wait_start) < embed_wait_timeout:
                        try:
                            qsize = embed_queue.qsize()
                            if qsize > 0:
                                self._set_status(message=f"Waiting for embedding queue to empty ({qsize} items remaining)...")
                                try:
                                    asyncio.run(broadcast("training_status", self.status()))
                                except RuntimeError:
                                    pass
                            time.sleep(0.1)
                        except Exception:
                            break
                    
                    if not embed_queue.empty():
                        emit_log(f"Warning: Embedding queue still has {embed_queue.qsize()} items after timeout")
                    else:
                        emit_log("Embedding queue emptied successfully")
                    
                    # Stop the embedding worker
                    embed_stop.set()
                    try:
                        if 'embed_thread' in locals() and embed_thread.is_alive():
                            embed_thread.join(timeout=5)
                    except Exception:
                        pass

                # Finalize
                if self._should_stop():
                    self._set_status(
                        state="cancelled",
                        message="Training terminated",
                        ended_at=time.time(),
                        frames_processed=int(locals().get('processed_total', 0)),
                        total_frames=int(locals().get('total', 0)),
                        elapsed_s=float(time.time() - locals().get('run_started_ts', time.time())),
                    )
                    emit_log("Training job terminated by user")
                    try:
                        if 'training_doc_id' in locals() and training_doc_id is not None:
                            mark_training_status(
                                training_doc_id,
                                "terminated",
                                frames_processed=int(processed_total),
                                fps_effective=float(self.status().get("fps", 0.0)),
                            )
                    except Exception as e:
                        emit_log(f"Warning: failed to update training doc on terminate: {e}")
                else:
                    self._set_status(
                        state="completed",
                        message="Training completed",
                        ended_at=time.time(),
                        frames_processed=int(locals().get('processed_total', 0)),
                        total_frames=int(locals().get('total', 0)),
                        elapsed_s=float(time.time() - locals().get('run_started_ts', time.time())),
                    )
                    emit_log("Training job completed successfully")
                    try:
                        if 'training_doc_id' in locals() and training_doc_id is not None:
                            mark_training_status(
                                training_doc_id,
                                "completed",
                                frames_processed=int(processed_total),
                                fps_effective=float(self.status().get("fps", 0.0)),
                            )
                    except Exception as e:
                        emit_log(f"Warning: failed to update training doc on complete: {e}")
                try:

                    asyncio.run(broadcast("training_status", self.status()))
                except RuntimeError:
                    pass
            except Exception:
                err = traceback.format_exc()
                self._set_status(state="failed", message=f"Training failed: {err}", ended_at=time.time())
                logging.getLogger(__name__).exception("Training job failed: %s", err)
                try:
                    if 'training_doc_id' in locals() and training_doc_id is not None:
                        mark_training_status(
                            training_doc_id,
                            "failed",
                            error=err,
                        )
                except Exception:
                    pass
                try:
                    asyncio.run(broadcast("training_log", {"message": f"Training failed: {err}"}))
                except RuntimeError:
                    pass
                try:

                    asyncio.run(broadcast("training_status", self.status()))
                except RuntimeError:
                    pass
            finally:
                with self._lock:
                    self._thread = None

        t = threading.Thread(target=_runner, name="training-od", daemon=True)
        with self._lock:
            self._thread = t
        t.start()
        return True

    # Emit frame for UI preview
    def _emit_frame_bgr(self, frame_index: int, frame_bgr, boxes=None, labels=None, scores=None, score_threshold=0.15):
        try:

            fb = frame_bgr.copy()
            b = boxes or []
            l = labels or []
            s = scores or []
            h, w = fb.shape[:2]
            for i, box in enumerate(b):
                if s and i < len(s) and s[i] < score_threshold:
                    continue
                # Support optional 5th element as track id
                if isinstance(box, (list, tuple)) and len(box) >= 5:
                    x1, y1, x2, y2, tid = box[0], box[1], box[2], box[3], box[4]
                else:
                    x1, y1, x2, y2 = box
                    tid = None
                x1 = int(max(0, min(w - 1, x1)))
                y1 = int(max(0, min(h - 1, y1)))
                x2 = int(max(0, min(w - 1, x2)))
                y2 = int(max(0, min(h - 1, y2)))
                cv2.rectangle(fb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_txt = l[i] if i < len(l) else ""
                if tid is not None:
                    # Append tracker id when available
                    if label_txt:
                        label_txt = f"{label_txt} #{int(tid)}"
                    else:
                        label_txt = f"#{int(tid)}"
                score_val = s[i] if i < len(s) else None
                if label_txt or score_val is not None:
                    text = f"{label_txt}" if score_val is None else f"{label_txt} {score_val:.2f}"
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    y_text = max(0, y1 - 6)
                    cv2.rectangle(fb, (x1, y_text - th - 4), (x1 + tw + 4, y_text), (0, 255, 0), -1)
                    cv2.putText(fb, text, (x1 + 2, y_text - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            pil = Image.fromarray(fb[:, :, ::-1])
            b64 = image_to_base64_jpeg(pil)

            asyncio.run(broadcast("training_frame", {"index": frame_index, "image": b64}))
        except Exception:
            pass

_STATE = _TrainingState()


def start_od_training(mosaic: bool = False, extra_args: Optional[Dict[str, Any]] = None) -> bool:
    return _STATE.start_od(mosaic=mosaic, extra_args=extra_args)


def training_status() -> Dict[str, Any]:
    return _STATE.status()


