from __future__ import annotations

import threading
import logging
import os
import time
import traceback
from pathlib import Path
import json
from typing import Any, Dict, Optional
from PIL import Image
from ..utils.media_utils import make_mosaic_grid, image_to_base64_jpeg
from ..utils.florence_utils import load_florence, run_florence_od_batch, parse_od_output, enable_trt_vision
from ..trackers.florence_sort import FlorenceSort
from .system_service import get_configuration

from ..socket.socket_manager import broadcast


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
        self._stop_event.clear()
        self._set_status(state="running", progress=0, message="Training started", started_at=time.time(), ended_at=None)
        try:
            import asyncio
            asyncio.run(broadcast("training_status", self.status()))
        except RuntimeError:
            pass

        def _runner() -> None:
            try:
                def emit_log(message: str) -> None:
                    logging.getLogger(__name__).info(message)
                    try:
                        import asyncio
                        asyncio.run(broadcast("training_log", {"message": message}))
                    except RuntimeError:
                        pass
                # Minimal in-app OD pipeline stub: load image(s), optionally create mosaic,
                # and simulate detection steps. Replace with real model calls here.
                input_path = (extra_args or {}).get("input") if extra_args else None
                # Resolve path against configured library_path
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
                                logging.getLogger(__name__).warning("Rejected path outside library root: %s", candidate_resolved)
                        else:
                            resolved_path = candidate.resolve()
                    except Exception:
                        resolved_path = None
                emit_log(f"Received training request: input_path={input_path} resolved={resolved_path} mosaic={mosaic}")
                imgs: list[Image.Image] = []
                if resolved_path is not None:
                    p = resolved_path
                    if p.is_dir():
                        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
                            for fp in p.glob(ext):
                                try:
                                    imgs.append(Image.open(fp).convert("RGB"))
                                except Exception:
                                    pass
                    elif p.is_file():
                        # If it's a common video, sample frames; else try as image
                        try:
                            suffix = p.suffix.lower()
                            if suffix in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
                                import cv2
                                cap = cv2.VideoCapture(str(p))
                                if not cap.isOpened():
                                    raise RuntimeError(f"Failed to open video: {p}")
                                frame_stride = int((extra_args or {}).get("frame_stride", 10))
                                max_frames = int((extra_args or {}).get("max_frames", 12))
                                read_idx = 0
                                added = 0
                                while added < max_frames:
                                    ok, frame_bgr = cap.read()
                                    if not ok:
                                        break
                                    if (read_idx % max_frames) % frame_stride != 0:
                                        read_idx += 1
                                        continue
                                    imgs.append(Image.fromarray(frame_bgr[:, :, ::-1]))
                                    added += 1
                                    read_idx += 1
                                cap.release()
                                logging.getLogger(__name__).info("Sampled %d frame(s) from video", added)
                            else:
                                imgs.append(Image.open(p).convert("RGB"))
                        except Exception:
                            logging.getLogger(__name__).warning("Failed to open input as image/video: %s; falling back to blank image", p)

                if not imgs:
                    # Fallback to a blank image if nothing provided
                    imgs = [Image.new("RGB", (640, 480), (30, 30, 30))]
                self._set_status(message=f"Found {len(imgs)} image(s)")
                emit_log(f"Found {len(imgs)} image(s) for processing")
                try:
                    import asyncio
                    asyncio.run(broadcast("training_status", self.status()))
                except RuntimeError:
                    pass

                # Prepare Florence-2
                model_id = (extra_args or {}).get("model") or (extra_args or {}).get("model_id") or "microsoft/Florence-2-large-ft"
                self._set_status(message=f"Loading {model_id}...")
                emit_log(f"Loading model {model_id} ...")
                try:
                    import asyncio
                    asyncio.run(broadcast("training_status", self.status()))
                except RuntimeError:
                    pass

                # Honor dtype_mode from configuration
                cfg_vals_dtype = get_configuration()
                dtype_mode = cfg_vals_dtype.get("dtype_mode")
                processor, model, device = load_florence(model_id=model_id, dtype_mode=dtype_mode)
                emit_log(f"Model {model_id} loaded on device {device}")
                self._set_status(message=f"{model_id} loaded")
                try:
                    import asyncio
                    asyncio.run(broadcast("training_status", self.status()))
                except RuntimeError:
                    pass

                # If user provided a single image, process once; if video, stream frames; if multiple images, batch sequentially
                if resolved_path is not None and resolved_path.is_file() and resolved_path.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
                    # Video pipeline closely following florence2 structure
                    import cv2
                    cap = cv2.VideoCapture(str(resolved_path))
                    if not cap.isOpened():
                        raise RuntimeError(f"Failed to open video: {resolved_path}")
                    emit_log(f"Opened video {resolved_path}")
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    # Pull defaults from configuration if not provided via extra_args
                    cfg_vals = get_configuration()
                    tp = dict(cfg_vals.get("training_params", {}))
                    # Flatten fallback for backward compatibility
                    if not tp:
                        tp = {k: cfg_vals.get(k) for k in (
                            "resize_width","batch_size","frame_stride","detect_every","max_new_tokens","mosaic_cols","mosaic_tile_scale","detection_debug","tf32","dtype_mode","score_threshold","interpolate_boxes"
                        ) if cfg_vals.get(k) is not None}
                    resize_width = int((extra_args or {}).get("resize_width", tp.get("resize_width", 0)))
                    target_w = resize_width if resize_width > 0 else w
                    target_h = int(round(h * (target_w / float(w)))) if target_w != w else h

                    batch_size = max(1, int((extra_args or {}).get("batch_size", tp.get("batch_size", 1))))
                    frame_stride = max(1, int((extra_args or {}).get("frame_stride", tp.get("frame_stride", 1))))
                    detect_every = max(1, int((extra_args or {}).get("detect_every", tp.get("detect_every", 10))))
                    max_new_tokens = int((extra_args or {}).get("max_new_tokens", tp.get("max_new_tokens", 256)))
                    mosaic_cols_cfg = int(tp.get("mosaic_cols", 3))
                    default_scale_cfg = 0.5 if mosaic_cols_cfg == 2 else (1.0 / max(1, mosaic_cols_cfg))
                    mosaic_tile_scale_cfg = float(tp.get("mosaic_tile_scale", default_scale_cfg))
                    detection_debug = bool(tp.get("detection_debug", True))
                    tf32 = bool(tp.get("tf32", True))
                    dtype_mode = tp.get("dtype_mode", 'float32')
                    score_threshold = float(tp.get("score_threshold", 0.15))
                    interpolate_boxes = bool(tp.get("interpolate_boxes", True))

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
                    }
                    emit_log(f"Training config: {json.dumps(effective_cfg)}")

                    from queue import Queue, Empty, Full
                    detect_queue: Queue[tuple[int, Image.Image]] = Queue(maxsize=1024)
                    results_map: dict[int, tuple[list, list, list]] = {}
                    worker_stop = threading.Event()
                    eof_event = threading.Event()  # Signals end-of-video to flush partial mosaics

                    def _emit_frame_bgr(frame_index: int, frame_bgr, boxes=None, labels=None, scores=None):
                        try:
                            import cv2 as _cv2
                            fb = frame_bgr.copy()
                            b = boxes or []
                            l = labels or []
                            s = scores or []
                            h, w = fb.shape[:2]
                            for i, box in enumerate(b):
                                if s and i < len(s) and s[i] < score_threshold:
                                    continue
                                x1, y1, x2, y2 = box
                                x1 = int(max(0, min(w - 1, x1)))
                                y1 = int(max(0, min(h - 1, y1)))
                                x2 = int(max(0, min(w - 1, x2)))
                                y2 = int(max(0, min(h - 1, y2)))
                                _cv2.rectangle(fb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                label_txt = l[i] if i < len(l) else ""
                                score_val = s[i] if i < len(s) else None
                                if label_txt or score_val is not None:
                                    text = f"{label_txt}" if score_val is None else f"{label_txt} {score_val:.2f}"
                                    (tw, th), _ = _cv2.getTextSize(text, _cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                    y_text = max(0, y1 - 6)
                                    _cv2.rectangle(fb, (x1, y_text - th - 4), (x1 + tw + 4, y_text), (0, 255, 0), -1)
                                    _cv2.putText(fb, text, (x1 + 2, y_text - 2), _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, _cv2.LINE_AA)
                            pil = Image.fromarray(fb[:, :, ::-1])
                            b64 = image_to_base64_jpeg(pil)
                            import asyncio
                            asyncio.run(broadcast("training_frame", {"index": frame_index, "image": b64}))
                        except Exception:
                            pass

                    # Attempt to enable/build TRT engine using ONLY the first frame to derive Florence pixel size
                    try:
                        trt_cache_dir = Path.home() / ".florence_trt_cache"
                        trt_cache_dir.mkdir(parents=True, exist_ok=True)
                        if device.startswith("cuda"):
                            # Read a single frame solely to compute vision size
                            pos_backup = cap.get(1)  # CAP_PROP_POS_FRAMES
                            ok_probe, probe_bgr = cap.read()
                            if ok_probe:
                                if (target_w, target_h) != (w, h):
                                    import cv2 as _cv2
                                    probe_bgr = _cv2.resize(probe_bgr, (target_w, target_h))
                                probe_pil = Image.fromarray(probe_bgr[:, :, ::-1])
                                _pv = processor(text="", images=probe_pil, return_tensors="pt")
                                # Use Florence default pixel size (e.g., 768x768)
                                vision_h, vision_w = int(_pv["pixel_values"].shape[-2]), int(_pv["pixel_values"].shape[-1])
                                emit_log(f"Using TRT vision size {vision_h}x{vision_w}")
                                # Only announce build if engine does not already exist
                                engine_path = (trt_cache_dir / f"vision_tower_{vision_h}x{vision_w}.plan")
                                if not engine_path.exists():
                                    self._set_status(message=f"Building TRT engine {vision_h}x{vision_w}... This may take a while")
                                    emit_log(f"Building TRT engine {vision_h}x{vision_w} ...")
                                    try:
                                        import asyncio
                                        asyncio.run(broadcast("training_status", self.status()))
                                    except RuntimeError:
                                        pass
                                ok_trt = enable_trt_vision(
                                    processor,
                                    model,
                                    vision_h,
                                    vision_w,
                                    fp16=True,
                                    opset=17,
                                    cache_dir=str(trt_cache_dir),
                                )
                                emit_log(f"TRT vision enablement: {ok_trt}")
                                if ok_trt:
                                    self._set_status(message="TensorRT vision enabled")
                                    try:
                                        import asyncio
                                        asyncio.run(broadcast("training_status", self.status()))
                                    except RuntimeError:
                                        pass
                            # Reset back to frame 0 to avoid skipping content
                            try:
                                cap.set(1, 0)  # CAP_PROP_POS_FRAMES
                            except Exception:
                                pass
                    except Exception as e:
                        emit_log(f"TRT setup skipped/failed: {e}")

                    # Enable TF32 if requested and running on CUDA
                    try:
                        if tf32 and str(device).startswith("cuda"):
                            import torch
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
                    def _det_worker():
                        try:
                            staged: list[tuple[int, Image.Image]] = []
                            while not worker_stop.is_set() or staged or not detect_queue.empty():
                                # If termination requested, purge staged and queue immediately
                                if self._should_stop():
                                    staged = []
                                    try:
                                        from queue import Empty as _QEmpty
                                        while True:
                                            try:
                                                _ = detect_queue.get_nowait()
                                                detect_queue.task_done()
                                            except _QEmpty:
                                                break
                                    except Exception:
                                        pass
                                    break
                                # If end-of-video, flush any remaining frames without requiring full mosaic
                                if eof_event.is_set():
                                    # Drain any remaining queued items without blocking
                                    try:
                                        while True:
                                            from queue import Empty as _QEmpty2
                                            try:
                                                item = detect_queue.get_nowait()
                                                staged.append(item)
                                                detect_queue.task_done()
                                            except _QEmpty2:
                                                break
                                    except Exception:
                                        pass
                                    if staged:
                                        idxs = [i for (i, _) in staged]
                                        imgs = [im for (_, im) in staged]
                                        outs = run_florence_od_batch(model, processor, imgs, device=device, max_new_tokens=max_new_tokens)
                                        for j, od in enumerate(outs):
                                            bxs, lbls, scs = parse_od_output(od)
                                            results_map[idxs[j]] = (bxs, lbls, scs)
                                        staged = []
                                        continue
                                    else:
                                        # Nothing left to process
                                        break
                                # Mosaic parameters from configuration
                                cfg_vals_worker = get_configuration()
                                cols = max(1, int(cfg_vals_worker.get("mosaic_cols", 3)))
                                cols = min(cols, 5)
                                need = cols * cols
                                # Match script defaults: 2x2 -> 0.5, 3x3 -> 1/3
                                default_scale = 0.5 if cols == 2 else (1.0 / max(1, cols))
                                tile_scale = float(cfg_vals_worker.get("mosaic_tile_scale", default_scale))
                                tile_scale = max(0.1, min(tile_scale, 1.0))
                                # Fill staged up to cols*cols frames
                                while len(staged) < need and not worker_stop.is_set():
                                    try:
                                        item = detect_queue.get(timeout=0.02)
                                        staged.append(item)
                                    except Empty:
                                        break
                                if len(staged) >= need:
                                    group = staged[:need]
                                    staged = staged[need:]
                                    idxs = [i for (i, _) in group]
                                    imgs = [im for (_, im) in group]
                                    # Build mosaic scaled by tile_scale (script parity)
                                    base_w, base_h = imgs[0].width, imgs[0].height
                                    tile_w = max(1, int(base_w * tile_scale))
                                    tile_h = max(1, int(base_h * tile_scale))
                                    from PIL import Image as _PILImage
                                    tiles = [im.resize((tile_w, tile_h), _PILImage.BILINEAR) for im in imgs[:need]]
                                    mosaic = _PILImage.new('RGB', (tile_w * cols, tile_h * cols))
                                    positions = [(c * tile_w, r * tile_h) for r in range(cols) for c in range(cols)]
                                    for pos, timg in zip(positions, tiles):
                                        mosaic.paste(timg, pos)
                                    # Run OD once on mosaic
                                    outs = run_florence_od_batch(model, processor, [mosaic], device=device, max_new_tokens=max_new_tokens)
                                    od = outs[0]
                                    m_boxes, m_labels, m_scores = parse_od_output(od)
                                    if detection_debug:
                                        try:
                                            boxes_dbg = [(round(b[0],1), round(b[1],1), round(b[2],1), round(b[3],1)) for b in m_boxes][:20]
                                            emit_log(f"[DEBUG] mosaic det: batch_idx=0 num_boxes={len(m_boxes)} labels_head={m_labels[:5]} boxes_head={boxes_dbg}")
                                        except Exception:
                                            emit_log(f"[DEBUG] mosaic det: batch_idx=0 num_boxes={len(m_boxes)} (boxes debug unavailable)")
                                    # Split back to per-tile and remap to original frame coordinates
                                    per_boxes = [[] for _ in range(need)]
                                    per_labels = [[] for _ in range(need)]
                                    per_scores = [[] for _ in range(need)]
                                    for b, l, s in zip(m_boxes, m_labels, m_scores):
                                        x1, y1, x2, y2 = b
                                        cx = 0.5 * (x1 + x2)
                                        cy = 0.5 * (y1 + y2)
                                        col = min(cols - 1, int(cx // tile_w))
                                        row = min(cols - 1, int(cy // tile_h))
                                        qi = row * cols + col
                                        ox, oy = positions[qi]
                                        # Scale back to original size: mirror script (1.0 / tile_scale)
                                        sx = 1.0 / float(tile_scale)
                                        sy = 1.0 / float(tile_scale)
                                        rx1 = (x1 - ox) * sx
                                        ry1 = (y1 - oy) * sy
                                        rx2 = (x2 - ox) * sx
                                        ry2 = (y2 - oy) * sy
                                        per_boxes[qi].append((rx1, ry1, rx2, ry2))
                                        per_labels[qi].append(l)
                                        per_scores[qi].append(s)
                                    # Store results for each corresponding frame index
                                    for k in range(need):
                                        results_map[idxs[k]] = (per_boxes[k], per_labels[k], per_scores[k])
                                    # Mark queue tasks done for the 9 frames
                                    for _ in range(need):
                                        detect_queue.task_done()
                                    continue
                                # If terminating and leftovers remain, process them directly
                                if worker_stop.is_set() and staged:
                                    idxs = [i for (i, _) in staged]
                                    imgs = [im for (_, im) in staged]
                                    outs = run_florence_od_batch(model, processor, imgs, device=device, max_new_tokens=max_new_tokens)
                                    for j, od in enumerate(outs):
                                        bxs, lbls, scs = parse_od_output(od)
                                        if detection_debug:
                                            try:
                                                boxes_dbg = [(round(b[0],1), round(b[1],1), round(b[2],1), round(b[3],1)) for b in bxs][:20]
                                                emit_log(f"[DEBUG] leftover det: batch_idx={j} frame_idx={idxs[j]} num_boxes={len(bxs)} boxes_head={boxes_dbg}")
                                            except Exception:
                                                emit_log(f"[DEBUG] leftover det: batch_idx={j} frame_idx={idxs[j]} num_boxes={len(bxs)}")
                                        results_map[idxs[j]] = (bxs, lbls, scs)
                                    for _ in staged:
                                        detect_queue.task_done()
                                    staged = []
                        except Exception as e:
                            emit_log(f"Detection worker stopped: {e}")

                    worker_thread = threading.Thread(target=_det_worker, name="od-worker", daemon=True)
                    worker_thread.start()

                    # Prefetch detection frames by seeking directly to indices: 0, detect_every, 2*detect_every, ...
                    prefetch_thread = None
                    if total > 0:
                        def _det_prefetcher():
                            try:
                                import cv2 as _cv2
                                cap2 = _cv2.VideoCapture(str(resolved_path))
                                if not cap2.isOpened():
                                    emit_log("Prefetch: failed to open video")
                                    return
                                for idx in range(0, total, detect_every):
                                    if worker_stop.is_set():
                                        break
                                    try:
                                        cap2.set(_cv2.CAP_PROP_POS_FRAMES, idx)
                                        ok, fb = cap2.read()
                                        if not ok:
                                            continue
                                        if (target_w, target_h) != (w, h):
                                            fb = _cv2.resize(fb, (target_w, target_h))
                                        pil = Image.fromarray(fb[:, :, ::-1])
                                        # Block if queue is full, but remain responsive to stop
                                        while not worker_stop.is_set():
                                            try:
                                                detect_queue.put((idx, pil), timeout=0.05)
                                                break
                                            except Full:
                                                continue
                                    except Exception as e:
                                        emit_log(f"Prefetch error at frame {idx}: {e}")
                                        continue
                                cap2.release()
                                emit_log("Prefetch: completed")
                            except Exception as e:
                                emit_log(f"Prefetch thread error: {e}")

                        prefetch_thread = threading.Thread(target=_det_prefetcher, name="od-prefetch", daemon=True)
                        prefetch_thread.start()

                    try:
                        buffer_frames: dict[int, tuple] = {}
                        next_emit = 0
                        frame_index = 0
                        import cv2
                        import numpy as np
                        prev_gray_for_flow = None
                        last_boxes_for_flow: list | None = None
                        last_scores_for_flow: list | None = None
                        last_base_labels_for_flow: list | None = None
                        # Also persist raw detection boxes for interpolation
                        last_raw_boxes_for_flow: list | None = None
                        last_raw_labels_for_flow: list | None = None
                        last_raw_scores_for_flow: list | None = None
                        track_id_to_label: dict[int, str] = {}
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
                                    from queue import Empty as _QEmpty
                                    while True:
                                        try:
                                            _ = detect_queue.get_nowait()
                                            detect_queue.task_done()
                                        except _QEmpty:
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
                                    # Draw only raw detections (no tracking)
                                    draw_boxes_all = bxs or []
                                    draw_labels_all = lbls or []
                                    draw_scores_all = scs or []
                                    _emit_frame_bgr(next_emit, fb, draw_boxes_all, draw_labels_all, draw_scores_all)
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
                                    qr = len(results_map)
                                    self._set_status(progress=pct, message=f"Processing frame {next_emit}{'/' + str(total) if total>0 else ''}", queue_detect=qd, queue_detect_max=qmax, queue_results=qr, fps=round(fps_current, 2))
                                    try:
                                        import asyncio
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
                                # Draw only raw detections (no tracking)
                                if nd:
                                    raw_boxes_to_draw = bxs
                                    raw_labels_to_draw = lbls
                                    raw_scores_to_draw = scs
                                else:
                                    if interpolate_boxes and last_raw_boxes_for_flow is not None and prev_gray_for_flow is not None and len(last_raw_boxes_for_flow) > 0:
                                        raw_interp = _lk_propagate(prev_gray_for_flow, curr_gray, last_raw_boxes_for_flow)
                                        raw_boxes_to_draw = raw_interp
                                        raw_labels_to_draw = last_raw_labels_for_flow or []
                                        raw_scores_to_draw = last_raw_scores_for_flow or []
                                    else:
                                        raw_boxes_to_draw = bxs or []
                                        raw_labels_to_draw = lbls or []
                                        raw_scores_to_draw = scs or []
                                _emit_frame_bgr(next_emit, fb, raw_boxes_to_draw, raw_labels_to_draw, raw_scores_to_draw)
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
                                self._set_status(progress=pct, message=f"Processing frame {next_emit}{'/' + str(total) if total>0 else ''}", queue_detect=qd, queue_detect_max=qmax, queue_results=qr, fps=round(fps_current, 2))
                                try:
                                    import asyncio
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
                        try:
                            # Drain any remaining items to prevent join hang
                            from queue import Empty as _QEmpty
                            while True:
                                try:
                                    _ = detect_queue.get_nowait()
                                    detect_queue.task_done()
                                except _QEmpty:
                                    break
                            detect_queue.join()
                        except Exception:
                            pass
                        cap.release()
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
                            import asyncio
                            asyncio.run(broadcast("training_status", self.status()))
                        except RuntimeError:
                            pass
                        outs = run_florence_od_batch(model, processor, batch, device=device, max_new_tokens=96)
                        bxs, lbls, scs = parse_od_output(outs[0])
                        # Emit preview frame with boxes
                        try:
                            from ..utils.media_utils import draw_boxes_pil
                            pil_drawn = draw_boxes_pil(batch[0], bxs, lbls, scs, color=(0, 255, 0), score_thr=0.15)
                            frame_b64 = image_to_base64_jpeg(pil_drawn)
                            import asyncio
                            asyncio.run(broadcast("training_frame", {"index": i, "image": frame_b64}))
                        except Exception:
                            pass
                        pct = int((i + 1) / total_steps * 100)
                        self._set_status(progress=pct, message=f"Florence-2 OD {i+1}/{total_steps}...")
                        try:
                            import asyncio
                            asyncio.run(broadcast("training_status", self.status()))
                        except RuntimeError:
                            pass
                        time.sleep(0.05)

                # Finalize
                if self._should_stop():
                    self._set_status(state="cancelled", message="Training terminated", ended_at=time.time())
                    emit_log("Training job terminated by user")
                else:
                    self._set_status(state="completed", message="Training completed", ended_at=time.time())
                    emit_log("Training job completed successfully")
                try:
                    import asyncio

                    asyncio.run(broadcast("training_status", self.status()))
                except RuntimeError:
                    pass
            except Exception:
                err = traceback.format_exc()
                self._set_status(state="failed", message=f"Training failed: {err}", ended_at=time.time())
                logging.getLogger(__name__).exception("Training job failed: %s", err)
                try:
                    import asyncio
                    asyncio.run(broadcast("training_log", {"message": f"Training failed: {err}"}))
                except RuntimeError:
                    pass
                try:
                    import asyncio

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


_STATE = _TrainingState()


def start_od_training(mosaic: bool = False, extra_args: Optional[Dict[str, Any]] = None) -> bool:
    return _STATE.start_od(mosaic=mosaic, extra_args=extra_args)


def training_status() -> Dict[str, Any]:
    return _STATE.status()


