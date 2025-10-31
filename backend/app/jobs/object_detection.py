from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable, Optional, List, Any, Dict
from datetime import datetime
import asyncio
import time
import os
import logging
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from ultralytics import RTDETR

from ..database.mongo import update_one, get_collection
from ..trackers.sort import Sort
from ..socket.socket_manager import broadcast
from ..service.system_service import get_configuration
from ..utils.media_utils import image_to_base64_jpeg
from .job_interface import Job, update_job_progress

logger = logging.getLogger(__name__)

class ObjectDetection(Job):
    def __init__(self, progress_callback: Callable[[float, float, float], None] | None = None) -> None:
        self._progress_callback = progress_callback
        self._load_model()
        # Configure batch size
        self.batch_size: int = 256

    def set_model_path_and_load(self, path: Path) -> None:
        try:
            if not path.exists() or not path.is_file():
                return
            # Update and (re)load
            self.model_path = path
            self._model = YOLO(str(self.model_path))
            logger.info(f"Loaded model: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model {path}: {e}")
            # Try RTDETR as fallback
            try:
                self._model = RTDETR(str(self.model_path))  # type: ignore
                logger.info(f"Loaded RTDETR model: {self.model_path}")
            except Exception as e2:
                logger.error(f"Failed to load RTDETR model {self.model_path}: {e2}")
                # Leave model as-is on failure
                pass

    def _default_model_loaded(self) -> bool:
        cfg = get_configuration()
        lib_path = cfg.get("library_path", "/")
        models_dir = Path(lib_path) / "model"
        default_name = str(cfg.get("default_model") or "").strip()
        if default_name:
            candidate = (models_dir / default_name)
            if candidate.exists() and candidate.is_file() and candidate.absolute() == self.model_path and self._model is not None:
                return True
            else:
                return False
        return False

    def _load_model(self) -> None:
        cfg = get_configuration()
        lib_path = cfg.get("library_path", "/")
        models_dir = Path(lib_path) / "model"
        self.model_path: Optional[Path] = None
        self._model: Optional[YOLO] = None

        # Preferred root for managed models
        #managed_models_root = Path("/app/media/model")
        # 1) If a default_model is configured, use it under managed root
        try:
            default_name = str(cfg.get("default_model") or "").strip()
            if default_name:
                candidate = (models_dir / default_name)
                if candidate.exists() and candidate.is_file():
                    self.model_path = candidate
                    logger.info(f"Using model: {self.model_path}")
        except Exception:
            pass
        # 2) Fallback: pick most recent file under managed root
        try:
            if self.model_path is None and models_dir.exists() and models_dir.is_dir():
                candidates_mm: List[Path] = [p for p in models_dir.glob("**/*") if p.is_file()]
                if candidates_mm:
                    self.model_path = max(candidates_mm, key=lambda p: p.stat().st_mtime)
                    logger.info(f"Using model: {self.model_path}")
        except Exception:
            pass
        # 3) Legacy fallback: look under library_path/model for most recent file
        try:
            if self.model_path is None:
                models_dir = Path(lib_path) / "model"
                if models_dir.exists() and models_dir.is_dir():
                    candidates_legacy: List[Path] = [p for p in models_dir.iterdir() if p.is_file()]
                    if candidates_legacy:
                        self.model_path = max(candidates_legacy, key=lambda p: p.stat().st_mtime)
                        logger.info(f"Using model: {self.model_path}")
        except Exception:
            pass
        # Fallback: look for common .pt weights in backend root directory
        try:
            if self.model_path is None:
                backend_root = Path(__file__).resolve().parents[2]
                if backend_root.exists():
                    candidates2: List[Path] = [p for p in backend_root.iterdir() if p.is_file() and p.suffix.lower() == ".pt"]
                    if candidates2:
                        self.model_path = max(candidates2, key=lambda p: p.stat().st_mtime)
        except Exception:
            pass
        
        # Actually load the model if we found a path
        if self.model_path is not None:
            try:
                self._model = YOLO(str(self.model_path))
                logger.info(f"Loaded model: {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load model {self.model_path}: {e}")
                # Try RTDETR as fallback
                try:
                    self._model = RTDETR(str(self.model_path))  # type: ignore
                    logger.info(f"Loaded RTDETR model: {self.model_path}")
                except Exception as e2:
                    logger.error(f"Failed to load RTDETR model {self.model_path}: {e2}")
                    self._model = None

    async def process(self, job_id: str, media_path: str) -> None:
        self.job_id = job_id
        
        p = Path(media_path)
        if not p.exists():
            raise FileNotFoundError(str(p))

        logger.info(f"Processing job: {job_id} with media: {media_path}")

        if not self._default_model_loaded():
            self._load_model()
        else:
            logger.info(f"Using model: {self.model_path}")
        
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            # Try as image: make single-frame stream
            img = Image.open(str(p)).convert("RGB")
            frames_rgb: List[Image.Image] = [img]
            total_frames = 1
            frames_processed = 0
            t0 = time.time()
            preview_b64 = None
            # Run "inference" on single frame
            await self._run_inference_batch(frames_rgb)
            frames_processed = 1
            # Generate preview (width 200px)
            prev = img.copy()
            w, h = prev.size
            if w > 0:
                new_w = 200
                new_h = max(1, int(h * (new_w / float(w))))
                prev = prev.resize((new_w, new_h), Image.BILINEAR)
                preview_b64 = image_to_base64_jpeg(prev, quality=75)
            update_job_progress(
                job_id,
                100.0,
                float(frames_processed),
                float(total_frames),
                frames_processed=frames_processed,
                preview_b64=preview_b64,
                emit=False,
            )
            return

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        except Exception:
            total_frames = 0

        frames_processed = 0
        batch: List[Image.Image] = []
        batch_meta: List[Dict[str, Any]] = []  # per-frame: {idx, w, h, ts_ms}
        last_update = time.time()
        start_time = last_update
        preview_b64: Optional[str] = None
        # Frame bucket buffering
        BUCKET = 256
        fb_frames: List[dict] = []
        fb_start_idx: Optional[int] = None
        imgw = 0
        imgh = 0
        block_counts: Dict[str, int] = {}

        try:
            # Initialize SORT tracker
            tracker = Sort(max_age=5, min_hits=0, iou_threshold=0.3)
            track_span: Dict[int, Dict[str, int]] = {}  # track_id -> {from_i, to_i}
            current_frame_index = 0
            last_batch = False
            try:
                vid_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            except Exception:
                vid_fps = 0.0
            # Before processing, ensure we use the model specified in the job document if present
            try:
                coll_jobs = get_collection("jobs")
                job_doc = coll_jobs.find_one({"job_id": job_id}, {"model_path": 1}) or {}
                mp = job_doc.get("model_path")
                if isinstance(mp, str) and mp:
                    mpp = Path(mp)
                    if mpp.exists() and mpp.is_file():
                        # Load if different from current
                        cur = getattr(self, "model_path", None)
                        if str(cur) != str(mpp):
                            self.set_model_path_and_load(mpp)
            except Exception:
                logger.error(f"Failed to get model path for job: {job_id}")
                raise Exception(f"Failed to get model path for job: {job_id}")

            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    # End of stream; handle final partial batch after loop
                    break

                # Convert to PIL RGB
                frame_rgb = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                # Capture image size
                if imgw == 0 or imgh == 0:
                    imgw, imgh = frame_rgb.size
                if preview_b64 is None:
                    # Create 200px-wide preview once we have first frame
                    w, h = frame_rgb.size
                    if w > 0:
                        new_w = 200
                        new_h = max(1, int(h * (new_w / float(w))))
                        prev = frame_rgb.resize((new_w, new_h), Image.BILINEAR)
                        preview_b64 = image_to_base64_jpeg(prev, quality=75)

                batch.append(frame_rgb)
                # Timestamp from fps when available
                try:
                    vid_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                except Exception:
                    vid_fps = 0.0
                cur_idx_peek = frames_processed + len(batch) - 1
                if vid_fps and vid_fps > 0:
                    ts_ms_peek = int((cur_idx_peek / vid_fps) * 1000.0)
                else:
                    ts_ms_peek = int((time.time() - start_time) * 1000)
                w0, h0 = frame_rgb.size
                batch_meta.append({"idx": cur_idx_peek, "w": w0, "h": h0, "t": ts_ms_peek})
                if len(batch) >= self.batch_size:
                    dets_with_arrays = await self._run_inference_batch(batch)
                    # Run tracker per frame and append frames with detections to bucket buffer
                    for meta, det_pair in zip(batch_meta, dets_with_arrays):
                        dets, det_np = det_pair
                        frame_idx = int(meta.get("idx", 0))
                        if fb_start_idx is None:
                            fb_start_idx = frame_idx
                        # Update tracker
                        trk_out = tracker.update(det_np if det_np is not None else np.empty((0,5)))
                        # Map tracker IDs back onto normalized dets using IoU match in pixel space
                        trk_ids = self._assign_tracker_ids(det_np, trk_out)
                        enriched = []
                        for j, d in enumerate(dets):
                            tid = trk_ids.get(j)
                            # Always attach a tracker id; if unmatched, create a temp id per detection
                            if tid is None:
                                tid = int((frame_idx << 20) + j)  # temp unique
                            d = dict(d)
                            d["k"] = int(tid)
                            # Track span update
                            span = track_span.get(int(tid))
                            if span is None:
                                track_span[int(tid)] = {"from_i": frame_idx, "to_i": frame_idx}
                            else:
                                span["to_i"] = frame_idx
                            enriched.append(d)
                        fb_frames.append({
                            "i": frame_idx,
                            "t": int(meta.get("t", 0)),
                            "o": enriched,
                        })
                        # Update block summary counts
                        if enriched:
                            for d in enriched:
                                cname = d.get("c")
                                if cname:
                                    block_counts[cname] = block_counts.get(cname, 0) + 1
                    frames_processed += len(batch)
                    batch.clear()
                    batch_meta.clear()

                # Note: fb_frames are appended after inference when batch flushes
                current_frame_index += 1
                # Flush every 256 frames, or when explicitly requested via last_batch flag
                if (len(fb_frames) >= BUCKET) or last_batch:
                    try:
                        coll = get_collection("frame_buckets")
                        # Create indexes once best-effort
                        try:
                            coll.create_index([("job_id", 1), ("block_start_idx", 1)])
                            coll.create_index([("job_id", 1), ("block_end_idx", 1)])
                            coll.create_index([("job_id", 1), ("has_objects", 1), ("block_start_idx", 1)])
                        except Exception:
                            pass
                        block_doc = {
                            "job_id": job_id,
                            "block_start_idx": int(fb_start_idx),
                            "block_end_idx": int(fb_frames[-1]["i"]),
                            "fps": float(vid_fps or 0.0),
                            "imgw": imgw,
                            "imgh": imgh,
                            "has_objects": any(bool(fr.get("o")) for fr in fb_frames),
                            "object_counts": dict(block_counts),
                            "frames": fb_frames,
                        }
                        coll.insert_one(block_doc)
                    except Exception:
                        logger.error(f"Failed to insert frame bucket for job: {job_id}")
                        pass
                    finally:
                        fb_frames = []
                        fb_start_idx = None
                        block_counts = {}

                # Periodic progress update (every ~2s)
                now = time.time()
                if (now - last_update) >= 2.0:
                    progress_pct = 0.0
                    if total_frames > 0:
                        progress_pct = min(100.0, (frames_processed / float(total_frames)) * 100.0)
                    # Refresh preview from current frame (200px width)
                    try:
                        w_cur, h_cur = frame_rgb.size
                        if w_cur > 0:
                            new_w_cur = 200
                            new_h_cur = max(1, int(h_cur * (new_w_cur / float(w_cur))))
                            prev_cur = frame_rgb.resize((new_w_cur, new_h_cur), Image.BILINEAR)
                            preview_b64 = image_to_base64_jpeg(prev_cur, quality=75)
                    except Exception:
                        pass
                    update_job_progress(
                        job_id,
                        progress_pct,
                        float(frames_processed),
                        float(total_frames if total_frames > 0 else max(1, frames_processed)),
                        frames_processed=frames_processed,
                        preview_b64=preview_b64,
                    )
                    last_update = now

            # Flush remaining detections for partial batch
            if batch:
                dets_with_arrays = await self._run_inference_batch(batch)
                for meta, det_pair in zip(batch_meta, dets_with_arrays):
                    dets, det_np = det_pair
                    frame_idx = int(meta.get("idx", 0))
                    if fb_start_idx is None:
                        fb_start_idx = frame_idx
                    trk_out = tracker.update(det_np if det_np is not None else np.empty((0,5)))
                    trk_ids = self._assign_tracker_ids(det_np, trk_out)
                    enriched = []
                    for j, d in enumerate(dets):
                        tid = trk_ids.get(j)
                        if tid is None:
                            tid = int((frame_idx << 20) + j)
                        d = dict(d)
                        d["k"] = int(tid)
                        span = track_span.get(int(tid))
                        if span is None:
                            track_span[int(tid)] = {"from_i": frame_idx, "to_i": frame_idx}
                        else:
                            span["to_i"] = frame_idx
                        enriched.append(d)
                    fb_frames.append({
                        "i": frame_idx,
                        "t": int(meta.get("t", 0)),
                        "o": enriched,
                    })
                    if enriched:
                        for d in enriched:
                            cname = d.get("c")
                            if cname:
                                block_counts[cname] = block_counts.get(cname, 0) + 1
                frames_processed += len(batch)
                batch.clear()
                batch_meta.clear()
                # Mark as last_batch and force a flush of the final partial block
                last_batch = True
                if (len(fb_frames) >= BUCKET) or last_batch:
                    try:
                        coll = get_collection("frame_buckets")
                        try:
                            coll.create_index([( "job_id", 1), ("block_start_idx", 1)])
                            coll.create_index([( "job_id", 1), ("block_end_idx", 1)])
                            coll.create_index([( "job_id", 1), ("has_objects", 1), ("block_start_idx", 1)])
                        except Exception:
                            pass
                        block_doc = {
                            "job_id": job_id,
                            "block_start_idx": int(fb_start_idx if fb_start_idx is not None else (fb_frames[0]["i"] if fb_frames else 0)),
                            "block_end_idx": int(fb_frames[-1]["i"]) if fb_frames else (fb_start_idx or 0),
                            "fps": float(vid_fps or 0.0),
                            "imgw": imgw,
                            "imgh": imgh,
                            "has_objects": any(bool(fr.get("o")) for fr in fb_frames),
                            "object_counts": dict(block_counts),
                            "frames": fb_frames,
                        }
                        if fb_frames:
                            coll.insert_one(block_doc)
                    except Exception:
                        logger.error(f"Failed to insert frame bucket for job: {job_id}")
                        pass
                    finally:
                        fb_frames = []
                        fb_start_idx = None
                        block_counts = {}

            # Final progress update at 100% (DB only; UI will get 'completed' event from worker)
            progress_pct = 100.0
            update_job_progress(
                job_id,
                progress_pct,
                float(max(frames_processed, total_frames)),
                float(total_frames if total_frames > 0 else max(1, frames_processed)),
                frames_processed=frames_processed,
                preview_b64=preview_b64,
                emit=False,
            )
            # Flush remaining frame bucket (if any)
            if fb_frames:
                try:
                    coll = get_collection("frame_buckets")
                    block_doc = {
                        "job_id": job_id,
                        "block_start_idx": int(fb_start_idx if fb_start_idx is not None else fb_frames[0]["i"]),
                        "block_end_idx": int(fb_frames[-1]["i"]),
                        "fps": float(vid_fps or 0.0),
                        "imgw": imgw,
                        "imgh": imgh,
                        "has_objects": any(bool(fr.get("o")) for fr in fb_frames),
                        "object_counts": dict(block_counts),
                        "frames": fb_frames,
                    }
                    coll.insert_one(block_doc)
                except Exception:
                    pass
            # Persist track spans: update each frame's detections with from/to
            try:
                coll = get_collection("frame_buckets")
                for tid, span in track_span.items():
                    from_i = int(span.get("from_i", 0))
                    to_i = int(span.get("to_i", from_i))
                    # Compute timestamps using vid_fps if available
                    def ms_for(idx: int) -> int:
                        if vid_fps and vid_fps > 0:
                            return int((idx / vid_fps) * 1000.0)
                        return 0
                    from_t = ms_for(from_i)
                    to_t = ms_for(to_i)
                    # Update all frames in range for this job; iterate blocks to avoid large arrayFilters
                    for block in coll.find({"job_id": job_id, "block_end_idx": {"$gte": from_i}, "block_start_idx": {"$lte": to_i}}, {"_id":1, "frames":1}):
                        frs = block.get("frames") or []
                        changed = False
                        for fr in frs:
                            fi = int(fr.get("i", -1))
                            if fi < from_i or fi > to_i:
                                continue
                            outs = []
                            for det in (fr.get("o") or []):
                                if int(det.get("k", -1)) == int(tid):
                                    if det.get("from") != from_t or det.get("to") != to_t or det.get("from_frame") != from_i or det.get("to_frame") != to_i:
                                        det = dict(det)
                                        det["from"] = from_t
                                        det["to"] = to_t
                                        det["from_frame"] = from_i
                                        det["to_frame"] = to_i
                                        changed = True
                                outs.append(det)
                            if changed:
                                fr["o"] = outs
                        if changed:
                            try:
                                coll.update_one({"_id": block.get("_id")}, {"$set": {"frames": frs}})
                            except Exception:
                                pass
            except Exception:
                logger.error(f"Failed to update track spans for job: {job_id}")
                pass
        except Exception:
            raise
        finally:
            cap.release()

    async def _run_inference_batch(self, frames: List[Image.Image]) -> List[tuple[List[Dict[str, Any]], Any]]:
        # Run inference using Ultralytics YOLO and return (lean detections, det_array Nx5 pixels)
        def _infer_sync(images: List[Image.Image]) -> List[tuple[List[Dict[str, Any]], Any]]:
            try:
                if self._model is None:
                    logger.error("No model loaded for inference")
                    return [([], np.empty((0,5), dtype=float)) for _ in images]
                np_imgs = [np.array(im) for im in images]
                # Choose device automatically
                try:                    
                    use_device = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    use_device = "cpu"
                # Predict across frameworks
                try:
                    results = self._model.predict(np_imgs, verbose=False, device=use_device)
                except Exception:
                    try:
                        results = self._model(np_imgs)  # type: ignore
                    except Exception:
                        results = []
                out: List[tuple[List[Dict[str, Any]], Any]] = []
                for res, im in zip(results, images):
                    dets: List[Dict[str, Any]] = []
                    det_np = np.empty((0,5), dtype=float)
                    try:
                        w, h = im.size
                        boxes = getattr(res, 'boxes', None)
                        xyxy = boxes.xyxy.cpu().numpy() if (boxes is not None and hasattr(boxes, 'xyxy')) else []
                        conf = boxes.conf.cpu().numpy() if (boxes is not None and hasattr(boxes, 'conf')) else []
                        cls_ids = boxes.cls.cpu().numpy().astype(int) if (boxes is not None and hasattr(boxes, 'cls')) else []
                        # Prefer names on result; fallback to model
                        name_map = {}
                        try:
                            nm = getattr(res, 'names', None)
                            if isinstance(nm, dict):
                                name_map = nm
                        except Exception:
                            name_map = {}
                        if not name_map:
                            mm = getattr(self._model, 'names', None)
                            if isinstance(mm, dict):
                                name_map = mm
                        # Build pixel det array for tracker
                        if len(xyxy) > 0:
                            px = np.array(xyxy, dtype=float)
                            sc = np.array(conf, dtype=float) if len(conf) == len(px) else np.ones((len(px),), dtype=float)
                            det_np = np.concatenate([px, sc.reshape(-1,1)], axis=1)
                        for i in range(len(xyxy)):
                            x1, y1, x2, y2 = xyxy[i]
                            # normalize to 0..1
                            nb = [
                                float(max(0.0, min(1.0, x1 / max(1.0, w)))),
                                float(max(0.0, min(1.0, y1 / max(1.0, h)))),
                                float(max(0.0, min(1.0, x2 / max(1.0, w)))),
                                float(max(0.0, min(1.0, y2 / max(1.0, h)))),
                            ]
                            cid = int(cls_ids[i]) if i < len(cls_ids) else -1
                            cname = name_map.get(cid, str(cid))
                            score = float(conf[i]) if i < len(conf) else 0.0
                            dets.append({"c": cname, "s": score, "b": nb})
                    except Exception as e:
                        logger.error(f"Failed to process detections: {e}")
                        dets = []
                    out.append((dets, det_np))
                return out
            except Exception as e:
                logger.error(f"Failed to run inference: {e}")
                # Swallow inference errors for now; in future, log or propagate
                return [([], np.empty((0,5), dtype=float)) for _ in images]

        return await asyncio.to_thread(_infer_sync, frames)

    def _assign_tracker_ids(self, det_np: Any, trk_out: Any) -> Dict[int, int]:
        """Map detection indices to tracker IDs using IoU matching in pixel space."""
        try:
            if det_np is None or len(det_np) == 0 or trk_out is None or len(trk_out) == 0:
                return {}
            det_boxes = det_np[:, :4]
            trk_boxes = np.array(trk_out)[:, :4]
            trk_ids = np.array(trk_out)[:, 4].astype(int)
            # Compute IoU matrix
            ious = self._iou_matrix(det_boxes, trk_boxes)
            mapping: Dict[int,int] = {}
            for d_idx in range(len(det_boxes)):
                t_idx = int(np.argmax(ious[d_idx]))
                if ious[d_idx, t_idx] >= 0.3:
                    mapping[d_idx] = int(trk_ids[t_idx])
            return mapping
        except Exception:
            return {}

    @staticmethod
    def _iou_matrix(a: Any, b: Any) -> Any:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if a.size == 0 or b.size == 0:
            return np.zeros((len(a), len(b)), dtype=float)
        ious = np.zeros((len(a), len(b)), dtype=float)
        for i in range(len(a)):
            x1 = np.maximum(a[i,0], b[:,0])
            y1 = np.maximum(a[i,1], b[:,1])
            x2 = np.minimum(a[i,2], b[:,2])
            y2 = np.minimum(a[i,3], b[:,3])
            w = np.maximum(0.0, x2 - x1)
            h = np.maximum(0.0, y2 - y1)
            inter = w * h
            area_a = (a[i,2]-a[i,0]) * (a[i,3]-a[i,1])
            area_b = (b[:,2]-b[:,0]) * (b[:,3]-b[:,1])
            union = area_a + area_b - inter
            ious[i,:] = np.where(union > 0, inter / union, 0.0)
        return ious

    def _build_results(self, *, job_id: str, fps: float, frame_w: int, frame_h: int, model_name: str, model_path: str) -> List[Dict[str, Any]]:
        coll_fb = get_collection("frame_buckets")
        # Prefer known dims from job doc; fall back to first block dims if missing
        out_width = int(frame_w or 0)
        out_height = int(frame_h or 0)

        # Group detections by tracker id
        tracks: Dict[int, Dict[str, Any]] = {}
        try:
            cur = coll_fb.find({"job_id": job_id}, {"frames": 1, "imgw": 1, "imgh": 1}).sort([("block_start_idx", 1)])
        except Exception:
            cur = []

        for block in cur:
            if out_width <= 0:
                try:
                    out_width = int(block.get("imgw") or out_width)
                except Exception:
                    pass
            if out_height <= 0:
                try:
                    out_height = int(block.get("imgh") or out_height)
                except Exception:
                    pass
            frames = block.get("frames") or []
            for fr in frames:
                try:
                    fi = int(fr.get("i", -1))
                except Exception:
                    continue
                objs = fr.get("o") or []
                for det in objs:
                    try:
                        conf = float(det.get("s", 0.0))
                    except Exception:
                        conf = 0.0
                    try:
                        tid = int(det.get("k"))
                    except Exception:
                        # skip unmatched without stable id
                        continue
                    cls_name = det.get("c") or "unknown"
                    nb = det.get("b") or [0.0, 0.0, 0.0, 0.0]
                    try:
                        x1 = float(nb[0]) * max(1, out_width)
                        y1 = float(nb[1]) * max(1, out_height)
                        x2 = float(nb[2]) * max(1, out_width)
                        y2 = float(nb[3]) * max(1, out_height)
                        w = max(0.0, x2 - x1)
                        h = max(0.0, y2 - y1)
                        # cx = x1 + (w / 2.0)
                        # cy = y1 + (h / 2.0)
                    except Exception:
                        continue

                    # Use persisted track span when available
                    try:
                        from_fi = int(det.get("from_frame", fi))
                    except Exception:
                        from_fi = fi
                    try:
                        to_fi = int(det.get("to_frame", fi))
                    except Exception:
                        to_fi = fi

                    tr = tracks.get(tid)
                    if tr is None:
                        tr = {
                            "object": cls_name,
                            "start_frame_number": from_fi,
                            "end_frame_number": to_fi,
                            "detections": [],
                        }
                        tracks[tid] = tr
                    else:
                        if from_fi < tr["start_frame_number"]:
                            tr["start_frame_number"] = from_fi
                        if to_fi > tr["end_frame_number"]:
                            tr["end_frame_number"] = to_fi
                    tr["detections"].append({
                        "confidence": f"{conf:.5f}",
                        "x": float(x1),
                        "y": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "w": float(w),
                        "h": float(h),
                        "height": int(out_height or 0),
                        "width": int(out_width or 0),
                        "object": cls_name,
                    })

        model_id = model_name or (os.path.basename(model_path) if model_path else str(self.model_path or ""))

        results: List[Dict[str, Any]] = []
        for tr in tracks.values():
            sf = int(tr.get("start_frame_number", 0))
            ef = int(tr.get("end_frame_number", sf))
            if fps and fps > 0:
                start_ms = int((sf / fps) * 1000.0)
                end_ms = int((ef / fps) * 1000.0)
            else:
                start_ms = 0
                end_ms = 0
            # Merge all per-frame boxes into one encompassing bbox
            x_min = None
            y_min = None
            x_max = None
            y_max = None
            max_conf = 0.0
            for d in tr.get("detections", []) or []:
                try:
                    x1 = float(d.get("x", 0.0))
                    y1 = float(d.get("y", 0.0))
                    # Prefer x2/y2 if present; fallback to x+w/y+h
                    if d.get("x2") is not None and d.get("y2") is not None:
                        x2 = float(d.get("x2"))
                        y2 = float(d.get("y2"))
                    else:
                        w = float(d.get("w", 0.0))
                        h = float(d.get("h", 0.0))
                        x2 = x1 + max(0.0, w)
                        y2 = y1 + max(0.0, h)
                    c = float(d.get("confidence", 0.0)) if isinstance(d.get("confidence"), (int, float)) else float(str(d.get("confidence", "0")).strip() or 0.0)
                except Exception:
                    continue
                x_min = x1 if x_min is None else min(x_min, x1)
                y_min = y1 if y_min is None else min(y_min, y1)
                x_max = x2 if x_max is None else max(x_max, x2)
                y_max = y2 if y_max is None else max(y_max, y2)
                if c > max_conf:
                    max_conf = c

            if x_min is None or y_min is None or x_max is None or y_max is None:
                # No valid boxes; skip
                continue

            # Skip whole track if its maximum confidence never reaches threshold
            if max_conf < 0.5:
                continue

            merged_det = {
                "confidence": f"{max_conf:.5f}",
                "x": float(x_min),
                "y": float(y_min),
                "w": float(max(0.0, x_max - x_min)),
                "h": float(max(0.0, y_max - y_min)),
                "height": int(out_height or 0),
                "width": int(out_width or 0),
                "object": tr.get("object"),
            }

            results.append({
                "start": start_ms,
                "end": end_ms,
                "start_frame_number": sf,
                "end_frame_number": ef,
                "object": tr.get("object"),
                "model_id": model_id,
                "detections": [merged_det],
                "sub_items": [],
            })

        return results


