from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import time
import logging
import asyncio

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from ultralytics import SAM
from .job_interface import Job, update_job_progress
from ..utils.media_utils import image_to_base64_jpeg
from ..database.mongo import get_collection

logger = logging.getLogger(__name__)

class FindAnything(Job):
    """Find-anything job using a generic segment-anything model to propose boxes.

    TODO: create version with SAM2 model
    """

    def __init__(self, *, frame_stride: int = 15) -> None:
        self.frame_stride = max(1, int(frame_stride))
        self._sam_model = None  # lazy-loaded

    def _lazy_load_model(self) -> None:
        if self._sam_model is not None:
            return
        logger.info("Loading SAM model")
        # Try a few known entry points; gracefully continue if unavailable
        try:
            # Ultralytics may route SAM weights via YOLO API
            
            try:
                # Attempt common lightweight SAM weights if present in model directory
                self._sam_model = YOLO("sam_b.pt")
            except Exception:
                self._sam_model = None
        except Exception:
            self._sam_model = None
        if self._sam_model is None:
            try:
                
                # Will require SAM weights present alongside app, else stays None
                self._sam_model = SAM("sam_b.pt")  # type: ignore
            except Exception:
                self._sam_model = None

        logger.info("SAM model loaded") if self._sam_model is not None else logger.error("SAM model not loaded")

    def _infer_boxes(self, frame_rgb: Image.Image) -> List[Tuple[float, float, float, float]]:
        """Return a list of boxes in absolute pixel xyxy using SAM if available.
        Falls back to empty list if no model is available.
        """
        self._lazy_load_model()
        if self._sam_model is None:
            return []
        np_img = np.array(frame_rgb)
        try:
            # Prefer predict API; fallback to calling
            try:
                results = self._sam_model.predict(np_img, verbose=False)
            except Exception:
                results = self._sam_model(np_img)  # type: ignore
            boxes_out: List[Tuple[float, float, float, float]] = []
            for res in results:
                # If boxes exist directly
                boxes = getattr(res, 'boxes', None)
                if boxes is not None and hasattr(boxes, 'xyxy'):
                    arr = boxes.xyxy.cpu().numpy()
                    for row in arr:
                        x1, y1, x2, y2 = [float(v) for v in row[:4]]
                        boxes_out.append((x1, y1, x2, y2))
                    continue
                # Else derive boxes from masks if present
                masks = getattr(res, 'masks', None)
                if masks is not None and hasattr(masks, 'data'):
                    md = masks.data  # torch tensor [N,H,W]
                    try:
                        arr = md.cpu().numpy()
                    except Exception:
                        arr = None
                    if arr is not None:
                        for m in arr:
                            ys, xs = np.where(m > 0.5)
                            if ys.size == 0 or xs.size == 0:
                                continue
                            x1 = float(np.min(xs)); y1 = float(np.min(ys))
                            x2 = float(np.max(xs)); y2 = float(np.max(ys))
                            boxes_out.append((x1, y1, x2, y2))
            return boxes_out
        except Exception:
            return []

    async def process(self, job_id: str, media_path: str) -> None:
        # Run the heavy processing in a background thread to avoid blocking the event loop
        await asyncio.to_thread(self._process_sync, job_id, media_path)

    def _process_sync(self, job_id: str, media_path: str) -> None:
        p = Path(media_path)
        if not p.exists():
            raise FileNotFoundError(str(p))

        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise RuntimeError("Failed to open media")

        logger.info(f"Processing video: {media_path}")
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        except Exception:
            total_frames = 0
        try:
            vid_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        except Exception:
            vid_fps = 0.0
        try:
            imgw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            imgh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        except Exception:
            imgw, imgh = 0, 0

        frames_processed = 0
        preview_b64: Optional[str] = None
        last_update = time.time()
        first_preview_sent = False

        BUCKET = 256
        fb_frames: List[Dict[str, Any]] = []
        fb_start_idx: Optional[int] = None
        block_counts: Dict[str, int] = {}

        frame_idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            # Compute timestamp in ms for this frame
            t_ms = int((frame_idx / vid_fps) * 1000.0) if vid_fps and vid_fps > 0 else 0
            if frame_idx % self.frame_stride == 0:
                frame_rgb = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                boxes = self._infer_boxes(frame_rgb)
                # Normalize and pack detections
                dets: List[Dict[str, Any]] = []
                for (x1, y1, x2, y2) in boxes:
                    if imgw > 0 and imgh > 0:
                        nb = [
                            float(max(0.0, min(1.0, x1 / float(max(1, imgw))))),
                            float(max(0.0, min(1.0, y1 / float(max(1, imgh))))),
                            float(max(0.0, min(1.0, x2 / float(max(1, imgw))))),
                            float(max(0.0, min(1.0, y2 / float(max(1, imgh))))),
                        ]
                    else:
                        nb = [0.0, 0.0, 0.0, 0.0]
                    dets.append({"c": "unknown", "s": 1.0, "b": nb})
                    block_counts["unknown"] = block_counts.get("unknown", 0) + 1
                if fb_start_idx is None:
                    fb_start_idx = frame_idx
                fb_frames.append({"i": int(frame_idx), "t": t_ms, "o": dets})
                # Update preview from current processed frame (small)
                try:
                    w, h = frame_rgb.size
                    if w > 0:
                        new_w = 200
                        new_h = max(1, int(h * (new_w / float(w))))
                        prev = frame_rgb.resize((new_w, new_h), Image.BILINEAR)
                        preview_b64 = image_to_base64_jpeg(prev, quality=75)
                except Exception:
                    pass
                # Emit immediately on first preview so UI shows quickly
                if not first_preview_sent and preview_b64 is not None:
                    pct0 = 0.0
                    if total_frames > 0:
                        pct0 = min(100.0, (frames_processed / float(total_frames)) * 100.0)
                    update_job_progress(
                        job_id,
                        pct0,
                        float(frames_processed),
                        float(total_frames if total_frames > 0 else max(1, frames_processed)),
                        frames_processed=frames_processed,
                        preview_b64=preview_b64,
                    )
                    first_preview_sent = True
            frames_processed += 1

            # Flush periodically
            if fb_frames and ((len(fb_frames) >= BUCKET) or (frame_idx + 1 >= total_frames)):                
                try:
                    coll = get_collection("frame_buckets")
                    try:
                        coll.create_index([("job_id", 1), ("block_start_idx", 1)])
                        coll.create_index([("job_id", 1), ("block_end_idx", 1)])
                        coll.create_index([("job_id", 1), ("has_objects", 1), ("block_start_idx", 1)])
                    except Exception:
                        pass
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
                finally:
                    fb_frames = []
                    fb_start_idx = None
                    block_counts = {}

            # Progress update ~2s
            now = time.time()
            if (now - last_update) >= 2.0:
                
                pct = 0.0
                if total_frames > 0:
                    pct = min(100.0, (frames_processed / float(total_frames)) * 100.0)
                update_job_progress(
                    job_id,
                    pct,
                    float(frames_processed),
                    float(total_frames if total_frames > 0 else max(1, frames_processed)),
                    frames_processed=frames_processed,
                    preview_b64=preview_b64,
                    emit=True
                )
                last_update = now
                logger.info(f"[{media_path}] Processed {frames_processed}/{total_frames} frames")

            frame_idx += 1

        # Final flush if any remaining
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
        
        #done 
        update_job_progress(
            job_id,
            100.0,
            float(max(frames_processed, total_frames)),
            float(total_frames if total_frames > 0 else max(1, frames_processed)),
            frames_processed=frames_processed,
            preview_b64=preview_b64
        )
        cap.release()


