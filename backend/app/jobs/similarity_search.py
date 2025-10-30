from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from .job_interface import Job
from .job_interface import update_job_progress
from ..database.mongo import get_collection
from ..service.embedder_service import is_model_loaded, preload_model, embed_image
from ..service.system_service import get_configuration
from ..database.milvus import search_embeddings
from ..utils.media_utils import image_to_base64_jpeg


class SimilaritySearch(Job):
    def __init__(self, *, threshold: float | None = 0.7, batch_size: int = 4) -> None:
        cfg = get_configuration()
        self.threshold: float = float(threshold if threshold is not None else (cfg.get("similarity_threshold") or 0.7))
        self.batch_size: int = int(batch_size or 4)
        self.collection_name: str = str(cfg.get("similarity_collection") or cfg.get("things_collection") or "things")
        self.top_k: int = 5
        # Ensure embedder is warm
        try:
            if not is_model_loaded():
                hf_token = cfg.get("hf_token") or None
                model_id = cfg.get("dinov3_model") or "facebook/dinov3-vitb16-pretrain-lvd1689m"
                preload_model(model_id=model_id, hf_token=hf_token)
        except Exception:
            pass

    async def process(self, job_id: str, media_path: str) -> None:
        # Run the heavy workflow off the event loop to avoid blocking API endpoints
        await asyncio.to_thread(self._process_sync, job_id, media_path)

    def _process_sync(self, job_id: str, media_path: str) -> None:
        p = Path(media_path)
        if not p.exists():
            raise FileNotFoundError(str(p))

        cap = cv2.VideoCapture(str(p))
        is_video = cap.isOpened()
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if is_video else 1
        vid_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) if is_video else 0.0
        imgw = imgh = 0
        preview_b64: Optional[str] = None
        frames_processed = 0
        last_update = time.time()

        # frame bucket state
        BUCKET = 256
        fb_frames: List[dict] = []
        fb_start_idx: Optional[int] = None

        def flush_bucket() -> None:
            if not fb_frames:
                return
            try:
                coll = get_collection("frame_buckets")
                try:
                    coll.create_index([("job_id", 1), ("block_start_idx", 1)])
                    coll.create_index([("job_id", 1), ("block_end_idx", 1)])
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
                    "object_counts": {},
                    "frames": fb_frames,
                }
                coll.insert_one(block_doc)
            except Exception:
                pass
            finally:
                fb_frames.clear()

        def to_pil(frame_bgr: Any) -> Image.Image:
            return Image.fromarray(frame_bgr[:, :, ::-1])

        idx = 0
        while True:
            frame_bgr = None
            if is_video:
                ok, f = cap.read()
                if not ok:
                    break
                frame_bgr = f
            else:
                frame_bgr = cv2.imread(str(p))
                if frame_bgr is None:
                    break
            if imgw == 0 or imgh == 0:
                h, w = frame_bgr.shape[:2]
                imgw, imgh = w, h

            # embed and search
            try:
                pil = to_pil(frame_bgr)
                # Create/update 200px-wide preview
                if preview_b64 is None:
                    try:
                        w, h = pil.size
                        if w > 0:
                            new_w = 200
                            new_h = max(1, int(h * (new_w / float(w))))
                            prev = pil.resize((new_w, new_h), Image.BILINEAR)
                            preview_b64 = image_to_base64_jpeg(prev, quality=75)
                    except Exception:
                        preview_b64 = None
                emb = embed_image(pil)
                results = search_embeddings(self.collection_name, [emb], top_k=self.top_k) or []
            except Exception:
                results = []

            # compact similarity results per frame
            sims: List[Dict[str, Any]] = []
            try:
                rows = results[0] if isinstance(results, list) and results else []
                for hit in rows:
                    try:
                        # PyMilvus Hit: attributes often: id, distance, entity
                        ref_id = getattr(hit, 'id', None)                        
                        dist = getattr(hit, 'distance', None)
                        # Treat returned value as similarity score (higher is better)
                        if not isinstance(dist, (int, float)):
                            continue
                        score = float(dist)
                        # Extract payload/metadata
                        meta = None
                        img_id = None
                        try:
                            ent = getattr(hit, 'entity', None)
                            if ent is not None:
                                payload_val = ent.get('payload') if hasattr(ent, 'get') else None
                                if isinstance(payload_val, str):
                                    
                                    try:
                                        meta = json.loads(payload_val)
                                    except Exception:
                                        meta = {"payload": payload_val}
                                elif isinstance(payload_val, dict):
                                    meta = payload_val
                        except Exception:
                            meta = None
                        label = None
                        if isinstance(meta, dict):
                            label = meta.get('label') or meta.get('name') or meta.get('filename')
                            img_id = meta.get('image_id')
                        sims.append({"ref": ref_id, "s": score, "m": {"label": label} if label else {}, "img_id": img_id if img_id else None})
                    except Exception:
                        pass
            except Exception:
                pass

            fb_frames.append({"i": idx, "t": int((idx / vid_fps) * 1000.0) if vid_fps > 0 else 0, "o": sims})
            if fb_start_idx is None:
                fb_start_idx = idx

            frames_processed += 1
            idx += 1
            if len(fb_frames) >= BUCKET:
                flush_bucket()
                fb_start_idx = None

            now = time.time()
            if (now - last_update) >= 2.0:
                pct = min(100.0, (frames_processed / float(total_frames or max(1, frames_processed))) * 100.0)
                try:
                    # Refresh preview from current frame
                    try:
                        w_cur, h_cur = pil.size if 'pil' in locals() else (0, 0)
                        if w_cur and h_cur:
                            new_w_cur = 200
                            new_h_cur = max(1, int(h_cur * (new_w_cur / float(w_cur))))
                            prev_cur = pil.resize((new_w_cur, new_h_cur), Image.BILINEAR)
                            preview_b64 = image_to_base64_jpeg(prev_cur, quality=75)
                    except Exception:
                        pass
                    update_job_progress(job_id, pct, float(frames_processed), float(total_frames or frames_processed), frames_processed=frames_processed, preview_b64=preview_b64)
                except Exception:
                    pass
                last_update = now

            if not is_video:
                break

        # final flush
        flush_bucket()
        cap.release()


