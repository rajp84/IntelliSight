import asyncio
import logging
import queue
import time
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, List

import cv2
import numpy as np
from PIL import Image
from ..utils.media_utils import image_to_base64_jpeg
import json as _json

logger = logging.getLogger(__name__)


def create_testing_threads(
    *,
    resolved_path: Path,
    is_video: bool,
    cap: Optional[cv2.VideoCapture],
    total_frames: int,
    batch_size: int,
    should_stop: Callable[[], bool],
    set_status: Callable[..., None],
    broadcast: Callable[[str, Any], Any],
    embed_image: Callable[[Image.Image], Any],
    search_embeddings: Callable[[str, List[List[float]], int], Any],
    threshold: float,
    flush_interval_sec: float = 0.40,
) -> Tuple[queue.Queue, queue.Queue, Callable[[], None], Callable[[], None], Callable[[], None]]:
    """
    Build reader, batcher, and processor worker callables and associated queues.

    Returns (frame_queue, result_queue, reader, batcher, processor)
    """

    frame_queue: "queue.Queue[tuple[int, np.ndarray]]" = queue.Queue(maxsize=1024)
    result_queue: "queue.Queue[Any]" = queue.Queue(maxsize=1024)

    def reader() -> None:
        try:
            frame_idx_local = 0
            local_fps_counter = 0
            local_fps_t0 = time.time()
            while not should_stop():
                frame_data = None
                if is_video and cap is not None:
                    ok, frame_bgr = cap.read()
                    if not ok:
                        break
                    frame_data = frame_bgr
                else:
                    img = Image.open(str(resolved_path)).convert("RGB")
                    frame_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    frame_data = frame_bgr

                # enqueue
                try:
                    frame_queue.put((frame_idx_local, frame_data), timeout=0.5)
                except queue.Full:
                    if should_stop():
                        break
                    continue

                # status/progress
                if total_frames > 0:
                    pct = min(100, int(((frame_idx_local + 1) / total_frames) * 100))
                    set_status(
                        progress=pct,
                        message=f"Testing frame {frame_idx_local + 1}{'/' + str(total_frames) if total_frames>0 else ''}",
                        queue_frames=frame_queue.qsize(),
                        queue_frames_max=frame_queue.maxsize,
                        queue_results=result_queue.qsize(),
                        queue_results_max=result_queue.maxsize,
                    )
                    try:
                        asyncio.run(broadcast("testing_status", {}))
                    except RuntimeError:
                        pass
                if is_video:
                    local_fps_counter += 1
                    now = time.time()
                    if (now - local_fps_t0) >= 1.0:
                        fps = local_fps_counter / (now - local_fps_t0)
                        local_fps_t0 = now
                        local_fps_counter = 0
                        set_status(
                            fps=round(fps, 2),
                            queue_frames=frame_queue.qsize(),
                            queue_frames_max=frame_queue.maxsize,
                            queue_results=result_queue.qsize(),
                            queue_results_max=result_queue.maxsize,
                        )
                        try:
                            asyncio.run(broadcast("testing_status", {}))
                        except RuntimeError:
                            pass

                frame_idx_local += 1

        finally:
            try:
                frame_queue.put((-1, np.empty((0, 0, 3), dtype=np.uint8)), timeout=0.5)
            except Exception:
                pass

    def batcher() -> None:
        batch_frames: List[np.ndarray] = []
        batch_indices: List[int] = []
        batch_previews: List[Optional[str]] = []
        last_flush = time.time()
        batch_start_t: Optional[float] = None
        while not should_stop():
            try:
                idx, frame_bgr = frame_queue.get(timeout=0.05)
            except queue.Empty:
                idx = None
                frame_bgr = None

            if idx is not None:
                if idx == -1:
                    if batch_frames:
                        try:
                            embeddings: List[List[float]] = []
                            for f in batch_frames:
                                pil = Image.fromarray(f[:, :, ::-1])
                                emb = embed_image(pil)
                                embeddings.append(emb.tolist())
                            res = search_embeddings("things", embeddings, 5)
                            for bi, fidx in enumerate(batch_indices):
                                hits = (res[bi] if (res and len(res) > bi) else [])
                                preview_b64 = batch_previews[bi] if bi < len(batch_previews) else None
                                result_queue.put((fidx, hits, preview_b64))
                        except Exception as e:
                            logger.error(f"[TEST] Error in final batch processing: {e}")
                        finally:
                            batch_frames.clear()
                            batch_indices.clear()
                            batch_previews.clear()
                    try:
                        result_queue.put((-1, None), timeout=0.5)
                    except Exception:
                        pass
                    break

                batch_frames.append(frame_bgr)  # type: ignore[arg-type]
                batch_indices.append(idx)
                try:
                    pv: Optional[str] = image_to_base64_jpeg(Image.fromarray(frame_bgr[:, :, ::-1]))
                except Exception:
                    pv = None
                batch_previews.append(pv)
                if batch_start_t is None:
                    batch_start_t = time.time()
                try:
                    set_status(
                        queue_batch=len(batch_frames),
                        queue_batch_max=batch_size,
                    )
                    asyncio.run(broadcast("testing_status", {}))
                except RuntimeError:
                    pass

            time_since = time.time() - last_flush
            if (len(batch_frames) >= batch_size) or (len(batch_frames) > 0 and time_since >= flush_interval_sec):
                try:
                    embeddings: List[List[float]] = []
                    for f in batch_frames:
                        pil = Image.fromarray(f[:, :, ::-1])
                        emb = embed_image(pil)
                        embeddings.append(emb.tolist())
                    res = search_embeddings("things", embeddings, 5)
                    for bi, fidx in enumerate(batch_indices):
                        hits = (res[bi] if (res and len(res) > bi) else [])
                        preview_b64 = batch_previews[bi] if bi < len(batch_previews) else None
                        result_queue.put((fidx, hits, preview_b64))
                    last_sz = len(batch_frames)
                    last_ms = int(((time.time() - batch_start_t) * 1000)) if batch_start_t else None
                    set_status(
                        queue_frames=frame_queue.qsize(),
                        queue_frames_max=frame_queue.maxsize,
                        queue_results=result_queue.qsize(),
                        queue_results_max=result_queue.maxsize,
                        queue_batch=0,
                        queue_batch_max=batch_size,
                        last_batch_size=last_sz,
                        last_batch_ms=last_ms,
                    )
                    try:
                        asyncio.run(broadcast("testing_status", {}))
                    except RuntimeError:
                        pass
                except Exception as e:
                    logger.error(f"[TEST] Error processing batch of {len(batch_frames)}: {e}")
                finally:
                    batch_frames.clear()
                    batch_indices.clear()
                    batch_previews.clear()
                    last_flush = time.time()
                    batch_start_t = None

    def processor() -> None:
        while not should_stop():
            try:
                item = result_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if isinstance(item, tuple) and len(item) >= 1 and item[0] == -1:
                break
            if isinstance(item, tuple) and len(item) == 3:
                fidx, hits, frame_b64 = item
            else:
                fidx, hits = item  # type: ignore[misc]
                frame_b64 = None

            matches: List[dict] = []
            try:
                for i, h in enumerate(hits or []):
                    try:
                        pid = str(h.id) if hasattr(h, 'id') else str(getattr(h, "id", f"unknown_{i}"))
                    except Exception:
                        pid = f"unknown_{i}"
                    try:
                        raw_score = float(h.distance) if hasattr(h, 'distance') else float(getattr(h, "distance", 0.0))
                    except Exception:
                        raw_score = 0.0
                    if -1.0 <= raw_score <= 1.0:
                        similarity = max(0.0, min(1.0, raw_score))
                    else:
                        similarity = max(0.0, min(1.0, (2.0 - raw_score) / 2.0))

                    payload = None
                    try:
                        raw_payload = getattr(h, "payload", None)
                        if raw_payload:
                            if isinstance(raw_payload, str):
                                payload = _json.loads(raw_payload)
                            else:
                                payload = raw_payload
                        else:
                            if hasattr(h, 'entity'):
                                entity = h.entity
                                if isinstance(entity, dict):
                                    payload = entity.get("payload", entity)
                                else:
                                    payload = str(entity)
                    except Exception:
                        payload = None

                    if similarity >= threshold:
                        if payload is not None:
                            try:
                                _json.dumps(payload)
                            except Exception:
                                payload = str(payload)
                        matches.append({
                            "id": pid,
                            "distance": raw_score,
                            "similarity": similarity,
                            "payload": payload,
                        })

                out = {"index": fidx, "matches": matches}
                if frame_b64:
                    out["image"] = frame_b64
                asyncio.run(broadcast("testing_matches", out))
            except RuntimeError as e:
                logger.warning(f"[TEST] Frame {fidx}: Failed to emit matches: {e}")
            except Exception as ex:
                logger.error(f"[TEST] Frame {fidx}: Error processing results: {ex}")

    return frame_queue, result_queue, reader, batcher, processor


