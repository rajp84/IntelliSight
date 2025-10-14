import asyncio
import traceback
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2

from ..service.embedder_service import is_model_loaded, preload_model, embed_image
from ..service.system_service import get_configuration
from ..database.milvus import search_embeddings
from ..socket.socket_manager import broadcast
from ..workers.testing_workers import create_testing_threads

logger = logging.getLogger(__name__)


class _TestingState:
    def __init__(self):
        self._stop_event = threading.Event()
        self._status: Dict[str, Any] = {
            "state": "idle",
            "progress": 0,
            "message": "",
            "started_at": None,
            "ended_at": None,
            "fps": None,
        }

    def is_running(self) -> bool:
        return self._status["state"] == "running"

    def _set_status(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if v is not None:
                self._status[k] = v

    def status(self) -> Dict[str, Any]:
        return self._status.copy()

    def request_stop(self) -> None:
        self._stop_event.set()

    def _should_stop(self) -> bool:
        return self._stop_event.is_set()

    def start(self, *, path: str, threshold: float = 0.7, batch_size: int = 4) -> bool:
        if self.is_running():
            return False

        # Resolve file path under library root
        cfg = get_configuration()
        library_root = cfg.get("library_path") or ""
        resolved: Optional[Path] = None
        try:
            base = Path(library_root).expanduser().resolve() if library_root else None
            candidate = Path(path)
            if base is not None:
                cand_resolved = (base / candidate.as_posix().lstrip("/\\")).resolve()
                if str(cand_resolved).startswith(str(base)):
                    resolved = cand_resolved
            else:
                resolved = candidate.resolve()
        except Exception:
            resolved = None
        if resolved is None or not resolved.exists():
            return False

        self._stop_event.clear()
        self._set_status(state="running", progress=0, message="Testing started", started_at=time.time(), ended_at=None)
        try:
            asyncio.run(broadcast("testing_status", self.status()))
        except RuntimeError:
            pass

        # Ensure embedder is ready
        try:
            if not is_model_loaded():
                cfg = get_configuration()
                hf_token = cfg.get("hf_token") or None
                model_id = cfg.get("dinov3_model") or "facebook/dinov3-vitb16-pretrain-lvd1689m"
                preload_model(model_id=model_id, hf_token=hf_token)
        except Exception:
            pass

        def _runner() -> None:
            try:
                logger.info(f"[TEST] Starting testing for path: {resolved}, threshold: {threshold}, batch_size: {batch_size}")
                cap: Optional[cv2.VideoCapture] = None
                total = 0
                is_video = False
                
                # Open as video if applicable
                if resolved.is_file() and resolved.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm", ".mpg", ".mpeg"}:
                    is_video = True
                    logger.info(f"[TEST] Opening video file: {resolved}")
                    cap = cv2.VideoCapture(str(resolved))
                    if not cap.isOpened():
                        raise RuntimeError(f"Failed to open video: {resolved}")
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
                    logger.info(f"[TEST] Video opened successfully, total frames: {total}")
                    fps_timer_t0 = time.time()
                    fps_counter = 0
                else:
                    logger.info(f"[TEST] Processing as single image: {resolved}")

                frame_queue, result_queue, reader, batcher, processor = create_testing_threads(
                    resolved_path=resolved,
                    is_video=is_video,
                    cap=cap,
                    total_frames=total,
                    batch_size=batch_size,
                    should_stop=self._should_stop,
                    set_status=self._set_status,
                    broadcast=broadcast,
                    embed_image=lambda pil: embed_image(pil, normalize=True),
                    search_embeddings=lambda col, embs, k: search_embeddings(col, embs, top_k=k),
                    threshold=threshold,
                    flush_interval_sec=0.40,
                )

                # Launch threads
                reader_t = threading.Thread(target=reader, name="testing_reader", daemon=True)
                batcher_t = threading.Thread(target=batcher, name="testing_batcher", daemon=True)
                processor_t = threading.Thread(target=processor, name="testing_processor", daemon=True)

                reader_t.start()
                batcher_t.start()
                processor_t.start()

                # Wait for reader and batcher to finish, then signal processor
                reader_t.join()
                batcher_t.join()
                # Ensure processor finishes
                processor_t.join(timeout=2.0)

                # Clean up
                if cap is not None:
                    cap.release()

                # Mark as completed
                self._set_status(
                    state="completed",
                    progress=100,
                    message=f"Testing completed successfully",
                    ended_at=time.time(),
                )
                try:
                    asyncio.run(broadcast("testing_status", self.status()))
                except RuntimeError:
                    pass

                logger.info(f"[TEST] Testing completed successfully")

            except Exception as ex:
                logger.error(f"[TEST] Testing failed: {ex}")
                logger.debug(f"[TEST] Traceback: {traceback.format_exc()}")
                self._set_status(
                    state="failed",
                    message=f"Testing failed: {ex}",
                    ended_at=time.time(),
                )
                try:
                    asyncio.run(broadcast("testing_status", self.status()))
                except RuntimeError:
                    pass
            finally:
                self._stop_event.clear()

        t = threading.Thread(target=_runner, name="testing", daemon=True)
        t.start()
        return True


_STATE = _TestingState()


def start_test(*, path: str, threshold: float = 0.7, batch_size: int = 4) -> bool:
    """Start a test """
    return _STATE.start(path=path, threshold=threshold, batch_size=batch_size)


def testing_status() -> Dict[str, Any]:
    """Get the current status """
    return _STATE.status()


def terminate_test() -> None:
    """Terminate the current testing process """
    _STATE.request_stop()
