from __future__ import annotations

import asyncio
import os
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime

from ..database.mongo import create_collection, get_collection, update_one, find_one
from ..jobs.job_interface import Job
from ..jobs.object_detection import ObjectDetection
from ..jobs.similarity_search import SimilaritySearch
from ..jobs.find_anything import FindAnything
from ..socket.socket_manager import broadcast
import requests


logger = logging.getLogger(__name__)

# Global reference to the main event loop for cross-thread communication
_main_loop: Optional[asyncio.AbstractEventLoop] = None

def get_main_loop() -> Optional[asyncio.AbstractEventLoop]:
    """Get the main event loop for cross-thread communication."""
    return _main_loop

class ProcessWorker:
    def __init__(self, *, num_workers: Optional[int] = None) -> None:
        # Configurable via env PROCESS_NUM_WORKERS or PROCESS_WORKER_CONCURRENCY, default 2
        if num_workers is None:
            env_val = os.getenv("PROCESS_NUM_WORKERS") or os.getenv("PROCESS_WORKER_CONCURRENCY") or "2"
            try:
                num_workers = int(env_val)
            except Exception:
                num_workers = 2
        self.num_workers: int = max(1, int(num_workers))
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        # If set, detection handlers will switch to this model before starting the next detection job
        self.requested_detection_model_path: Optional[Path] = None

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop_event.clear()
        # Set the main event loop for cross-thread communication
        global _main_loop
        _main_loop = asyncio.get_running_loop()
        self._task = asyncio.create_task(self._run())
        logger.info("ProcessWorker started with num workers=%d", self.num_workers)

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop_event.set()
        try:
            await asyncio.wait_for(self._task, timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("ProcessWorker stop timed out; cancelling task")
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass
        finally:
            self._task = None
            # Clear the main loop reference
            global _main_loop
            _main_loop = None
            logger.info("ProcessWorker stopped")

    async def _run(self) -> None:
        # Ensure collection and index
        create_collection("jobs")
        coll = get_collection("jobs")
        try:
            coll.create_index([("status", 1), ("created_at", 1)])
            coll.create_index([("job_id", 1)], unique=True)
        except Exception:
            pass

        # Spawn long-lived worker tasks (one ObjectDetection per worker)
        workers: List[asyncio.Task] = [
            asyncio.create_task(self._worker_loop(i)) for i in range(self.num_workers)
        ]
        try:
            # Run until stop requested
            await self._stop_event.wait()
        finally:
            for t in workers:
                t.cancel()
            try:
                await asyncio.gather(*workers, return_exceptions=True)
            except Exception:
                pass

    def _fetch_next_job(self) -> Optional[Dict[str, Any]]:
        try:
            # atomically claim one job in accepted state
            res = find_one(
                "jobs",
                filter={"status": {"$in": ["accepted", "queued"]}},
                sort=[("created_at", 1)],
            )
            if not res:
                return None
            job_id = res.get("job_id")
            if not job_id:
                return None
            upd = update_one(
                "jobs",
                {"job_id": job_id, "status": res.get("status")},
                {"$set": {"status": "running", "started_at":  datetime.utcnow().isoformat() + "Z"}},
            )
            if upd.get("modified", 0) == 0:
                return None
            # Notify UI that job moved to running
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(broadcast("job_update", {
                    "job_id": job_id,
                    "status": "running",
                    "started_at": datetime.utcnow().isoformat() + "Z",
                }))
            except Exception:
                pass
            return res
        except Exception as ex:
            logger.error("ProcessWorker fetch job failed: %s", ex)
            return None

    async def _worker_loop(self, worker_index: int) -> None:
        # Create one detector instance per worker
        current_job_id: Optional[str] = None

        # Per-worker instances of job handlers
        handlers: Dict[str, Job] = {
            "detection": ObjectDetection(),
            "similarity_search": SimilaritySearch(),
            "find_anything": FindAnything(),
        }

        while not self._stop_event.is_set():
            job = self._fetch_next_job()
            if job is None:
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    # Idle tick: if a model switch was requested, apply it now so we're ready before the next job
                    if self.requested_detection_model_path is not None:
                        try:
                            det = handlers.get("detection")
                            if det and getattr(det, "set_model_path_and_load", None):
                                current_mp = getattr(det, "model_path", None)
                                if str(current_mp) != str(self.requested_detection_model_path):
                                    target_path = self.requested_detection_model_path
                                    logger.info("(idle) Switching model to: %s", str(target_path))
                                    await asyncio.to_thread(det.set_model_path_and_load, target_path)
                                    logger.info("(idle) Model switched to: %s", str(target_path))
                            self.requested_detection_model_path = None
                        except Exception:
                            self.requested_detection_model_path = None
                continue
            job_id = str(job.get("job_id"))
            media_file = str(job.get("media_file"))
            current_job_id = job_id
            try:
                job_type = str(job.get("type") or "detection").strip()
                handler = handlers.get(job_type) or handlers["detection"]

                # If a new detection model has been requested, apply it before starting a detection job
                if job_type == "detection" and self.requested_detection_model_path is not None:
                    try:
                        det = handlers.get("detection")
                        if det and getattr(det, "set_model_path_and_load", None):
                            current_mp = getattr(det, "model_path", None)
                            if str(current_mp) != str(self.requested_detection_model_path):
                                target_path = self.requested_detection_model_path
                                logger.info("Switching model to: %s", str(target_path))
                                await asyncio.to_thread(det.set_model_path_and_load, target_path)
                                logger.info("Model switched to: %s", str(target_path))
                        self.requested_detection_model_path = None
                    except Exception:
                        # Keep the old model if switching fails
                        self.requested_detection_model_path = None

                # If the job document already specifies a model_path, ensure we load it before processing
                if job_type == "detection":
                    try:
                        det = handlers.get("detection")
                        doc = find_one("jobs", {"job_id": job_id}) or {}
                        jp = str(doc.get("model_path") or "").strip()
                        if jp:
                            from pathlib import Path as _P
                            p = _P(jp)
                            if p.exists() and p.is_file():
                                cur_mp = getattr(det, "model_path", None)
                                if str(cur_mp) != str(p):
                                    logger.info("Loading model from job document: %s", str(p))
                                    await asyncio.to_thread(det.set_model_path_and_load, p)
                    except Exception:
                        pass

                # Record the active model to the job document (after any switch above)
                if job_type == "detection":
                    try:
                        det = handlers.get("detection")
                        mp = str(getattr(det, "model_path")) if det and getattr(det, "model_path", None) else None
                        if mp:
                            mn = os.path.basename(mp)
                            update_one(
                                "jobs",
                                {"job_id": job_id},
                                {"$set": {"model_path": mp, "model_name": mn}},
                            )
                            try:
                                await broadcast("job_update", {"job_id": job_id, "model_path": mp, "model_name": mn})
                            except Exception:
                                pass
                    except Exception:
                        pass

                await asyncio.to_thread(lambda: asyncio.run(handler.process(job_id, media_file)))
                
                update_one(
                    "jobs",
                    {"job_id": job_id},
                    {
                        "$set": {"status": "completed", "completed_at": datetime.utcnow().isoformat() + "Z", "progress": 100.0},
                        "$unset": {"preview": ""},
                    },
                )
                try:
                    await broadcast("job_update", {
                        "job_id": job_id,
                        "status": "completed",
                        "progress": 100.0,
                        "completed_at": datetime.utcnow().isoformat() + "Z",
                    })
                except Exception:
                    pass
                # Best-effort success callback for any job type
                try:
                    await asyncio.to_thread(self._post_completion_callback, job_id, handler)
                except Exception:
                    pass
                logger.info("Worker %d processed job %s (%s)", worker_index, job_id, media_file)
            except Exception as ex:
                try:
                    update_one(
                        "jobs",
                        {"job_id": job_id},
                        {"$set": {"status": "failed", "error": str(ex), "failed_at": datetime.utcnow()}},
                    )
                except Exception:
                    pass
                try:
                    await broadcast("job_update", {
                        "job_id": job_id,
                        "status": "failed",
                        "error": str(ex),
                        "failed_at": datetime.utcnow().isoformat() + "Z",
                    })
                except Exception:
                    pass
                # Best-effort error callback for any job type
                try:
                    await asyncio.to_thread(self._post_error_callback, job_id)
                except Exception:
                    pass
                logger.error("Worker %d job failed: %s", worker_index, ex)
            finally:
                current_job_id = None

    # Callback helpers
    
    def _post_completion_callback(self, job_id: str, handler: Job) -> None:
        try:
            coll_jobs = get_collection("jobs")
            job_doc = coll_jobs.find_one({"job_id": job_id}, {"callback_url": 1, "fps": 1, "frame_width": 1, "frame_height": 1, "model_path": 1, "model_name": 1}) or {}
            callback_url = str(job_doc.get("callback_url") or "").strip()
            if not callback_url:
                return
            # Build results via handler if available
            results: List[Dict[str, Any]] = []
            try:
                build = getattr(handler, "_build_results", None)
                if callable(build):
                    results = build(
                        job_id=job_id,
                        fps=float(job_doc.get("fps") or 0.0),
                        frame_w=int(job_doc.get("frame_width") or 0),
                        frame_h=int(job_doc.get("frame_height") or 0),
                        model_name=str(job_doc.get("model_name") or ""),
                        model_path=str(job_doc.get("model_path") or ""),
                    ) or []
            except Exception:
                results = []
            payload = {
                "job_id": job_id,
                "progress": 100,
                "status": "Finished",
                "result": results,
            }
            try:
                logger.info(f"Posting callback to {callback_url} with payload: {payload}")
                requests.post(callback_url, json=payload, timeout=10)
            except Exception as e:
                logger.error(f"Failed to post callback to {callback_url}: {e}")
        except Exception as e:
            logger.error(f"Failed to post callback: {e}")
            pass

    def _post_error_callback(self, job_id: str) -> None:
        try:
            coll_jobs = get_collection("jobs")
            job_doc = coll_jobs.find_one({"job_id": job_id}, {"callback_url": 1}) or {}
            callback_url = str(job_doc.get("callback_url") or "").strip()
            if not callback_url:
                return
            payload = {
                "job_id": job_id,
                "progress": 100,
                "status": "Failed",
                "result": [],
            }
            try:
                logger.info(f"Posting error callback to {callback_url} with payload: {payload}")
                requests.post(callback_url, json=payload, timeout=10)
            except Exception as e:
                logger.error(f"Failed to post error callback to {callback_url}: {e}")
        except Exception as e:
            logger.error(f"Failed to post error callback: {e}")
            pass

# Singleton accessor for app lifecycle
_worker: Optional[ProcessWorker] = None


def get_worker() -> ProcessWorker:
    global _worker
    if _worker is None:
        _worker = ProcessWorker()
    return _worker


