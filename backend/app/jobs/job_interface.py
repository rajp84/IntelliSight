from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
from datetime import datetime
import asyncio
import logging
from ..database.mongo import update_one
from ..socket.socket_manager import broadcast

logger = logging.getLogger(__name__)
class Job(ABC):
    @abstractmethod
    async def process(self, job_id: str, media_path: str) -> None:
        pass


def update_job_progress(
    job_id: str,
    progress_pct: float,
    step: float,
    total_steps: float,
    *,
    frames_processed: Optional[int] = None,
    preview_b64: Optional[str] = None,
    emit: bool = True,
) -> None:
    try:
        set_doc = {
            "status": "running",
            "progress": progress_pct,
            "step": step,
            "total_steps": total_steps,
            "updated_at": datetime.utcnow(),
        }
        if frames_processed is not None:
            set_doc["frames_processed"] = int(frames_processed)
        if preview_b64 is not None:
            set_doc["preview"] = preview_b64
        update_one("jobs", {"job_id": job_id}, {"$set": set_doc})
    except Exception as e:
        logger.error(f"Error updating job {job_id}: {e}")
        pass
    if emit:
        try:
            payload = {
                "job_id": job_id,
                "status": "running",
                "progress": progress_pct,
                "step": step,
                "total_steps": total_steps,
                "updated_at": datetime.utcnow().isoformat() + "Z",
            }
            if frames_processed is not None:
                payload["frames_processed"] = int(frames_processed)
            if preview_b64 is not None:
                payload["preview"] = preview_b64
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(broadcast("job_update", payload))
            except RuntimeError:
                # No running loop - we're in a background thread
                # Schedule the broadcast to run in the main event loop
                try:
                    import concurrent.futures
                    # Get the main event loop from the process worker
                    from ..workers.process_worker import get_main_loop
                    main_loop = get_main_loop()
                    if main_loop and not main_loop.is_closed():
                        asyncio.run_coroutine_threadsafe(broadcast("job_update", payload), main_loop)
                    else:
                        # Fallback: try to run in a new loop (less reliable)
                        asyncio.run(broadcast("job_update", payload))
                except Exception as e:
                    logger.error(f"Error broadcasting job update for job {job_id}: {e}")
        except Exception as e:
            logger.error(f"Error in update_job_progress for job {job_id}: {e}")
            pass


