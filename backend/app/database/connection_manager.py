from __future__ import annotations

import asyncio
import logging
from typing import Optional

from .mongo import get_client as get_mongo_client, connect as mongo_connect, disconnect as mongo_disconnect
from .milvus import connect as milvus_connect, disconnect as milvus_disconnect
from pymongo.errors import PyMongoError
from pymilvus import utility


logger = logging.getLogger(__name__)


_monitor_task: Optional[asyncio.Task] = None


def _ping_mongo() -> bool:
    try:
        client = get_mongo_client()
        client.admin.command("ping")
        return True
    except Exception as exc:  # broad to catch network/transient errors
        logger.warning("MongoDB ping failed: %s", exc)
        return False


def _ping_milvus() -> bool:
    try:
        # list_collections is a cheap call and validates the connection
        utility.list_collections()
        return True
    except Exception as exc:
        logger.warning("Milvus ping failed: %s", exc)
        return False


async def _connect_with_retry(connect_fn, ping_fn, name: str, initial_delay: float = 0.5, max_delay: float = 30.0) -> None:
    delay = initial_delay
    while True:
        try:
            connect_fn()
            if ping_fn():
                logger.info("%s connection established", name)
                return
            else:
                raise RuntimeError(f"{name} ping failed after connect")
        except Exception as exc:
            logger.error("%s connection failed, retrying in %.1fs: %s", name, delay, exc)
            await asyncio.sleep(delay)
            delay = min(delay * 2.0, max_delay)


async def init_connections() -> None:
    # Attempt initial connections with retry until success
    await asyncio.gather(
        _connect_with_retry(mongo_connect, _ping_mongo, "MongoDB"),
        _connect_with_retry(milvus_connect, _ping_milvus, "Milvus"),
    )


async def _monitor_loop(poll_interval: float = 10.0) -> None:
    # Periodically ping; if disconnected, retry until reconnected
    while True:
        try:
            mongo_ok = _ping_mongo()
            if not mongo_ok:
                logger.info("Reconnecting to MongoDB...")
                await _connect_with_retry(mongo_connect, _ping_mongo, "MongoDB")

            milvus_ok = _ping_milvus()
            if not milvus_ok:
                logger.info("Reconnecting to Milvus...")
                await _connect_with_retry(milvus_connect, _ping_milvus, "Milvus")
        except Exception as exc:
            logger.error("Connection monitor loop error: %s", exc)
        await asyncio.sleep(poll_interval)


async def start_monitor() -> None:
    global _monitor_task
    if _monitor_task is None or _monitor_task.done():
        _monitor_task = asyncio.create_task(_monitor_loop())
        logger.info("Connection monitor started")


async def stop_monitor() -> None:
    global _monitor_task
    if _monitor_task is not None:
        _monitor_task.cancel()
        try:
            await _monitor_task
        except asyncio.CancelledError:
            pass
        _monitor_task = None
        logger.info("Connection monitor stopped")


async def shutdown() -> None:
    await stop_monitor()
    # Optional: close connections explicitly (clients typically handle shutdown gracefully)
    try:
        mongo_disconnect()
    except Exception:
        pass
    try:
        milvus_disconnect()
    except Exception:
        pass


