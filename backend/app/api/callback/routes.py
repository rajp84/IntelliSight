from __future__ import annotations

from ...service.system_service import get_configuration
from fastapi import APIRouter, Request
from pathlib import Path
from datetime import datetime
import uuid
import json


router = APIRouter()


@router.post("")
async def receive_callback(request: Request):
    body: bytes = await request.body()
    # Resolve backend root and ensure debug directory
    cfg = get_configuration()
    backend_root = Path(cfg.get("library_path"))
    debug_dir = backend_root / "debug"
    try:
        debug_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    fname = f"callback_{ts}_{uuid.uuid4().hex}.json"
    fpath = debug_dir / fname
    try:
        # Try to pretty-print JSON; if not JSON, write raw text
        try:
            data = await request.json()
            pretty = json.dumps(data, indent=2, ensure_ascii=False)
        except Exception:
            try:
                pretty = json.dumps(json.loads(body.decode("utf-8")), indent=2, ensure_ascii=False)
            except Exception:
                try:
                    pretty = body.decode("utf-8")
                except Exception:
                    pretty = body.decode("latin-1", errors="ignore")
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(pretty)
        return {"status": "ok", "path": str(fpath)}
    except Exception as ex:
        return {"status": "error", "error": str(ex)}


