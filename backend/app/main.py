from fastapi import FastAPI, Request, responses
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware
from pathlib import Path
import logging

from .socket.socket_manager import sio
import socketio

fastapi_app = FastAPI(title="AI+ GUI Template App")
app = socketio.ASGIApp(sio, other_asgi_app=fastapi_app)

# Ensure console logging for our app loggers
root_logger = logging.getLogger()
if not root_logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
logging.getLogger("app").setLevel(logging.INFO)
# Enable debug logging for testing and Milvus
logging.getLogger("app.service.test_service_batched").setLevel(logging.DEBUG)
logging.getLogger("app.database.milvus").setLevel(logging.DEBUG)

# === DEV ONLY CORS (disable in prod if serving same origin) ===
# Set env flag to toggle this if you like
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:4200"],  # Angular dev server
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# GZip for static & API responses
fastapi_app.add_middleware(GZipMiddleware, minimum_size=1024)

# ---- API routes under /api ----
from .api.routes import router as base_router
from .api.example_1.routes import router as example_1_router
from .api.example_2.routes import router as example_2_router
from .api.system.routes import router as system_router
from .api.train.routes import router as train_router
from .api.test.routes import router as test_router
from .database.connection_manager import init_connections, start_monitor, shutdown
from .service.system_service import start_system_stats_broadcast

ROUTER_PREFIX = "/api"
fastapi_app.include_router(base_router, prefix=ROUTER_PREFIX)
fastapi_app.include_router(example_1_router, prefix=ROUTER_PREFIX + "/example1")
fastapi_app.include_router(example_2_router, prefix=ROUTER_PREFIX + "/example2")
fastapi_app.include_router(system_router, prefix=ROUTER_PREFIX)
fastapi_app.include_router(train_router, prefix=ROUTER_PREFIX + "/train")
fastapi_app.include_router(test_router, prefix=ROUTER_PREFIX + "/test")

# ---- Startup/Shutdown: DB Connections ----
@fastapi_app.on_event("startup")
async def _on_startup():
    logging.getLogger(__name__).info("Initializing database connections...")
    await init_connections()
    await start_monitor()
    # start system stats broadcaster
    try:
        start_system_stats_broadcast()
    except Exception as ex:
        logging.getLogger(__name__).warning("Failed to start system stats broadcast: %s", ex)


@fastapi_app.on_event("shutdown")
async def _on_shutdown():
    logging.getLogger(__name__).info("Shutting down connection monitor and databases...")
    await shutdown()

# ---- Static (Angular) files ----
# In prod, we'll copy Angular dist to /app/static
static_dir = Path(__file__).resolve().parent.parent / "static"
if static_dir.exists():
    fastapi_app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

# ---- SPA fallback: return index.html for unmatched non-/api routes ----
# (StaticFiles(html=True) already serves index.html for 404 inside that mount,
# but if you want explicit fallback, you can add the below.)
@fastapi_app.middleware("http")
async def spa_fallback(request: Request, call_next):
    if request.url.path.startswith("/api"):
        return await call_next(request)
    response = await call_next(request)
    # If not found by StaticFiles, send index.html so Angular router can handle it
    if response.status_code == 404 and static_dir.exists():
        index = static_dir / "index.html"
        if index.is_file():
            return responses.FileResponse(index)
    return response
