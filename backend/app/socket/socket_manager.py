from __future__ import annotations

import socketio


# Socket.IO async server (ASGI) - allow same-origin and dev servers
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",
    logger=False,
    engineio_logger=False,
)


@sio.event
async def connect(sid, environ):
    # You can add auth checks here using environ
    await sio.save_session(sid, {"rooms": set()})


@sio.event
async def disconnect(sid):
    pass


@sio.event
async def ping(sid, data=None):
    await sio.emit("pong", {"ok": True}, to=sid)


async def broadcast(event: str, data):
    """Emit an event to all connected clients."""
    await sio.emit(event, data)


async def emit_to_sid(sid: str, event: str, data):
    await sio.emit(event, data, to=sid)


