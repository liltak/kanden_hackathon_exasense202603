"""FastAPI backend for ExaSense.

Provides REST API + WebSocket endpoints for the factory energy optimization pipeline.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .database import close_db, init_db
from .routes import auth, chat, mesh, reconstruction, report, simulation, solar_animation
from .schemas import ConfigResponse
from .ws import manager as ws_manager

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("ExaSense API starting up")
    await init_db()
    await ws_manager.start_listener()
    try:
        from .storage import ensure_buckets

        ensure_buckets()
    except Exception:
        logger.warning("MinIO not available, skipping bucket setup")
    yield
    await ws_manager.stop_listener()
    await close_db()
    logger.info("ExaSense API shutting down")


app = FastAPI(
    title="ExaSense API",
    description="Factory Energy Optimization Solution API",
    version="0.3.0",
    lifespan=lifespan,
)

# CORS — allow configured origins or fall back to dev defaults
_cors_origins = os.environ.get("CORS_ORIGINS", "").split(",")
_cors_origins = [o.strip() for o in _cors_origins if o.strip()]
if not _cors_origins:
    _cors_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(auth.router)
app.include_router(simulation.router)
app.include_router(mesh.router)
app.include_router(chat.router)
app.include_router(report.router)
app.include_router(solar_animation.router)
app.include_router(reconstruction.router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "exasense", "version": "0.3.0"}


@app.get("/api/config", response_model=ConfigResponse)
async def get_config():
    """Return simulation configuration as JSON."""
    config_path = CONFIGS_DIR / "solar_params.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return ConfigResponse(**config)


@app.websocket("/api/ws/simulation/{task_id}")
async def ws_simulation_progress(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for real-time simulation progress."""
    await ws_manager.connect(task_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(task_id, websocket)
