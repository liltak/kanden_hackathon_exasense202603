"""3D Reconstruction API routes.

Orchestrates reconstruction on a remote H100 GPU server via HTTP API:
  1. POST images to H100 as multipart
  2. Stream NDJSON progress updates
  3. GET result mesh from H100
  4. Load into mesh cache
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import uuid
from pathlib import Path

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from ..schemas import ReconstructionStatus
from ..ws import manager as ws_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/reconstruction", tags=["reconstruction"])

H100_API_URL = os.environ.get("H100_API_URL", "http://h100:8001")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
E2E_RESULTS_DIR = PROJECT_ROOT / "data" / "e2e_results"

# In-memory task store
_tasks: dict[str, ReconstructionStatus] = {}


def _update_task(
    task_id: str,
    *,
    status: str | None = None,
    progress: float | None = None,
    step: str | None = None,
    message: str | None = None,
    mesh_id: str | None = None,
) -> None:
    """Update in-memory task state."""
    task = _tasks[task_id]
    if status is not None:
        task.status = status
    if progress is not None:
        task.progress = progress
    if step is not None:
        task.step = step
    if message is not None:
        task.message = message
    if mesh_id is not None:
        task.mesh_id = mesh_id


async def _run_reconstruction_api(
    task_id: str,
    image_files: list[tuple[str, bytes]],
    method: str,
    output_format: str = "mesh",
) -> None:
    """Execute reconstruction pipeline via H100 HTTP API."""
    timeout = httpx.Timeout(600.0, connect=30.0)

    try:
        async with httpx.AsyncClient(base_url=H100_API_URL, timeout=timeout) as client:
            # Step 1: POST images to H100 and stream NDJSON progress
            _update_task(task_id, status="running", progress=0.05, step="upload", message="画像をH100に送信中...")
            await ws_manager.send_progress(task_id, "upload", 0.05, "画像をH100に送信中...")

            # Build multipart files
            multipart_files = [
                ("files", (fname, data, "image/jpeg"))
                for fname, data in image_files
            ]

            async with client.stream(
                "POST",
                "/reconstruct",
                files=multipart_files,
                data={"method": method, "output_format": output_format},
            ) as response:
                response.raise_for_status()

                h100_task_id: str | None = None

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("Skipping invalid NDJSON line: %s", line)
                        continue

                    h100_task_id = event.get("task_id", h100_task_id)
                    step = event.get("step", "")
                    progress = event.get("progress", 0.0)
                    message = event.get("message", "")

                    _update_task(task_id, progress=progress, step=step, message=message)
                    await ws_manager.send_progress(task_id, step, progress, message)

                    if step == "error":
                        raise RuntimeError(message)

            # Step 2: Download result from H100
            if h100_task_id is None:
                raise RuntimeError("No task_id received from H100 stream")

            mesh_id = f"recon_{task_id}"

            from .mesh import _mesh_cache

            if output_format == "glb":
                # GLB point cloud: download and cache raw bytes
                _update_task(task_id, progress=0.90, step="download", message="GLBをダウンロード中...")
                await ws_manager.send_progress(task_id, "download", 0.90, "GLBをダウンロード中...")

                glb_response = await client.get(f"/reconstruct/{h100_task_id}/glb")
                glb_response.raise_for_status()

                _mesh_cache[mesh_id] = {
                    "glb_bytes": glb_response.content,
                    "filename": f"recon_{task_id}.glb",
                }
            else:
                # Mesh PLY: download, load, and cache
                _update_task(task_id, progress=0.90, step="download", message="メッシュをダウンロード中...")
                await ws_manager.send_progress(task_id, "download", 0.90, "メッシュをダウンロード中...")

                mesh_response = await client.get(f"/reconstruct/{h100_task_id}/mesh")
                mesh_response.raise_for_status()

                _update_task(task_id, progress=0.95, step="load", message="メッシュをロード中...")
                await ws_manager.send_progress(task_id, "load", 0.95, "メッシュをロード中...")

                import trimesh

                mesh = trimesh.load(
                    io.BytesIO(mesh_response.content),
                    file_type="ply",
                    force="mesh",
                )
                _mesh_cache[mesh_id] = {"mesh": mesh, "filename": f"recon_{task_id}.ply"}

            _update_task(
                task_id,
                status="complete",
                progress=1.0,
                step="done",
                message="3D復元完了",
                mesh_id=mesh_id,
            )
            await ws_manager.send_progress(task_id, "done", 1.0, "3D復元完了")

    except Exception as e:
        logger.exception("Reconstruction failed for task %s", task_id)
        _update_task(task_id, status="failed", message=str(e))
        await ws_manager.send_progress(task_id, "error", 0, str(e))


def _find_local_preset_mesh(preset_name: str) -> Path | None:
    """Find a preset mesh PLY in local e2e_results directory."""
    d = E2E_RESULTS_DIR / preset_name
    if not d.is_dir():
        return None
    for candidate in [d / "mesh.ply", d / "mesh" / "mesh.ply"]:
        if candidate.exists():
            return candidate
    return None


def _scan_local_presets() -> list[dict]:
    """Scan local data/e2e_results/ for available preset datasets."""
    presets = []
    if not E2E_RESULTS_DIR.is_dir():
        return presets

    for d in sorted(E2E_RESULTS_DIR.iterdir()):
        if not d.is_dir():
            continue

        mesh_path = _find_local_preset_mesh(d.name)
        if mesh_path is None:
            continue

        meta = {}
        meta_path = d / "h100_e2e_results.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                pass

        mesh_info = meta.get("mesh", {})
        presets.append({
            "name": d.name,
            "n_faces": mesh_info.get("n_faces", 0),
            "n_vertices": mesh_info.get("n_vertices", 0),
            "surface_area_m2": mesh_info.get("surface_area_m2", 0),
        })

    return presets


async def _load_preset_mesh(task_id: str, preset_name: str) -> None:
    """Load a pre-built mesh — try H100 API first, fall back to local file."""
    try:
        _update_task(task_id, status="running", progress=0.3, step="download", message="メッシュを読み込み中...")
        await ws_manager.send_progress(task_id, "download", 0.3, "メッシュを読み込み中...")

        mesh_bytes: bytes | None = None

        # Try H100 API first
        try:
            timeout = httpx.Timeout(60.0, connect=5.0)
            async with httpx.AsyncClient(base_url=H100_API_URL, timeout=timeout) as client:
                resp = await client.get(f"/presets/{preset_name}/mesh")
                resp.raise_for_status()
                mesh_bytes = resp.content
                logger.info("Preset %s loaded from H100 API", preset_name)
        except Exception:
            logger.info("H100 API unavailable, trying local file for preset %s", preset_name)

        # Fall back to local file
        if mesh_bytes is None:
            local_path = _find_local_preset_mesh(preset_name)
            if local_path is None:
                raise FileNotFoundError(f"Preset '{preset_name}' not found locally or on H100")
            mesh_bytes = local_path.read_bytes()
            logger.info("Preset %s loaded from local file: %s", preset_name, local_path)

        _update_task(task_id, progress=0.7, step="load", message="メッシュをロード中...")
        await ws_manager.send_progress(task_id, "load", 0.7, "メッシュをロード中...")

        import trimesh

        mesh = trimesh.load(
            io.BytesIO(mesh_bytes),
            file_type="ply",
            force="mesh",
        )
        mesh_id = f"preset_{task_id}"

        from .mesh import _mesh_cache

        _mesh_cache[mesh_id] = {"mesh": mesh, "filename": f"{preset_name}_mesh.ply"}

        _update_task(
            task_id,
            status="complete",
            progress=1.0,
            step="done",
            message="プリセットメッシュ読み込み完了",
            mesh_id=mesh_id,
        )
        await ws_manager.send_progress(task_id, "done", 1.0, "プリセットメッシュ読み込み完了")

    except Exception as e:
        logger.exception("Preset load failed for task %s", task_id)
        _update_task(task_id, status="failed", message=str(e))
        await ws_manager.send_progress(task_id, "error", 0, str(e))


@router.get("/presets")
async def list_presets():
    """List available pre-built reconstruction datasets.

    Tries H100 API first, falls back to scanning local data/e2e_results/.
    """
    # Try H100 API
    try:
        timeout = httpx.Timeout(10.0, connect=3.0)
        async with httpx.AsyncClient(base_url=H100_API_URL, timeout=timeout) as client:
            resp = await client.get("/presets")
            resp.raise_for_status()
            return resp.json()
    except Exception:
        logger.info("H100 API unavailable for presets, using local scan")

    # Fall back to local scan
    return {"presets": _scan_local_presets()}


@router.post("/load-preset", response_model=ReconstructionStatus)
async def load_preset(name: str = Form(...)):
    """Load a pre-built mesh from H100 by dataset name."""
    task_id = f"preset_{uuid.uuid4().hex[:8]}"

    _tasks[task_id] = ReconstructionStatus(
        task_id=task_id,
        status="pending",
        progress=0.0,
        message="プリセット読み込み開始...",
    )

    asyncio.create_task(_load_preset_mesh(task_id, name))

    return _tasks[task_id]


@router.post("/start", response_model=ReconstructionStatus)
async def start_reconstruction(
    files: list[UploadFile] = File(...),
    method: str = Form("vggt"),
    output_format: str = Form("mesh"),
):
    """Start 3D reconstruction from uploaded images.

    - **files**: Multiple image files (JPEG/PNG)
    - **method**: Reconstruction method — 'vggt' or 'colmap'
    - **output_format**: 'mesh' (Poisson surface) or 'glb' (textured point cloud)
    """
    if not files:
        raise HTTPException(status_code=400, detail="No images provided")

    if method not in ("vggt", "colmap"):
        raise HTTPException(status_code=400, detail="method must be 'vggt' or 'colmap'")

    if output_format not in ("mesh", "glb"):
        raise HTTPException(status_code=400, detail="output_format must be 'mesh' or 'glb'")

    task_id = f"recon_{uuid.uuid4().hex[:8]}"

    # Read all file contents before passing to background task
    # (UploadFile objects become invalid after the request handler returns)
    image_files: list[tuple[str, bytes]] = []
    for f in files:
        content = await f.read()
        fname = f.filename or f"image_{uuid.uuid4().hex[:4]}.jpg"
        image_files.append((fname, content))

    _tasks[task_id] = ReconstructionStatus(
        task_id=task_id,
        status="pending",
        progress=0.0,
        message="キューに追加されました",
    )

    asyncio.create_task(_run_reconstruction_api(task_id, image_files, method, output_format))

    return _tasks[task_id]


@router.get("/{task_id}", response_model=ReconstructionStatus)
async def get_reconstruction_status(task_id: str):
    """Poll reconstruction task status."""
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return _tasks[task_id]
