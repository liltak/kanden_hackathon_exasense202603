"""H100 GPU reconstruction API server.

Runs on the H100 machine, exposing VGGT inference and mesh processing
via HTTP endpoints with NDJSON streaming progress.

Usage:
    uv run uvicorn src.reconstruction.serve:app --host 0.0.0.0 --port 8001 --workers 1
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="ExaSense H100 Reconstruction API", version="0.1.0")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
E2E_RESULTS_DIR = PROJECT_ROOT / "data" / "e2e_results"

# GPU lock: only one reconstruction at a time to avoid OOM
_gpu_lock = asyncio.Lock()

# Store completed mesh paths for download
_result_paths: dict[str, Path] = {}


def _ndjson_line(task_id: str, step: str, progress: float, message: str) -> str:
    """Format a single NDJSON progress line."""
    return json.dumps(
        {"task_id": task_id, "step": step, "progress": progress, "message": message},
        ensure_ascii=False,
    ) + "\n"


def _run_pipeline(task_id: str, image_dir: Path, output_dir: Path, method: str) -> Path:
    """Run VGGT + mesh processing synchronously (called via asyncio.to_thread).

    Returns the path to the generated mesh file.
    """
    from .vggt_runner import run_vggt
    from .mesh_processor import process_reconstruction

    # Step 1: VGGT inference
    logger.info("Task %s: starting VGGT inference", task_id)
    result = run_vggt(
        image_dir=image_dir,
        output_dir=output_dir,
        device="cuda",
    )
    logger.info("Task %s: VGGT done, %d points", task_id, result.num_points)

    # Step 2: Mesh processing
    point_cloud_path = output_dir / "point_cloud.ply"
    mesh_path = output_dir / "mesh.ply"

    logger.info("Task %s: starting mesh processing", task_id)
    process_reconstruction(
        point_cloud_path=point_cloud_path,
        output_path=mesh_path,
        method="poisson",
    )
    logger.info("Task %s: mesh processing done", task_id)

    return mesh_path


@app.post("/reconstruct")
async def reconstruct(
    files: list[UploadFile] = File(...),
    method: str = Form("vggt"),
):
    """Run 3D reconstruction from uploaded images.

    Returns an NDJSON stream of progress updates.
    """
    task_id = uuid.uuid4().hex[:12]

    # Save uploaded images before entering the streaming response
    # (UploadFile objects become invalid after the request handler returns)
    tmp_base = Path(tempfile.mkdtemp(prefix=f"recon_{task_id}_"))
    image_dir = tmp_base / "images"
    image_dir.mkdir()
    output_dir = tmp_base / "output"
    output_dir.mkdir()

    for f in files:
        content = await f.read()
        fname = f.filename or f"image_{uuid.uuid4().hex[:4]}.jpg"
        (image_dir / fname).write_bytes(content)

    async def generate():
        yield _ndjson_line(task_id, "upload", 0.05, "Images received")

        try:
            async with _gpu_lock:
                yield _ndjson_line(task_id, "vggt", 0.15, "VGGT推論開始")

                mesh_path = await asyncio.to_thread(
                    _run_pipeline, task_id, image_dir, output_dir, method
                )

                yield _ndjson_line(task_id, "vggt", 0.50, "VGGT完了")
                yield _ndjson_line(task_id, "mesh", 0.55, "メッシュ生成中")
                yield _ndjson_line(task_id, "mesh", 0.80, "メッシュ完了")

                _result_paths[task_id] = mesh_path

            yield _ndjson_line(task_id, "done", 1.0, "完了")

        except Exception as e:
            logger.exception("Reconstruction failed for task %s", task_id)
            yield _ndjson_line(task_id, "error", 0.0, str(e))

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
    )


@app.get("/reconstruct/{task_id}/mesh")
async def get_mesh(task_id: str):
    """Download the reconstructed mesh PLY file."""
    mesh_path = _result_paths.get(task_id)
    if mesh_path is None or not mesh_path.exists():
        return {"error": "Mesh not found"}, 404
    return FileResponse(
        path=str(mesh_path),
        media_type="application/octet-stream",
        filename="mesh.ply",
    )


@app.get("/presets")
async def list_presets():
    """List available pre-built reconstruction datasets."""
    presets = []
    if not E2E_RESULTS_DIR.is_dir():
        return {"presets": presets}

    for d in sorted(E2E_RESULTS_DIR.iterdir()):
        if not d.is_dir():
            continue

        # Look for mesh.ply directly or under mesh/ subdirectory
        mesh_path = d / "mesh.ply"
        if not mesh_path.exists():
            mesh_path = d / "mesh" / "mesh.ply"
        if not mesh_path.exists():
            continue

        # Load metadata if available
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

    return {"presets": presets}


@app.get("/presets/{name}/mesh")
async def get_preset_mesh(name: str):
    """Download a pre-built mesh by dataset name."""
    d = E2E_RESULTS_DIR / name

    # Prevent path traversal
    if not d.resolve().is_relative_to(E2E_RESULTS_DIR.resolve()):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "Invalid preset name"}, status_code=400)

    mesh_path = d / "mesh.ply"
    if not mesh_path.exists():
        mesh_path = d / "mesh" / "mesh.ply"
    if not mesh_path.exists():
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "Mesh not found"}, status_code=404)

    return FileResponse(
        path=str(mesh_path),
        media_type="application/octet-stream",
        filename=f"{name}_mesh.ply",
    )


@app.get("/health")
async def health():
    return {"status": "ok", "service": "exasense-h100", "gpu": True}
