"""Celery task for 3D reconstruction pipeline (Phase 1-2).

Runs on GPU workers: VGGT/COLMAP → point cloud → mesh generation.
Requires CUDA-enabled environment with torch, open3d, and optionally COLMAP.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from .celery_app import celery_app

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@celery_app.task(bind=True, name="exasense.run_reconstruction")
def run_reconstruction_task(
    self,
    image_dir: str,
    output_dir: str | None = None,
    method: str = "vggt",
    mesh_resolution: int = 50000,
) -> dict:
    """Run 3D reconstruction from images.

    Args:
        image_dir: Path to directory containing input images.
        output_dir: Path for output files. Auto-generated if None.
        method: Reconstruction method — 'vggt' (fast, GPU) or 'colmap' (accurate, slower).
        mesh_resolution: Target number of faces for mesh simplification.

    Returns:
        Dict with status, mesh_path, num_vertices, num_faces, elapsed_seconds.
    """
    t0 = time.time()

    if output_dir is None:
        output_dir = str(DATA_DIR / "reconstruction" / f"recon_{self.request.id[:8]}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    self.update_state(
        state="PROGRESS",
        meta={"step": "init", "progress": 0.0, "message": "Initializing reconstruction..."},
    )

    try:
        if method == "vggt":
            mesh_path = _run_vggt(self, image_dir, out_path, mesh_resolution)
        elif method == "colmap":
            mesh_path = _run_colmap(self, image_dir, out_path, mesh_resolution)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'vggt' or 'colmap'.")

        import trimesh

        mesh = trimesh.load(mesh_path, force="mesh")
        elapsed = time.time() - t0

        # Upload to MinIO if available
        object_key = None
        try:
            from ..api.storage import BUCKET_MESHES, upload_bytes

            mesh_bytes = mesh_path.read_bytes()
            object_key = f"reconstruction/{self.request.id[:8]}/mesh.ply"
            upload_bytes(BUCKET_MESHES, object_key, mesh_bytes)
        except Exception:
            pass

        return {
            "status": "complete",
            "mesh_path": str(mesh_path),
            "object_key": object_key,
            "num_vertices": len(mesh.vertices),
            "num_faces": len(mesh.faces),
            "surface_area_m2": float(mesh.area),
            "elapsed_seconds": elapsed,
        }

    except Exception as e:
        logger.exception("Reconstruction failed")
        return {"status": "failed", "error": str(e), "elapsed_seconds": time.time() - t0}


def _run_vggt(self, image_dir: str, out_path: Path, mesh_resolution: int) -> Path:
    """Run VGGT-based reconstruction (fast, single-pass)."""
    self.update_state(
        state="PROGRESS",
        meta={"step": "vggt_inference", "progress": 0.2, "message": "Running VGGT inference..."},
    )

    from ..reconstruction.vggt_pipeline import run_vggt

    point_cloud_path = out_path / "points.ply"
    run_vggt(image_dir, str(point_cloud_path))

    self.update_state(
        state="PROGRESS",
        meta={"step": "meshing", "progress": 0.6, "message": "Generating mesh from point cloud..."},
    )

    from ..reconstruction.mesh_processor import point_cloud_to_mesh

    mesh_path = out_path / "mesh.ply"
    point_cloud_to_mesh(str(point_cloud_path), str(mesh_path), target_faces=mesh_resolution)

    self.update_state(
        state="PROGRESS",
        meta={"step": "complete", "progress": 1.0, "message": "Reconstruction complete"},
    )

    return mesh_path


def _run_colmap(self, image_dir: str, out_path: Path, mesh_resolution: int) -> Path:
    """Run COLMAP-based reconstruction (accurate, multi-step)."""
    self.update_state(
        state="PROGRESS",
        meta={"step": "feature_extraction", "progress": 0.1, "message": "Extracting features..."},
    )

    from ..reconstruction.colmap_pipeline import run_colmap

    sparse_path = out_path / "sparse"
    dense_path = out_path / "dense"

    self.update_state(
        state="PROGRESS",
        meta={"step": "matching", "progress": 0.3, "message": "Matching features..."},
    )

    run_colmap(image_dir, str(sparse_path), str(dense_path))

    self.update_state(
        state="PROGRESS",
        meta={"step": "meshing", "progress": 0.7, "message": "Generating mesh..."},
    )

    from ..reconstruction.mesh_processor import point_cloud_to_mesh

    mesh_path = out_path / "mesh.ply"
    point_cloud_path = dense_path / "fused.ply"
    point_cloud_to_mesh(str(point_cloud_path), str(mesh_path), target_faces=mesh_resolution)

    self.update_state(
        state="PROGRESS",
        meta={"step": "complete", "progress": 1.0, "message": "Reconstruction complete"},
    )

    return mesh_path
