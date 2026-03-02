"""Bridge between API layer and VLM inference pipeline.

Provides GPU detection, singleton VLMPipeline management,
and heatmap PNG rendering for VLM image input.
"""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.vlm.inference import VLMPipeline

logger = logging.getLogger(__name__)

_gpu_available: bool | None = None
_pipeline: VLMPipeline | None = None


def has_gpu() -> bool:
    """Check if CUDA GPU is available. Result is cached after first call."""
    global _gpu_available
    if _gpu_available is not None:
        return _gpu_available
    try:
        import torch

        _gpu_available = torch.cuda.is_available()
    except ImportError:
        _gpu_available = False
    logger.info("GPU available: %s", _gpu_available)
    return _gpu_available


def get_pipeline() -> VLMPipeline | None:
    """Return singleton VLMPipeline (lazy-loaded). Returns None on CPU."""
    global _pipeline
    if not has_gpu():
        return None
    if _pipeline is not None:
        return _pipeline

    logger.info("Loading VLM pipeline (first call)...")
    from src.vlm.inference import VLMPipeline as _Cls
    from src.vlm.model_loader import ModelConfig

    _pipeline = _Cls(model_config=ModelConfig(quantize_4bit=True))
    _pipeline.load()
    logger.info("VLM pipeline loaded")
    return _pipeline


def render_heatmap_png(mesh, irradiance_results) -> bytes:
    """Render a top-down irradiance heatmap as PNG bytes.

    Args:
        mesh: trimesh mesh (with .vertices, .faces attributes).
        irradiance_results: list of FaceIrradiance dataclasses or dicts.

    Returns:
        PNG image bytes.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.collections import PolyCollection

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    # Build per-face irradiance lookup
    face_values = np.zeros(len(faces), dtype=np.float64)
    for r in irradiance_results:
        fid = r.face_id if hasattr(r, "face_id") else r["face_id"]
        val = (
            r.annual_irradiance_kwh_m2
            if hasattr(r, "annual_irradiance_kwh_m2")
            else r["annual_irradiance_kwh_m2"]
        )
        if fid < len(face_values):
            face_values[fid] = val

    # Top-down projection (XY plane)
    polys = vertices[faces][:, :, :2]  # (n_faces, 3_verts, 2_xy)

    vmin, vmax = face_values.min(), max(face_values.max(), 1.0)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8, 8))
    coll = PolyCollection(
        polys,
        array=face_values,
        cmap="YlOrRd",
        norm=norm,
        edgecolors="grey",
        linewidths=0.3,
    )
    ax.add_collection(coll)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_title("Annual Irradiance Heatmap (kWh/m²)")
    fig.colorbar(coll, ax=ax, label="kWh/m²/year")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
