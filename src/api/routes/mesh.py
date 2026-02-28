"""Mesh management API routes."""

from __future__ import annotations

import io
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models import MeshFile
from ..schemas import MeshInfo

router = APIRouter(prefix="/api/mesh", tags=["mesh"])

# In-memory cache for loaded mesh objects (trimesh can't serialize to DB)
_mesh_cache: dict[str, dict] = {}


def _try_store_minio(mesh_id: str, data: bytes, suffix: str) -> str | None:
    """Upload mesh to MinIO, returns object key or None if unavailable."""
    try:
        from ..storage import BUCKET_MESHES, upload_bytes

        object_key = f"meshes/{mesh_id}.{suffix}"
        upload_bytes(BUCKET_MESHES, object_key, data, content_type="application/octet-stream")
        return object_key
    except Exception:
        return None


@router.post("/upload", response_model=MeshInfo)
async def upload_mesh(file: UploadFile, db: AsyncSession = Depends(get_db)):
    """Upload PLY/OBJ/STL/GLB mesh file."""
    import trimesh

    content = await file.read()
    suffix = file.filename.rsplit(".", 1)[-1].lower() if file.filename else "obj"

    try:
        mesh = trimesh.load(
            io.BytesIO(content),
            file_type=suffix,
            force="mesh",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load mesh: {e}")

    mesh_id = str(uuid.uuid4())[:8]

    # Upload to MinIO if available
    object_key = _try_store_minio(mesh_id, content, suffix)

    # Save metadata to DB
    mesh_row = MeshFile(
        id=mesh_id,
        filename=file.filename or f"upload.{suffix}",
        object_key=object_key or f"local:{mesh_id}",
        num_vertices=len(mesh.vertices),
        num_faces=len(mesh.faces),
        surface_area_m2=float(mesh.area),
        bounds_min=mesh.bounds[0].tolist(),
        bounds_max=mesh.bounds[1].tolist(),
    )
    db.add(mesh_row)
    await db.flush()

    # Cache in memory for fast access
    _mesh_cache[mesh_id] = {"mesh": mesh, "filename": file.filename}

    # Also store in _sim_store for simulation use
    from .. import _sim_store

    _sim_store["_uploaded_mesh"] = mesh

    return MeshInfo(
        mesh_id=mesh_id,
        num_vertices=len(mesh.vertices),
        num_faces=len(mesh.faces),
        surface_area_m2=float(mesh.area),
        bounds_min=mesh.bounds[0].tolist(),
        bounds_max=mesh.bounds[1].tolist(),
        download_url=f"/api/mesh/{mesh_id}/glb",
    )


def _mesh_to_glb(mesh, irradiance_results=None) -> bytes:
    """Convert trimesh mesh to GLB with optional irradiance heatmap baked as vertex colors."""
    import numpy as np
    import trimesh as _trimesh

    if irradiance_results:
        import matplotlib.cm as cm

        values = np.zeros(len(mesh.faces))
        for r in irradiance_results:
            fid = r.face_id if hasattr(r, "face_id") else r["face_id"]
            val = r.annual_irradiance_kwh_m2 if hasattr(r, "annual_irradiance_kwh_m2") else r["annual_irradiance_kwh_m2"]
            if fid < len(values):
                values[fid] = val

        vmin, vmax = values.min(), max(values.max(), 1)
        norm = (values - vmin) / (vmax - vmin)
        colors_rgba = (cm.YlOrRd(norm) * 255).astype(np.uint8)
        mesh.visual = _trimesh.visual.ColorVisual(face_colors=colors_rgba)
    else:
        nz = mesh.face_normals[:, 2]
        colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
        for i, z in enumerate(nz):
            if z > 0.5:
                colors[i] = [66, 165, 245, 255]
            elif abs(z) < 0.1:
                colors[i] = [144, 164, 174, 255]
            else:
                colors[i] = [141, 110, 99, 255]
        mesh.visual = _trimesh.visual.ColorVisual(face_colors=colors)

    # Z-up to Y-up conversion for Three.js
    import numpy as np

    transform = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)
    mesh_copy = mesh.copy()
    mesh_copy.apply_transform(transform)

    buf = io.BytesIO()
    scene = _trimesh.Scene(mesh_copy)
    scene.export(buf, file_type="glb")
    return buf.getvalue()


@router.get("/{mesh_id}/glb")
async def get_mesh_glb(mesh_id: str, db: AsyncSession = Depends(get_db)):
    """Get mesh as GLB (with face-normal coloring)."""
    entry = _mesh_cache.get(mesh_id)
    if not entry:
        # Check DB + MinIO
        mesh_row = await db.get(MeshFile, mesh_id)
        if not mesh_row:
            raise HTTPException(status_code=404, detail="Mesh not found")
        if mesh_row.object_key and not mesh_row.object_key.startswith("local:"):
            try:
                import trimesh

                from ..storage import BUCKET_MESHES, download_bytes

                data = download_bytes(BUCKET_MESHES, mesh_row.object_key)
                suffix = mesh_row.object_key.rsplit(".", 1)[-1]
                mesh = trimesh.load(io.BytesIO(data), file_type=suffix, force="mesh")
                entry = {"mesh": mesh, "filename": mesh_row.filename}
                _mesh_cache[mesh_id] = entry
            except Exception:
                raise HTTPException(status_code=404, detail="Mesh file not accessible")
        else:
            raise HTTPException(status_code=404, detail="Mesh not in cache")

    glb_bytes = _mesh_to_glb(entry["mesh"])
    return Response(content=glb_bytes, media_type="model/gltf-binary")


@router.get("/{mesh_id}/irradiance")
async def get_mesh_irradiance_glb(mesh_id: str, task_id: str | None = None, db: AsyncSession = Depends(get_db)):
    """Get mesh as GLB with irradiance heatmap baked in."""
    entry = _mesh_cache.get(mesh_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Mesh not found")

    irradiance = None
    if task_id:
        from .. import _sim_store

        sim = _sim_store.get(task_id)
        if sim:
            irradiance = sim.get("irradiance")

    glb_bytes = _mesh_to_glb(entry["mesh"], irradiance)
    return Response(content=glb_bytes, media_type="model/gltf-binary")


@router.get("/demo/{mesh_type}")
async def get_demo_mesh(mesh_type: str):
    """Get a demo mesh as GLB. Types: 'simple' or 'complex'."""
    from ...simulation.demo_factory import create_factory_complex, create_simple_factory

    if mesh_type == "simple":
        mesh = create_simple_factory()
    elif mesh_type == "complex":
        mesh = create_factory_complex()
    else:
        raise HTTPException(status_code=400, detail="Invalid mesh type. Use 'simple' or 'complex'")

    mesh_id = f"demo_{mesh_type}"
    _mesh_cache[mesh_id] = {"mesh": mesh, "filename": f"demo_{mesh_type}.glb"}

    glb_bytes = _mesh_to_glb(mesh)
    return Response(content=glb_bytes, media_type="model/gltf-binary")


@router.get("/demo/{mesh_type}/heatmap")
async def get_demo_heatmap(mesh_type: str, task_id: str | None = None):
    """Get demo mesh with irradiance heatmap as GLB."""
    from ...simulation.demo_factory import create_factory_complex, create_simple_factory

    if mesh_type == "simple":
        mesh = create_simple_factory()
    elif mesh_type == "complex":
        mesh = create_factory_complex()
    else:
        raise HTTPException(status_code=400, detail="Invalid mesh type")

    irradiance = None
    if task_id:
        from .. import _sim_store

        sim = _sim_store.get(task_id)
        if sim:
            irradiance = sim.get("irradiance")

    glb_bytes = _mesh_to_glb(mesh, irradiance)
    return Response(content=glb_bytes, media_type="model/gltf-binary")
