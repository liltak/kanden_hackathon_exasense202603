"""Ray casting for shadow detection on mesh surfaces.

Determines which mesh faces are illuminated or shaded at each time step
by casting rays from face centroids toward the sun.
"""

import logging
from dataclasses import dataclass

import numpy as np
import trimesh

logger = logging.getLogger(__name__)

try:
    from trimesh.ray import ray_pyembree  # noqa: F401

    HAS_EMBREE = True
except ImportError:
    HAS_EMBREE = False

logger.info("Ray backend: %s", "Embree (pyembree)" if HAS_EMBREE else "trimesh native")


@dataclass
class ShadowResult:
    """Shadow analysis result for a single time step."""

    illuminated: np.ndarray  # bool array (n_faces,) — True if face is lit
    n_faces: int
    sun_elevation: float
    sun_azimuth: float


def cast_shadows(
    mesh: trimesh.Trimesh,
    sun_direction: np.ndarray,
    face_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Determine which faces are illuminated by the sun.

    A face is illuminated if:
    1. Its normal has a positive dot product with the sun direction (facing the sun)
    2. A ray from its centroid toward the sun does not intersect other faces

    Args:
        mesh: Triangle mesh with face normals.
        sun_direction: Unit vector pointing toward sun in mesh coordinates (3,).
        face_indices: Optional subset of face indices to check. If None, check all.

    Returns:
        Boolean array (n_faces,) — True if face is illuminated.
    """
    sun_dir = np.asarray(sun_direction, dtype=np.float64)
    sun_dir = sun_dir / np.linalg.norm(sun_dir)

    n_faces = len(mesh.faces)
    if face_indices is None:
        face_indices = np.arange(n_faces)

    illuminated = np.zeros(n_faces, dtype=bool)

    centroids = mesh.triangles_center[face_indices]
    normals = mesh.face_normals[face_indices]

    # Step 1: Check if face is oriented toward the sun
    cos_angle = np.dot(normals, sun_dir)
    facing_sun = cos_angle > 0
    candidates = face_indices[facing_sun]
    candidate_centroids = centroids[facing_sun]

    if len(candidates) == 0:
        return illuminated

    # Step 2: Cast rays from centroids toward sun, check for occlusion
    # Offset origins slightly along normal to avoid self-intersection
    candidate_normals = normals[facing_sun]
    origins = candidate_centroids + candidate_normals * 0.01
    directions = np.tile(sun_dir, (len(origins), 1))

    # Use trimesh ray casting
    hits = mesh.ray.intersects_any(
        ray_origins=origins,
        ray_directions=directions,
    )

    # Face is illuminated if it faces the sun AND ray doesn't hit anything
    illuminated[candidates[~hits]] = True

    return illuminated


def compute_shadow_matrix(
    mesh: trimesh.Trimesh,
    sun_directions: np.ndarray,
    sun_visible: np.ndarray,
    min_face_area: float = 0.0,
) -> np.ndarray:
    """Compute shadow matrix for all time steps.

    Args:
        mesh: Triangle mesh.
        sun_directions: (T, 3) array of sun direction vectors.
        sun_visible: (T,) boolean array — True when sun is above horizon.
        min_face_area: Skip faces smaller than this area (m^2).

    Returns:
        (T, n_faces) boolean matrix — True if face is illuminated at time t.
    """
    n_times = len(sun_directions)
    n_faces = len(mesh.faces)

    # Filter faces by area
    face_indices = None
    if min_face_area > 0:
        areas = mesh.area_faces
        valid = areas >= min_face_area
        face_indices = np.where(valid)[0]

    shadow_matrix = np.zeros((n_times, n_faces), dtype=bool)

    for t in range(n_times):
        if not sun_visible[t]:
            continue
        shadow_matrix[t] = cast_shadows(mesh, sun_directions[t], face_indices)

    return shadow_matrix
