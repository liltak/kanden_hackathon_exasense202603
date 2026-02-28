"""Sky visibility matrix for accurate diffuse irradiance calculation.

Precomputes a visibility matrix V[N_faces, N_patches] using Reinhart sky
subdivision and ray casting. This enables accurate diffuse irradiance
computation that accounts for obstructions from surrounding geometry.
"""

import logging

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


def generate_reinhart_patches(mf: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Generate Reinhart sky subdivision patch directions and solid angles.

    The Reinhart subdivision refines the Tregenza hemisphere into
    mf^2 * 145 + 1 patches (mf=2 gives ~580 patches).

    Args:
        mf: Multiplication factor (1=Tregenza 145, 2=Reinhart ~580).

    Returns:
        directions: (N_patches, 3) unit vectors in ENU coordinates (upper hemisphere).
        solid_angles: (N_patches,) solid angle of each patch in steradians.
    """
    # Tregenza row definitions: (elevation_center_deg, n_patches_in_row)
    tregenza_rows = [
        (6, 30), (18, 30), (30, 24), (42, 24), (54, 18),
        (66, 12), (78, 6), (90, 1),
    ]

    directions = []
    solid_angles = []

    for elev_deg, n_base in tregenza_rows:
        n_patches = n_base * mf if elev_deg < 90 else 1

        elev_rad = np.radians(elev_deg)
        cos_el = np.cos(elev_rad)
        sin_el = np.sin(elev_rad)

        # Band width in elevation (roughly 12 degrees per Tregenza band)
        d_elev = np.radians(12.0)
        # Solid angle per patch in this row
        band_solid_angle = 2 * np.pi * np.sin(elev_rad) * d_elev
        patch_solid_angle = band_solid_angle / n_patches

        for j in range(n_patches):
            az_rad = 2 * np.pi * j / n_patches
            # ENU: East = sin(az)*cos(el), North = cos(az)*cos(el), Up = sin(el)
            east = np.sin(az_rad) * cos_el
            north = np.cos(az_rad) * cos_el
            up = sin_el
            directions.append([east, north, up])
            solid_angles.append(patch_solid_angle)

    directions = np.array(directions, dtype=np.float64)
    solid_angles = np.array(solid_angles, dtype=np.float64)

    # Normalize directions
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / norms

    logger.info("Reinhart sky: %d patches (mf=%d)", len(directions), mf)
    return directions, solid_angles


def compute_sky_visibility_matrix(
    mesh: trimesh.Trimesh,
    face_indices: np.ndarray | None = None,
    mf: int = 2,
    offset: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute sky visibility matrix for mesh faces.

    For each face, casts rays toward all sky patches and records which
    patches are visible (not blocked by other geometry).

    Args:
        mesh: Triangle mesh for ray casting.
        face_indices: Subset of face indices to analyze. If None, all faces.
        mf: Reinhart multiplication factor.
        offset: Ray origin offset along face normal to avoid self-intersection.

    Returns:
        visibility: (N_faces, N_patches) boolean matrix.
        patch_directions: (N_patches, 3) sky patch direction vectors.
        patch_solid_angles: (N_patches,) solid angles in steradians.
    """
    patch_dirs, patch_solid_angles = generate_reinhart_patches(mf=mf)
    n_patches = len(patch_dirs)

    if face_indices is None:
        face_indices = np.arange(len(mesh.faces))

    n_faces = len(face_indices)
    centroids = mesh.triangles_center[face_indices]
    normals = mesh.face_normals[face_indices]

    # Offset origins along face normal
    origins = centroids + normals * offset

    visibility = np.zeros((n_faces, n_patches), dtype=bool)

    logger.info(
        "Computing sky visibility: %d faces x %d patches = %d rays",
        n_faces, n_patches, n_faces * n_patches,
    )

    # Process each sky patch direction
    for j in range(n_patches):
        patch_dir = patch_dirs[j]

        # Only check faces whose normal has positive dot with patch direction
        # (face must be facing toward the sky patch to see it)
        cos_angle = np.dot(normals, patch_dir)
        candidates = cos_angle > 0

        if not np.any(candidates):
            continue

        candidate_origins = origins[candidates]
        ray_dirs = np.tile(patch_dir, (len(candidate_origins), 1))

        # Check for occlusion
        hits = mesh.ray.intersects_any(
            ray_origins=candidate_origins,
            ray_directions=ray_dirs,
        )

        # Visible = facing the patch AND no occlusion
        candidate_indices = np.where(candidates)[0]
        visibility[candidate_indices[~hits], j] = True

    logger.info("Sky visibility computed: mean SVF = %.2f", visibility.mean())
    return visibility, patch_dirs, patch_solid_angles


def compute_sky_view_factors(
    visibility: np.ndarray,
    solid_angles: np.ndarray,
) -> np.ndarray:
    """Compute sky view factor for each face from visibility matrix.

    SVF = sum(visible_patch_solid_angles) / (2 * pi)

    Args:
        visibility: (N_faces, N_patches) boolean matrix.
        solid_angles: (N_patches,) solid angles in steradians.

    Returns:
        svf: (N_faces,) sky view factor in [0, 1].
    """
    total_visible = np.dot(visibility.astype(np.float64), solid_angles)
    svf = total_visible / (2 * np.pi)
    return np.clip(svf, 0.0, 1.0)
