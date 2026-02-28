"""Annual irradiance calculation for mesh faces.

Integrates solar irradiance over the year for each face,
accounting for face orientation, shadows, and both direct and diffuse radiation.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class FaceIrradiance:
    """Irradiance result for a single face."""

    face_id: int
    annual_irradiance_kwh_m2: float
    annual_direct_kwh_m2: float
    annual_diffuse_kwh_m2: float
    area_m2: float
    normal: tuple[float, float, float]
    sun_hours: float  # hours per year with direct sunlight


def compute_face_irradiance(
    face_normals: np.ndarray,
    face_areas: np.ndarray,
    shadow_matrix: np.ndarray,
    sun_directions: np.ndarray,
    dni: np.ndarray,
    dhi: np.ndarray,
    time_step_hours: float = 1.0,
) -> list[FaceIrradiance]:
    """Compute annual irradiance for each mesh face.

    Args:
        face_normals: (n_faces, 3) face normal vectors.
        face_areas: (n_faces,) face areas in m^2.
        shadow_matrix: (T, n_faces) boolean — True if illuminated.
        sun_directions: (T, 3) sun direction unit vectors.
        dni: (T,) Direct Normal Irradiance in W/m^2.
        dhi: (T,) Diffuse Horizontal Irradiance in W/m^2.
        time_step_hours: Time step duration in hours.

    Returns:
        List of FaceIrradiance for each face.
    """
    n_faces = face_normals.shape[0]
    n_times = sun_directions.shape[0]

    # Compute cos(incidence angle) = dot(face_normal, sun_direction)
    # Shape: (T, n_faces)
    cos_incidence = np.dot(sun_directions, face_normals.T)
    cos_incidence = np.clip(cos_incidence, 0, 1)  # only positive contributions

    # Direct irradiance on each face at each time: DNI * cos(theta) * illuminated
    # dni shape: (T,) -> (T, 1)
    direct_irradiance = dni[:, np.newaxis] * cos_incidence * shadow_matrix

    # Diffuse irradiance: simplified isotropic model
    # Each face receives DHI weighted by its sky view factor
    # For a tilted surface, sky view factor ≈ (1 + cos(tilt)) / 2
    # where tilt = angle between face normal and vertical (up)
    up = np.array([0, 0, 1])
    cos_tilt = np.dot(face_normals, up)
    cos_tilt = np.clip(cos_tilt, 0, 1)
    sky_view_factor = (1 + cos_tilt) / 2  # (n_faces,)

    diffuse_irradiance = dhi[:, np.newaxis] * sky_view_factor[np.newaxis, :]

    # Integrate over time (convert W/m^2 * hours → Wh/m^2, then to kWh/m^2)
    annual_direct = np.sum(direct_irradiance * time_step_hours, axis=0) / 1000.0
    annual_diffuse = np.sum(diffuse_irradiance * time_step_hours, axis=0) / 1000.0
    annual_total = annual_direct + annual_diffuse

    # Sun hours: count time steps where face is illuminated
    sun_hours = np.sum(shadow_matrix, axis=0) * time_step_hours

    results = []
    for i in range(n_faces):
        results.append(
            FaceIrradiance(
                face_id=i,
                annual_irradiance_kwh_m2=float(annual_total[i]),
                annual_direct_kwh_m2=float(annual_direct[i]),
                annual_diffuse_kwh_m2=float(annual_diffuse[i]),
                area_m2=float(face_areas[i]),
                normal=tuple(face_normals[i].tolist()),
                sun_hours=float(sun_hours[i]),
            )
        )

    return results


def save_irradiance_results(results: list[FaceIrradiance], output_path: Path) -> None:
    """Save irradiance results to JSON."""
    data = [
        {
            "face_id": r.face_id,
            "annual_irradiance_kwh_m2": round(r.annual_irradiance_kwh_m2, 2),
            "annual_direct_kwh_m2": round(r.annual_direct_kwh_m2, 2),
            "annual_diffuse_kwh_m2": round(r.annual_diffuse_kwh_m2, 2),
            "area_m2": round(r.area_m2, 4),
            "normal": [round(n, 4) for n in r.normal],
            "sun_hours": round(r.sun_hours, 1),
        }
        for r in results
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def load_irradiance_results(path: Path) -> list[dict]:
    """Load irradiance results from JSON."""
    return json.loads(path.read_text())
