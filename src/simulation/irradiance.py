"""Annual irradiance calculation for mesh faces.

Integrates solar irradiance over the year for each face,
accounting for face orientation, shadows, and both direct and diffuse radiation.
Supports isotropic and Perez anisotropic diffuse models via pvlib.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

logger = logging.getLogger(__name__)


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


def _normal_to_tilt_azimuth(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert face normal vectors to surface tilt and azimuth angles.

    Args:
        normals: (n_faces, 3) face normal vectors in ENU coordinates.

    Returns:
        tilt: (n_faces,) surface tilt in degrees (0=horizontal, 90=vertical).
        azimuth: (n_faces,) surface azimuth in degrees (0=North, clockwise).
    """
    up = np.array([0, 0, 1])
    cos_tilt = np.clip(np.dot(normals, up), -1, 1)
    tilt = np.degrees(np.arccos(cos_tilt))

    # Surface azimuth: direction the surface faces (opposite of outward normal projected to xy)
    azimuth = np.degrees(np.arctan2(normals[:, 0], normals[:, 1])) % 360
    return tilt, azimuth


def compute_face_irradiance(
    face_normals: np.ndarray,
    face_areas: np.ndarray,
    shadow_matrix: np.ndarray,
    sun_directions: np.ndarray,
    dni: np.ndarray,
    dhi: np.ndarray,
    time_step_hours: float = 1.0,
    solar_zenith: np.ndarray | None = None,
    solar_azimuth: np.ndarray | None = None,
    ghi: np.ndarray | None = None,
    diffuse_model: str = "isotropic",
    albedo: float = 0.15,
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
        solar_zenith: (T,) solar zenith in degrees (required for Perez model).
        solar_azimuth: (T,) solar azimuth in degrees (required for Perez model).
        ghi: (T,) Global Horizontal Irradiance in W/m^2 (required for Perez model).
        diffuse_model: "isotropic" or "perez".
        albedo: Ground reflectance for ground-reflected irradiance (default 0.15).

    Returns:
        List of FaceIrradiance for each face.
    """
    n_faces = face_normals.shape[0]

    # Compute cos(incidence angle) = dot(face_normal, sun_direction)
    cos_incidence = np.dot(sun_directions, face_normals.T)
    cos_incidence = np.clip(cos_incidence, 0, 1)

    # Direct irradiance: DNI * cos(theta) * illuminated
    direct_irradiance = dni[:, np.newaxis] * cos_incidence * shadow_matrix

    # Diffuse irradiance
    if diffuse_model == "perez" and solar_zenith is not None and solar_azimuth is not None:
        logger.info("Using Perez anisotropic diffuse model (albedo=%.2f)", albedo)
        surface_tilt, surface_azimuth = _normal_to_tilt_azimuth(face_normals)

        if ghi is None:
            ghi = dni * np.cos(np.radians(solar_zenith)).clip(0) + dhi

        # Perez model requires extraterrestrial DNI
        day_of_year = np.arange(1, len(dni) + 1) * (365.0 / len(dni))
        dni_extra = pvlib.irradiance.get_extra_radiation(day_of_year)

        diffuse_irradiance = np.zeros((len(dni), n_faces))
        for i in range(n_faces):
            poa = pvlib.irradiance.get_total_irradiance(
                surface_tilt=surface_tilt[i],
                surface_azimuth=surface_azimuth[i],
                solar_zenith=solar_zenith,
                solar_azimuth=solar_azimuth,
                dni=dni,
                ghi=ghi,
                dhi=dhi,
                dni_extra=dni_extra,
                model="perez",
                albedo=albedo,
            )
            # Extract diffuse + ground-reflected (exclude direct, we handle it separately)
            sky_diff = poa["poa_sky_diffuse"]
            gnd_diff = poa["poa_ground_diffuse"]
            # Handle both Series and ndarray returns from pvlib
            if hasattr(sky_diff, "fillna"):
                sky_diff = sky_diff.fillna(0).values
                gnd_diff = gnd_diff.fillna(0).values
            else:
                sky_diff = np.nan_to_num(sky_diff, nan=0.0)
                gnd_diff = np.nan_to_num(gnd_diff, nan=0.0)
            diffuse_irradiance[:, i] = sky_diff + gnd_diff
    else:
        if diffuse_model == "perez":
            logger.warning(
                "Perez model requested but solar_zenith/solar_azimuth not provided. "
                "Falling back to isotropic model."
            )
        # Isotropic model: DHI * (1 + cos(tilt)) / 2
        up = np.array([0, 0, 1])
        cos_tilt = np.clip(np.dot(face_normals, up), 0, 1)
        sky_view_factor = (1 + cos_tilt) / 2
        diffuse_irradiance = dhi[:, np.newaxis] * sky_view_factor[np.newaxis, :]

    # Integrate over time (W/m^2 * hours → kWh/m^2)
    annual_direct = np.sum(direct_irradiance * time_step_hours, axis=0) / 1000.0
    annual_diffuse = np.sum(diffuse_irradiance * time_step_hours, axis=0) / 1000.0
    annual_total = annual_direct + annual_diffuse

    # Sun hours
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
