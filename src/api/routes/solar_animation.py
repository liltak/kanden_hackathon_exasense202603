"""Solar animation API routes — sun positions and shadow timeline."""

from __future__ import annotations

import datetime

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from ..schemas import ShadowTimelineResponse, SunPositionEntry, SunPositionsResponse

router = APIRouter(prefix="/api/solar", tags=["solar-animation"])


def _enu_to_yup(vec: np.ndarray) -> list[float]:
    """Convert ENU [east, north, up] to Three.js Y-up [east, up, -north]."""
    return [float(vec[0]), float(vec[2]), float(-vec[1])]


@router.get("/positions", response_model=SunPositionsResponse)
async def get_solar_positions(
    date: str = Query(..., description="Date in YYYY-MM-DD format"),
    lat: float = Query(34.69, ge=-90, le=90),
    lng: float = Query(135.50, ge=-180, le=180),
    freq: int = Query(15, ge=1, le=60, description="Time step in minutes"),
):
    """Return sun positions for a single day with Y-up direction vectors."""
    from ...simulation.solar_position import compute_solar_positions

    try:
        target_date = datetime.date.fromisoformat(date)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {date}")

    year = target_date.year
    solar = compute_solar_positions(
        latitude=lat,
        longitude=lng,
        year=year,
        freq_minutes=freq,
        timezone="Asia/Tokyo",
    )

    # Filter to the target date
    mask = solar.times.date == target_date
    times_day = solar.times[mask]
    az_day = solar.azimuth[mask]
    el_day = solar.elevation[mask]

    if len(times_day) == 0:
        raise HTTPException(status_code=400, detail="No data for the specified date")

    # Only include daytime (sun above horizon)
    sun_dirs_enu = solar.sun_direction_vectors()[mask]

    entries: list[SunPositionEntry] = []
    for i in range(len(times_day)):
        if el_day[i] <= 0:
            continue
        entries.append(
            SunPositionEntry(
                time=times_day[i].strftime("%H:%M"),
                azimuth=round(float(az_day[i]), 2),
                elevation=round(float(el_day[i]), 2),
                direction_y_up=_enu_to_yup(sun_dirs_enu[i]),
            )
        )

    return SunPositionsResponse(
        date=date,
        latitude=lat,
        longitude=lng,
        freq_minutes=freq,
        positions=entries,
    )


@router.get("/shadow-timeline", response_model=ShadowTimelineResponse)
async def get_shadow_timeline(
    date: str = Query(..., description="Date in YYYY-MM-DD format"),
    mesh_source: str = Query("complex", description="'simple' or 'complex'"),
    lat: float = Query(34.69, ge=-90, le=90),
    lng: float = Query(135.50, ge=-180, le=180),
    freq: int = Query(15, ge=1, le=60),
):
    """Return shadow matrix for a single day (T x N_faces boolean)."""
    import trimesh

    from ...simulation.demo_factory import create_factory_complex, create_simple_factory
    from ...simulation.ray_caster import compute_shadow_matrix
    from ...simulation.solar_position import compute_solar_positions

    try:
        target_date = datetime.date.fromisoformat(date)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {date}")

    # Load mesh
    if mesh_source == "simple":
        mesh = create_simple_factory()
    elif mesh_source == "complex":
        mesh = create_factory_complex()
    else:
        raise HTTPException(status_code=400, detail="Invalid mesh_source. Use 'simple' or 'complex'")

    # Compute solar positions for the day
    year = target_date.year
    solar = compute_solar_positions(
        latitude=lat, longitude=lng, year=year, freq_minutes=freq, timezone="Asia/Tokyo"
    )

    mask = solar.times.date == target_date
    el_day = solar.elevation[mask]
    sun_dirs_enu = solar.sun_direction_vectors()[mask]

    # Daytime only
    daytime = el_day > 0
    sun_dirs_daytime = sun_dirs_enu[daytime]
    sun_visible_daytime = np.ones(len(sun_dirs_daytime), dtype=bool)

    if len(sun_dirs_daytime) == 0:
        return ShadowTimelineResponse(
            date=date,
            mesh_source=mesh_source,
            n_faces=len(mesh.faces),
            n_steps=0,
            times=[],
            shadow_matrix=[],
        )

    # Compute shadow matrix
    shadow_mat = compute_shadow_matrix(mesh, sun_dirs_daytime, sun_visible_daytime)

    # Build time labels for daytime steps
    times_day = solar.times[mask]
    times_daytime = times_day[daytime]
    time_labels = [t.strftime("%H:%M") for t in times_daytime]

    return ShadowTimelineResponse(
        date=date,
        mesh_source=mesh_source,
        n_faces=len(mesh.faces),
        n_steps=len(time_labels),
        times=time_labels,
        shadow_matrix=shadow_mat.tolist(),
    )
