"""Solar position calculation using pvlib.

Computes sun azimuth and elevation for a given location over an entire year.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pvlib


@dataclass
class SolarPositionResult:
    """Result of solar position calculation."""

    times: pd.DatetimeIndex
    azimuth: np.ndarray      # degrees, 0=North, clockwise
    elevation: np.ndarray    # degrees above horizon
    zenith: np.ndarray       # degrees from vertical (90 - elevation)

    @property
    def sun_visible(self) -> np.ndarray:
        """Boolean mask: True when sun is above horizon."""
        return self.elevation > 0

    def sun_direction_vectors(self) -> np.ndarray:
        """Unit vectors pointing toward sun in ENU (East-North-Up) coordinates.

        Returns:
            (N, 3) array of unit vectors [east, north, up]
        """
        az_rad = np.radians(self.azimuth)
        el_rad = np.radians(self.elevation)
        cos_el = np.cos(el_rad)
        east = np.sin(az_rad) * cos_el
        north = np.cos(az_rad) * cos_el
        up = np.sin(el_rad)
        return np.column_stack([east, north, up])


def compute_solar_positions(
    latitude: float,
    longitude: float,
    year: int = 2025,
    freq_minutes: int = 60,
    timezone: str = "Asia/Tokyo",
    altitude: float = 0.0,
) -> SolarPositionResult:
    """Compute solar positions for a full year.

    Args:
        latitude: Degrees north.
        longitude: Degrees east.
        year: Year to simulate.
        freq_minutes: Time resolution in minutes.
        timezone: Timezone string.
        altitude: Site altitude in meters.

    Returns:
        SolarPositionResult with times, azimuth, elevation, zenith arrays.
    """
    times = pd.date_range(
        start=f"{year}-01-01",
        end=f"{year}-12-31 23:59",
        freq=f"{freq_minutes}min",
        tz=timezone,
    )

    location = pvlib.location.Location(
        latitude=latitude,
        longitude=longitude,
        tz=timezone,
        altitude=altitude,
    )

    solpos = location.get_solarposition(times)

    return SolarPositionResult(
        times=times,
        azimuth=solpos["azimuth"].values,
        elevation=solpos["apparent_elevation"].values,
        zenith=solpos["apparent_zenith"].values,
    )


def compute_clear_sky_irradiance(
    latitude: float,
    longitude: float,
    year: int = 2025,
    freq_minutes: int = 60,
    timezone: str = "Asia/Tokyo",
    altitude: float = 0.0,
    model: str = "ineichen",
) -> pd.DataFrame:
    """Compute clear-sky DNI, DHI, GHI for a full year.

    Returns:
        DataFrame with columns: ghi, dni, dhi (W/m^2) indexed by time.
    """
    times = pd.date_range(
        start=f"{year}-01-01",
        end=f"{year}-12-31 23:59",
        freq=f"{freq_minutes}min",
        tz=timezone,
    )

    location = pvlib.location.Location(
        latitude=latitude,
        longitude=longitude,
        tz=timezone,
        altitude=altitude,
    )

    cs = location.get_clearsky(times, model=model)
    return cs
