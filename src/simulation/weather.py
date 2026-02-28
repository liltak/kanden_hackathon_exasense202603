"""Weather data loading from EPW files.

Provides real meteorological data as an alternative to clear-sky models,
enabling realistic annual energy yield predictions that account for
cloud cover, precipitation, and seasonal weather patterns.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

logger = logging.getLogger(__name__)


@dataclass
class WeatherData:
    """Weather data for simulation."""

    times: pd.DatetimeIndex
    ghi: np.ndarray  # Global Horizontal Irradiance (W/m^2)
    dni: np.ndarray  # Direct Normal Irradiance (W/m^2)
    dhi: np.ndarray  # Diffuse Horizontal Irradiance (W/m^2)
    temperature: np.ndarray | None = None  # Ambient temperature (°C)
    wind_speed: np.ndarray | None = None  # Wind speed (m/s)
    source: str = "epw"


def load_epw(epw_path: Path, year: int | None = None) -> WeatherData:
    """Load weather data from an EPW (EnergyPlus Weather) file.

    Args:
        epw_path: Path to the .epw file.
        year: Override the year in the EPW data (EPW files use a generic year).

    Returns:
        WeatherData with hourly irradiance and meteorological data.
    """
    epw_data, epw_meta = pvlib.iotools.read_epw(str(epw_path))
    logger.info(
        "Loaded EPW: %s (%.2f°N, %.2f°E, alt=%dm)",
        epw_meta.get("city", "Unknown"),
        epw_meta.get("latitude", 0),
        epw_meta.get("longitude", 0),
        epw_meta.get("altitude", 0),
    )

    times = epw_data.index
    if year is not None:
        times = times.map(lambda t: t.replace(year=year))
        times = pd.DatetimeIndex(times)

    return WeatherData(
        times=times,
        ghi=epw_data["ghi"].values.astype(np.float64),
        dni=epw_data["dni"].values.astype(np.float64),
        dhi=epw_data["dhi"].values.astype(np.float64),
        temperature=epw_data["temp_air"].values.astype(np.float64),
        wind_speed=epw_data["wind_speed"].values.astype(np.float64),
        source=f"epw:{epw_path.name}",
    )
