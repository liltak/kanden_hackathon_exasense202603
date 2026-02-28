"""Cell temperature model for solar panel temperature-dependent losses.

Computes panel cell temperature using pvlib SAPM model and calculates
temperature-dependent power correction factors. Japanese summer conditions
(35-40°C) can cause 10-20% output reduction.
"""

import logging

import numpy as np
import pvlib

logger = logging.getLogger(__name__)


def compute_cell_temperature(
    poa_global: np.ndarray,
    temp_air: np.ndarray,
    wind_speed: np.ndarray,
    model: str = "sapm",
    mount: str = "open_rack_glass_polymer",
) -> np.ndarray:
    """Compute cell temperature from environmental conditions.

    Args:
        poa_global: Plane-of-array irradiance (W/m^2), shape (T,).
        temp_air: Ambient air temperature (°C), shape (T,).
        wind_speed: Wind speed at panel height (m/s), shape (T,).
        model: Temperature model ("sapm" or "faiman").
        mount: Module mounting type for SAPM coefficients.

    Returns:
        Cell temperature in °C, shape (T,).
    """
    if model == "sapm":
        params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][mount]
        cell_temp = pvlib.temperature.sapm_cell(
            poa_global=poa_global,
            temp_air=temp_air,
            wind_speed=wind_speed,
            a=params["a"],
            b=params["b"],
            deltaT=params["deltaT"],
        )
    else:
        cell_temp = pvlib.temperature.faiman(
            poa_global=poa_global,
            temp_air=temp_air,
            wind_speed=wind_speed,
        )

    return np.asarray(cell_temp, dtype=np.float64)


def compute_temperature_loss(
    cell_temperature: np.ndarray,
    temp_coeff_pmax: float = -0.004,
    t_ref: float = 25.0,
) -> np.ndarray:
    """Compute temperature-dependent power correction factor.

    Args:
        cell_temperature: Cell temperature in °C, shape (T,).
        temp_coeff_pmax: Temperature coefficient of Pmax (%/°C as fraction, e.g., -0.004).
        t_ref: Reference temperature (STC = 25°C).

    Returns:
        Correction factor (0-1+), shape (T,). Values < 1 indicate loss.
    """
    factor = 1.0 + temp_coeff_pmax * (cell_temperature - t_ref)
    return np.clip(factor, 0.0, None)
