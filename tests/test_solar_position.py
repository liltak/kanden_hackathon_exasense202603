"""Tests for solar position calculation."""

import numpy as np
import pytest

from src.simulation.solar_position import compute_clear_sky_irradiance, compute_solar_positions


@pytest.fixture
def osaka_solar():
    """Solar positions for Osaka, 2025."""
    return compute_solar_positions(
        latitude=34.69,
        longitude=135.50,
        year=2025,
        freq_minutes=60,
    )


class TestSolarPosition:
    def test_returns_8760_points(self, osaka_solar):
        """1-hour resolution for a full year = 8760 points."""
        assert len(osaka_solar.times) == 8760

    def test_azimuth_range(self, osaka_solar):
        """Azimuth should be 0-360 degrees."""
        assert np.all(osaka_solar.azimuth >= 0)
        assert np.all(osaka_solar.azimuth <= 360)

    def test_elevation_range(self, osaka_solar):
        """Elevation should be between -90 and 90."""
        assert np.all(osaka_solar.elevation >= -90)
        assert np.all(osaka_solar.elevation <= 90)

    def test_summer_higher_elevation(self, osaka_solar):
        """Summer should have higher max elevation than winter."""
        # June 21 ≈ day 172, December 21 ≈ day 355
        june_start = 172 * 24
        june_end = june_start + 24
        dec_start = 355 * 24
        dec_end = dec_start + 24

        summer_max = np.max(osaka_solar.elevation[june_start:june_end])
        winter_max = np.max(osaka_solar.elevation[dec_start:dec_end])

        assert summer_max > winter_max
        assert summer_max > 70  # Osaka ~78° in summer
        assert winter_max < 40  # Osaka ~32° in winter

    def test_sun_direction_vectors_unit(self, osaka_solar):
        """Sun direction vectors should be unit vectors."""
        dirs = osaka_solar.sun_direction_vectors()
        visible = osaka_solar.sun_visible
        norms = np.linalg.norm(dirs[visible], axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_sun_visible_hours(self, osaka_solar):
        """Osaka should have ~4000-5000 sun hours per year."""
        sun_hours = np.sum(osaka_solar.sun_visible)
        assert 4000 < sun_hours < 5000


class TestClearSky:
    def test_irradiance_columns(self):
        cs = compute_clear_sky_irradiance(
            latitude=34.69, longitude=135.50, year=2025
        )
        assert "ghi" in cs.columns
        assert "dni" in cs.columns
        assert "dhi" in cs.columns

    def test_irradiance_non_negative(self):
        cs = compute_clear_sky_irradiance(
            latitude=34.69, longitude=135.50, year=2025
        )
        assert (cs["ghi"] >= 0).all()
        assert (cs["dni"] >= 0).all()
        assert (cs["dhi"] >= 0).all()

    def test_peak_dni_reasonable(self):
        """Peak DNI should be < 1100 W/m² (physical limit on Earth)."""
        cs = compute_clear_sky_irradiance(
            latitude=34.69, longitude=135.50, year=2025
        )
        assert cs["dni"].max() < 1100
