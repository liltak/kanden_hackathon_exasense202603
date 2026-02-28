"""Celery task for running solar simulations.

Wraps runner.py's run_simulation() with progress reporting via Celery state updates.
Used when Redis is available; otherwise the FastAPI async fallback is used.
"""

from __future__ import annotations

from pathlib import Path

from .celery_app import celery_app

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = PROJECT_ROOT / "data" / "simulation_results"


@celery_app.task(bind=True, name="exasense.run_simulation")
def run_simulation_task(
    self,
    latitude: float = 34.69,
    longitude: float = 135.50,
    year: int = 2025,
    time_resolution_minutes: int = 60,
    panel_efficiency_pct: float = 20.0,
    electricity_price_jpy: float = 30.0,
    mesh_source: str = "complex",
):
    """Run full solar simulation pipeline as a Celery task.

    Reports progress via self.update_state() with 4 steps:
    1. solar_positions (0-25%)
    2. clear_sky (25-50%)
    3. ray_casting (50-75%)
    4. irradiance (75-100%)
    """
    import time

    import numpy as np

    from ..simulation.demo_factory import create_factory_complex, create_simple_factory
    from ..simulation.irradiance import compute_face_irradiance
    from ..simulation.ray_caster import compute_shadow_matrix
    from ..simulation.roi_calculator import generate_roi_report
    from ..simulation.runner import load_config
    from ..simulation.solar_position import compute_clear_sky_irradiance, compute_solar_positions

    config_path = CONFIGS_DIR / "solar_params.yaml"
    config = load_config(config_path)
    config["location"]["latitude"] = latitude
    config["location"]["longitude"] = longitude
    config["simulation"]["year"] = year
    config["simulation"]["time_resolution_minutes"] = time_resolution_minutes
    config["panel"]["efficiency"] = panel_efficiency_pct / 100
    config["electricity"]["price_per_kwh"] = electricity_price_jpy

    if mesh_source == "simple":
        mesh = create_simple_factory()
    else:
        mesh = create_factory_complex()

    loc = config["location"]
    sim = config["simulation"]
    t0 = time.time()

    # Step 1: Solar positions
    self.update_state(state="PROGRESS", meta={"step": "solar_positions", "progress": 0.0})
    solar = compute_solar_positions(
        latitude=loc["latitude"],
        longitude=loc["longitude"],
        year=sim["year"],
        freq_minutes=sim["time_resolution_minutes"],
        timezone=loc["timezone"],
        altitude=loc.get("altitude", 0),
    )

    # Step 2: Clear sky
    self.update_state(state="PROGRESS", meta={"step": "clear_sky", "progress": 0.25})
    cs = compute_clear_sky_irradiance(
        latitude=loc["latitude"],
        longitude=loc["longitude"],
        year=sim["year"],
        freq_minutes=sim["time_resolution_minutes"],
        timezone=loc["timezone"],
        altitude=loc.get("altitude", 0),
        model=sim.get("dni_model", "ineichen"),
    )

    # Step 3: Ray casting
    self.update_state(state="PROGRESS", meta={"step": "ray_casting", "progress": 0.5})
    sun_dirs = solar.sun_direction_vectors()
    shadow_matrix = compute_shadow_matrix(
        mesh=mesh,
        sun_directions=sun_dirs,
        sun_visible=solar.sun_visible,
        min_face_area=config.get("mesh", {}).get("min_face_area_m2", 0),
    )

    # Step 4: Irradiance
    self.update_state(state="PROGRESS", meta={"step": "irradiance", "progress": 0.75})
    time_step_hours = sim["time_resolution_minutes"] / 60.0
    irr = compute_face_irradiance(
        face_normals=mesh.face_normals,
        face_areas=mesh.area_faces,
        shadow_matrix=shadow_matrix,
        sun_directions=sun_dirs,
        dni=cs["dni"].values,
        dhi=cs["dhi"].values if sim.get("include_diffuse", True) else np.zeros(len(cs)),
        time_step_hours=time_step_hours,
    )

    # ROI
    panel_cfg = config.get("panel", {})
    elec_cfg = config.get("electricity", {})
    roi = generate_roi_report(
        irr,
        panel_efficiency=panel_cfg.get("efficiency", 0.20),
        cost_per_kw_jpy=panel_cfg.get("cost_per_kw", 250_000),
        electricity_price_jpy=elec_cfg.get("price_per_kwh", 30),
        annual_price_increase=elec_cfg.get("annual_price_increase", 0.02),
        degradation_rate=panel_cfg.get("degradation_rate", 0.005),
        lifespan_years=panel_cfg.get("lifespan_years", 25),
    )

    elapsed = time.time() - t0

    # Serialize results
    return {
        "status": "complete",
        "elapsed_seconds": elapsed,
        "n_faces": len(irr),
        "n_proposals": len(roi.proposals),
        "total_capacity_kw": roi.total_capacity_kw,
        "total_annual_generation_kwh": roi.total_annual_generation_kwh,
        "total_annual_savings_jpy": roi.total_annual_savings_jpy,
        "overall_payback_years": roi.overall_payback_years,
        "overall_npv_25y_jpy": roi.overall_npv_25y_jpy,
    }
