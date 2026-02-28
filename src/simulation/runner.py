"""End-to-end solar simulation runner.

Orchestrates the full simulation pipeline:
solar positions → ray casting → irradiance → ROI → visualization.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
import yaml
from rich.console import Console
from rich.progress import Progress

from .irradiance import FaceIrradiance, compute_face_irradiance, save_irradiance_results
from .ray_caster import compute_shadow_matrix
from .roi_calculator import ROIReport, generate_roi_report
from .solar_position import compute_clear_sky_irradiance, compute_solar_positions
from .visualization import (
    create_irradiance_heatmap,
    create_sun_path_diagram,
    save_heatmap_html,
)
from .weather import load_epw

console = Console()


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_demo_mesh() -> trimesh.Trimesh:
    """Create a demo factory-like mesh for testing.

    Generates a simple factory building with:
    - A rectangular base/floor
    - Walls
    - A pitched roof
    """
    # Factory dimensions (meters)
    length, width, wall_height, ridge_height = 40, 20, 8, 12

    vertices = np.array(
        [
            # Floor corners (0-3)
            [0, 0, 0],
            [length, 0, 0],
            [length, width, 0],
            [0, width, 0],
            # Wall top corners (4-7)
            [0, 0, wall_height],
            [length, 0, wall_height],
            [length, width, wall_height],
            [0, width, wall_height],
            # Ridge line (8-9)
            [0, width / 2, ridge_height],
            [length, width / 2, ridge_height],
        ],
        dtype=np.float64,
    )

    faces = np.array(
        [
            # Floor
            [0, 1, 2],
            [0, 2, 3],
            # Front wall (y=0)
            [0, 1, 5],
            [0, 5, 4],
            # Back wall (y=width)
            [2, 3, 7],
            [2, 7, 6],
            # Left wall (x=0) - triangle gable
            [0, 3, 8],
            [0, 8, 4],
            [3, 7, 8],
            # Right wall (x=length) - triangle gable
            [1, 2, 9],
            [1, 9, 5],
            [2, 6, 9],
            # Roof - front slope (y=0 side)
            [4, 5, 9],
            [4, 9, 8],
            # Roof - back slope (y=width side)
            [6, 7, 8],
            [6, 8, 9],
        ],
        dtype=np.int64,
    )

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    return mesh


def run_simulation(
    mesh: trimesh.Trimesh,
    config: dict,
    output_dir: Path,
) -> tuple[list[FaceIrradiance], ROIReport]:
    """Run the full solar simulation pipeline.

    Args:
        mesh: Building mesh in ENU coordinates (meters).
        config: Configuration dictionary.
        output_dir: Directory for output files.

    Returns:
        Tuple of (irradiance_results, roi_report).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    loc = config["location"]
    sim = config["simulation"]

    with Progress() as progress:
        # Step 1: Solar positions
        task = progress.add_task("Computing solar positions...", total=4)
        solar = compute_solar_positions(
            latitude=loc["latitude"],
            longitude=loc["longitude"],
            year=sim["year"],
            freq_minutes=sim["time_resolution_minutes"],
            timezone=loc["timezone"],
            altitude=loc.get("altitude", 0),
        )
        progress.advance(task)

        # Step 2: Weather / irradiance data
        epw_path = sim.get("epw_file")
        if epw_path:
            progress.update(task, description="Loading EPW weather data...")
            epw_file = Path(epw_path)
            if not epw_file.is_absolute():
                epw_file = Path(__file__).parent.parent.parent / epw_file
            weather = load_epw(epw_file, year=sim["year"])
            cs = pd.DataFrame(
                {"ghi": weather.ghi, "dni": weather.dni, "dhi": weather.dhi},
                index=weather.times,
            )
            console.print(f"[yellow]Weather source: {weather.source}")
        else:
            progress.update(task, description="Computing clear sky irradiance...")
            cs = compute_clear_sky_irradiance(
                latitude=loc["latitude"],
                longitude=loc["longitude"],
                year=sim["year"],
                freq_minutes=sim["time_resolution_minutes"],
                timezone=loc["timezone"],
                altitude=loc.get("altitude", 0),
                model=sim.get("dni_model", "ineichen"),
            )
            console.print("[yellow]Weather source: clear-sky model")
        progress.advance(task)

        # Step 3: Ray casting
        progress.update(task, description="Computing shadows (ray casting)...")
        sun_dirs = solar.sun_direction_vectors()
        shadow_matrix = compute_shadow_matrix(
            mesh=mesh,
            sun_directions=sun_dirs,
            sun_visible=solar.sun_visible,
            min_face_area=config.get("mesh", {}).get("min_face_area_m2", 0),
        )
        progress.advance(task)

        # Step 4: Irradiance calculation
        progress.update(task, description="Computing annual irradiance...")
        time_step_hours = sim["time_resolution_minutes"] / 60.0
        diffuse_model = sim.get("diffuse_model", "isotropic")
        irradiance_results = compute_face_irradiance(
            face_normals=mesh.face_normals,
            face_areas=mesh.area_faces,
            shadow_matrix=shadow_matrix,
            sun_directions=sun_dirs,
            dni=cs["dni"].values,
            dhi=cs["dhi"].values if sim.get("include_diffuse", True) else np.zeros(len(cs)),
            time_step_hours=time_step_hours,
            solar_zenith=solar.zenith,
            solar_azimuth=solar.azimuth,
            ghi=cs["ghi"].values if "ghi" in cs.columns else None,
            diffuse_model=diffuse_model,
            albedo=sim.get("albedo", 0.15),
        )
        progress.advance(task)

    # Save irradiance results
    save_irradiance_results(
        irradiance_results, output_dir / "irradiance_results.json"
    )
    console.print(f"[green]Irradiance results saved to {output_dir / 'irradiance_results.json'}")

    # Generate ROI report
    panel_cfg = config.get("panel", {})
    elec_cfg = config.get("electricity", {})
    roi_report = generate_roi_report(
        irradiance_results,
        panel_efficiency=panel_cfg.get("efficiency", 0.20),
        cost_per_kw_jpy=panel_cfg.get("cost_per_kw", 250_000),
        electricity_price_jpy=elec_cfg.get("price_per_kwh", 30),
        annual_price_increase=elec_cfg.get("annual_price_increase", 0.02),
        degradation_rate=panel_cfg.get("degradation_rate", 0.005),
        lifespan_years=panel_cfg.get("lifespan_years", 25),
    )

    # Generate visualizations
    console.print("[blue]Generating visualizations...")

    # Sun path diagram
    sun_path_fig = create_sun_path_diagram(solar.azimuth, solar.elevation)
    save_heatmap_html(sun_path_fig, output_dir / "sun_path.html")

    # 3D irradiance heatmap
    heatmap_fig = create_irradiance_heatmap(mesh, irradiance_results)
    save_heatmap_html(heatmap_fig, output_dir / "irradiance_heatmap.html")

    console.print(f"[green]Visualizations saved to {output_dir}")

    # Print summary
    console.print("\n[bold]===== Simulation Summary =====")
    console.print(f"Location: {loc['latitude']}°N, {loc['longitude']}°E")
    console.print(f"Mesh faces: {len(mesh.faces)}")
    console.print(f"Suitable faces for panels: {len(roi_report.proposals)}")
    console.print(f"Total installable capacity: {roi_report.total_capacity_kw:.1f} kW")
    console.print(
        f"Annual generation: {roi_report.total_annual_generation_kwh:.0f} kWh"
    )
    console.print(
        f"Annual savings: ¥{roi_report.total_annual_savings_jpy:,.0f}"
    )
    console.print(f"Payback period: {roi_report.overall_payback_years:.1f} years")
    console.print(f"25-year NPV: ¥{roi_report.overall_npv_25y_jpy:,.0f}")

    return irradiance_results, roi_report


def run_demo(output_dir: Path | None = None, config_path: Path | None = None):
    """Run simulation with demo mesh and default config."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "solar_params.yaml"

    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "data" / "simulation_results"

    config = load_config(config_path)
    mesh = create_demo_mesh()

    console.print("[bold blue]ExaSense Solar Simulation Demo")
    console.print(f"Demo mesh: factory building ({len(mesh.faces)} faces)")
    console.print(f"Config: {config_path}\n")

    # Save demo mesh
    mesh.export(str(output_dir / "demo_mesh.obj"))

    return run_simulation(mesh, config, output_dir)


if __name__ == "__main__":
    run_demo()
