#!/usr/bin/env python3
"""H100 E2E Pipeline Test — Phase 1 (VGGT) → Phase 2 (Mesh) → Phase 3 (Solar).

Runs the full reconstruction + simulation pipeline on H100 GPU,
benchmarks performance, and compares with T4 baseline results.

Usage:
    uv run python scripts/run_h100_e2e.py [--image-dir DATA] [--max-images 20]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

console = Console()

# T4 baseline from data/e2e_results/e2e_results.json
T4_BASELINE = {
    "n_images": 10,
    "vggt_model_load_s": 239.8,
    "vggt_inference_s": 32.0,
    "vggt_peak_vram_gb": 10.45,
    "mesh_processing_s": 77.0,
    "solar_simulation_s": 1.4,
    "total_s": 360.6,
    "n_points": 1_740_480,
    "n_faces": 566_130,
    "annual_ghi_kwh_m2": 2118.7,
    "annual_savings_yen": 82_545.0,
    "payback_years": 4.9,
}


def get_gpu_info() -> dict:
    """Collect GPU information via nvidia-smi."""
    info = {"gpu_name": "unknown", "gpu_memory_gb": 0, "driver_version": "unknown"}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 3:
                info["gpu_name"] = parts[0].strip()
                info["gpu_memory_gb"] = round(float(parts[1].strip()) / 1024, 1)
                info["driver_version"] = parts[2].strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return info


def ensure_dataset(image_dir: Path) -> Path:
    """Ensure the dataset images exist, download if needed.

    Checks both the given path directly and an `images/` subdirectory.
    Returns the path to the directory containing the images.
    """
    extensions = {"*.JPG", "*.jpg", "*.jpeg", "*.png"}

    def _count_images(d: Path) -> int:
        return sum(len(list(d.glob(ext))) for ext in extensions)

    # Check if image_dir itself contains images
    if image_dir.is_dir() and _count_images(image_dir) > 0:
        n = _count_images(image_dir)
        console.print(f"[green]Dataset found: {n} images in {image_dir}")
        return image_dir

    # Check images/ subdirectory (Mip-NeRF 360 structure)
    images_subdir = image_dir / "images"
    if images_subdir.is_dir() and _count_images(images_subdir) > 0:
        n = _count_images(images_subdir)
        console.print(f"[green]Dataset found: {n} images in {images_subdir}")
        return images_subdir

    raise FileNotFoundError(
        f"No images found in {image_dir} or {image_dir / 'images'}. "
        f"Download a dataset first: bash scripts/download_data.sh [garden|south-building]"
    )


def run_vggt_phase(
    image_dir: Path,
    output_dir: Path,
    max_images: int,
    confidence_threshold: float,
    dtype: str,
) -> dict:
    """Phase 1: Run VGGT 3D reconstruction."""
    from src.reconstruction.vggt_runner import run_vggt

    console.print("\n[bold blue]═══ Phase 1: VGGT 3D Reconstruction ═══")
    console.print(f"  Images: {image_dir}")
    console.print(f"  Max images: {max_images}")
    console.print(f"  Confidence threshold: {confidence_threshold}")
    console.print(f"  Dtype: {dtype}")

    t0 = time.perf_counter()
    result = run_vggt(
        image_dir=image_dir,
        output_dir=output_dir,
        device="cuda",
        max_images=max_images,
        confidence_threshold=confidence_threshold,
        dtype=dtype,
    )
    elapsed = time.perf_counter() - t0

    # Load metadata for detailed timings
    meta_path = output_dir / "vggt_metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    return {
        "n_images": len(result.image_names),
        "n_points": result.num_points,
        "inference_time_s": round(result.inference_time_s, 2),
        "peak_vram_gb": round(result.peak_vram_gb, 2),
        "total_time_s": round(elapsed, 2),
        "timing_detail": metadata.get("timing", {}),
        "point_cloud_path": str(output_dir / "point_cloud.ply"),
        "depth_dir": str(output_dir / "depth_maps"),
        "camera_poses_path": str(output_dir / "camera_poses.json"),
    }


def run_mesh_phase(
    point_cloud_path: Path,
    output_dir: Path,
    target_faces: int,
) -> dict:
    """Phase 2: Run mesh processing pipeline."""
    from src.reconstruction.mesh_processor import GeoReference, process_reconstruction

    console.print("\n[bold blue]═══ Phase 2: Mesh Processing ═══")
    console.print(f"  Point cloud: {point_cloud_path}")
    console.print(f"  Target faces: {target_faces}")

    # Osaka geo-reference
    geo_ref = GeoReference(
        latitude=34.69,
        longitude=135.50,
        altitude=10.0,
    )

    mesh_output = output_dir / "mesh.ply"

    t0 = time.perf_counter()
    result = process_reconstruction(
        point_cloud_path=point_cloud_path,
        output_path=mesh_output,
        method="poisson",
        geo_ref=geo_ref,
        target_faces=target_faces,
    )
    elapsed = time.perf_counter() - t0

    return {
        "n_vertices": result.stats.num_vertices,
        "n_faces": result.stats.num_faces,
        "surface_area_m2": round(result.stats.surface_area_m2, 2),
        "n_roof_faces": result.stats.num_roof_faces,
        "n_wall_faces": result.stats.num_wall_faces,
        "n_ground_faces": result.stats.num_ground_faces,
        "total_time_s": round(elapsed, 2),
        "timing_detail": {k: round(v, 3) for k, v in result.timing_s.items()},
        "mesh_path": str(mesh_output),
    }


def run_solar_phase(mesh_path: Path, output_dir: Path) -> dict:
    """Phase 3: Run solar simulation on reconstructed mesh."""
    import trimesh
    import yaml

    from src.simulation.runner import run_simulation

    console.print("\n[bold blue]═══ Phase 3: Solar Simulation ═══")
    console.print(f"  Mesh: {mesh_path}")

    config_path = PROJECT_ROOT / "configs" / "solar_params.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load the reconstructed mesh
    mesh = trimesh.load(str(mesh_path), process=True)
    console.print(f"  Loaded mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")

    t0 = time.perf_counter()
    irradiance_results, roi_report = run_simulation(mesh, config, output_dir)
    elapsed = time.perf_counter() - t0

    # Count upward-facing faces (potential solar faces)
    up = np.array([0, 0, 1])
    cos_angles = np.dot(mesh.face_normals, up)
    upward_mask = cos_angles > 0.3  # ~72 degrees from horizontal
    upward_area = float(mesh.area_faces[upward_mask].sum())

    return {
        "n_mesh_faces": len(mesh.faces),
        "upward_faces": int(upward_mask.sum()),
        "upward_area_m2": round(upward_area, 2),
        "total_capacity_kw": round(roi_report.total_capacity_kw, 2),
        "annual_generation_kwh": round(roi_report.total_annual_generation_kwh, 1),
        "annual_savings_yen": round(roi_report.total_annual_savings_jpy, 0),
        "payback_years": round(roi_report.overall_payback_years, 1),
        "npv_25y_yen": round(roi_report.overall_npv_25y_jpy, 0),
        "n_proposals": len(roi_report.proposals),
        "total_time_s": round(elapsed, 2),
    }


def print_comparison_table(h100_results: dict):
    """Print a comparison table between H100 and T4 baseline."""
    table = Table(title="H100 vs T4 Benchmark Comparison", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("T4 (Baseline)", justify="right", style="yellow")
    table.add_column("H100", justify="right", style="green")
    table.add_column("Speedup", justify="right", style="cyan")

    vggt = h100_results.get("vggt", {})
    mesh = h100_results.get("mesh", {})
    solar = h100_results.get("solar", {})

    def speedup(t4_val: float, h100_val: float) -> str:
        if h100_val > 0:
            return f"{t4_val / h100_val:.1f}x"
        return "N/A"

    # Images
    table.add_row(
        "Input Images",
        str(T4_BASELINE["n_images"]),
        str(vggt.get("n_images", "?")),
        f"{vggt.get('n_images', 0) / T4_BASELINE['n_images']:.1f}x more" if vggt.get("n_images", 0) > T4_BASELINE["n_images"] else "same",
    )

    # VGGT timing
    table.add_row(
        "VGGT Inference",
        f"{T4_BASELINE['vggt_inference_s']:.1f}s",
        f"{vggt.get('inference_time_s', 0):.1f}s",
        speedup(T4_BASELINE["vggt_inference_s"], vggt.get("inference_time_s", 1)),
    )

    # Peak VRAM
    table.add_row(
        "Peak VRAM",
        f"{T4_BASELINE['vggt_peak_vram_gb']:.1f} GB",
        f"{vggt.get('peak_vram_gb', 0):.1f} GB",
        "",
    )

    # Points
    table.add_row(
        "Point Cloud",
        f"{T4_BASELINE['n_points']:,} pts",
        f"{vggt.get('n_points', 0):,} pts",
        f"{vggt.get('n_points', 0) / T4_BASELINE['n_points']:.1f}x" if vggt.get("n_points", 0) > 0 else "",
    )

    # Mesh processing
    table.add_row(
        "Mesh Processing",
        f"{T4_BASELINE['mesh_processing_s']:.1f}s",
        f"{mesh.get('total_time_s', 0):.1f}s",
        speedup(T4_BASELINE["mesh_processing_s"], mesh.get("total_time_s", 1)),
    )

    # Mesh faces
    table.add_row(
        "Mesh Faces",
        f"{T4_BASELINE['n_faces']:,}",
        f"{mesh.get('n_faces', 0):,}",
        "",
    )

    # Solar sim
    table.add_row(
        "Solar Simulation",
        f"{T4_BASELINE['solar_simulation_s']:.1f}s",
        f"{solar.get('total_time_s', 0):.1f}s",
        speedup(T4_BASELINE["solar_simulation_s"], solar.get("total_time_s", 1)),
    )

    # Total pipeline
    h100_total = vggt.get("total_time_s", 0) + mesh.get("total_time_s", 0) + solar.get("total_time_s", 0)
    table.add_row(
        "Total Pipeline",
        f"{T4_BASELINE['total_s']:.1f}s",
        f"{h100_total:.1f}s",
        speedup(T4_BASELINE["total_s"], h100_total),
    )

    # Solar results
    table.add_row("", "", "", "")
    table.add_row(
        "Annual Savings (¥)",
        f"¥{T4_BASELINE['annual_savings_yen']:,.0f}",
        f"¥{solar.get('annual_savings_yen', 0):,.0f}",
        "",
    )
    table.add_row(
        "Payback Period",
        f"{T4_BASELINE['payback_years']:.1f} years",
        f"{solar.get('payback_years', 0):.1f} years",
        "",
    )

    console.print()
    console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="H100 E2E Pipeline Test: VGGT → Mesh → Solar Simulation"
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "mipnerf360" / "garden",
        help="Path to dataset directory (default: data/raw/mipnerf360/garden)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=20,
        help="Number of images to process (default: 20, T4 used 10)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="VGGT confidence threshold (default: 0.3)",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Model dtype (default: float16)",
    )
    parser.add_argument(
        "--target-faces",
        type=int,
        default=20000,
        help="Target mesh faces after decimation (default: 20000)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "e2e_results" / "h100",
        help="Output directory for results",
    )
    parser.add_argument(
        "--skip-vggt",
        action="store_true",
        help="Skip VGGT (use existing point cloud from output-dir)",
    )
    parser.add_argument(
        "--skip-mesh",
        action="store_true",
        help="Skip mesh processing (use existing mesh from output-dir)",
    )
    args = parser.parse_args()

    console.print("[bold magenta]" + "=" * 60)
    console.print("[bold magenta]  ExaSense H100 E2E Pipeline Test")
    console.print("[bold magenta]" + "=" * 60)

    # System info
    gpu_info = get_gpu_info()
    console.print(f"\n[bold]System Info:")
    console.print(f"  GPU: {gpu_info['gpu_name']}")
    console.print(f"  VRAM: {gpu_info['gpu_memory_gb']} GB")
    console.print(f"  Driver: {gpu_info['driver_version']}")
    console.print(f"  Python: {platform.python_version()}")
    console.print(f"  Date: {datetime.now().isoformat()}")

    # Ensure output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vggt_output = args.output_dir / "vggt"
    mesh_output = args.output_dir / "mesh"
    solar_output = args.output_dir / "solar"

    pipeline_t0 = time.perf_counter()
    results = {
        "test_name": "h100_e2e_pipeline",
        "timestamp": datetime.now().isoformat(),
        "gpu": gpu_info,
        "params": {
            "max_images": args.max_images,
            "confidence_threshold": args.confidence_threshold,
            "dtype": args.dtype,
            "target_faces": args.target_faces,
        },
    }

    # --- Phase 1: VGGT ---
    if not args.skip_vggt:
        image_dir = ensure_dataset(args.image_dir)
        vggt_results = run_vggt_phase(
            image_dir=image_dir,
            output_dir=vggt_output,
            max_images=args.max_images,
            confidence_threshold=args.confidence_threshold,
            dtype=args.dtype,
        )
        results["vggt"] = vggt_results
        point_cloud_path = Path(vggt_results["point_cloud_path"])
    else:
        console.print("\n[yellow]Skipping VGGT (--skip-vggt)")
        point_cloud_path = vggt_output / "point_cloud.ply"
        if not point_cloud_path.exists():
            console.print(f"[red]Error: {point_cloud_path} not found")
            sys.exit(1)
        # Load existing metadata
        meta_path = vggt_output / "vggt_metadata.json"
        if meta_path.exists():
            results["vggt"] = json.loads(meta_path.read_text())

    # --- Phase 2: Mesh ---
    if not args.skip_mesh:
        mesh_results = run_mesh_phase(
            point_cloud_path=point_cloud_path,
            output_dir=mesh_output,
            target_faces=args.target_faces,
        )
        results["mesh"] = mesh_results
        mesh_path = Path(mesh_results["mesh_path"])
    else:
        console.print("\n[yellow]Skipping mesh processing (--skip-mesh)")
        mesh_path = mesh_output / "mesh.ply"
        if not mesh_path.exists():
            console.print(f"[red]Error: {mesh_path} not found")
            sys.exit(1)
        # Load existing metadata
        meta_path = mesh_path.with_suffix(".meta.json")
        if meta_path.exists():
            results["mesh"] = json.loads(meta_path.read_text())

    # --- Phase 3: Solar Simulation ---
    solar_results = run_solar_phase(
        mesh_path=mesh_path,
        output_dir=solar_output,
    )
    results["solar"] = solar_results

    # --- Summary ---
    pipeline_total = time.perf_counter() - pipeline_t0
    results["total_pipeline_s"] = round(pipeline_total, 2)

    # Save results
    results_path = args.output_dir / "h100_e2e_results.json"
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    console.print(f"\n[green]Results saved: {results_path}")

    # Print comparison
    print_comparison_table(results)

    # Summary banner
    console.print("\n[bold green]" + "=" * 60)
    console.print(f"[bold green]  Pipeline complete in {pipeline_total:.1f}s")
    console.print(f"[bold green]  T4 baseline: {T4_BASELINE['total_s']:.1f}s")
    if pipeline_total > 0:
        console.print(f"[bold green]  Speedup: {T4_BASELINE['total_s'] / pipeline_total:.1f}x")
    console.print("[bold green]" + "=" * 60)

    # Output files listing
    console.print("\n[bold]Output files:")
    for p in sorted(args.output_dir.rglob("*")):
        if p.is_file():
            size_mb = p.stat().st_size / (1024 * 1024)
            console.print(f"  {p.relative_to(args.output_dir)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
