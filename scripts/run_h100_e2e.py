#!/usr/bin/env python3
"""H100 E2E Pipeline Test — Phase 1→2→3→4 (VGGT→Mesh→Solar→VLM).

Runs the full reconstruction + simulation + VLM analysis pipeline on H100 GPU,
benchmarks performance, and compares with T4 baseline results.

Usage:
    uv run python scripts/run_h100_e2e.py [--image-dir DATA] [--max-images 20]
    uv run python scripts/run_h100_e2e.py --skip-vlm          # Skip VLM (Phase 4)
    uv run python scripts/run_h100_e2e.py --expected-extent 50 # 50m building
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
    foreground_mask: str | None = None,
    foreground_depth_sigma: float = 2.0,
) -> dict:
    """Phase 1: Run VGGT 3D reconstruction."""
    from src.reconstruction.vggt_runner import run_vggt

    console.print("\n[bold blue]═══ Phase 1: VGGT 3D Reconstruction ═══")
    console.print(f"  Images: {image_dir}")
    console.print(f"  Max images: {max_images}")
    console.print(f"  Confidence threshold: {confidence_threshold}")
    console.print(f"  Dtype: {dtype}")
    if foreground_mask:
        console.print(f"  Foreground mask: {foreground_mask} (σ={foreground_depth_sigma})")

    t0 = time.perf_counter()
    result = run_vggt(
        image_dir=image_dir,
        output_dir=output_dir,
        device="cuda",
        max_images=max_images,
        confidence_threshold=confidence_threshold,
        dtype=dtype,
        foreground_mask=foreground_mask,
        foreground_depth_sigma=foreground_depth_sigma,
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


def estimate_scale_factor(
    point_cloud_path: Path,
    expected_extent_m: float = 30.0,
) -> float:
    """Estimate scale factor from point cloud bounding box.

    VGGT outputs coordinates in normalized/arbitrary space.
    We estimate the real-world scale by comparing the point cloud
    extent to an expected building footprint size.

    Args:
        point_cloud_path: Path to the point cloud PLY file.
        expected_extent_m: Expected real-world extent in meters
            (default 30m, typical factory building).

    Returns:
        Scale factor to convert local units to meters.
    """
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(point_cloud_path))
    points = np.asarray(pcd.points)

    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    extents = bbox_max - bbox_min
    max_extent = float(extents.max())

    if max_extent < 1e-6:
        console.print("[yellow]  Warning: point cloud has near-zero extent")
        return 1.0

    scale = expected_extent_m / max_extent
    console.print(f"  Point cloud extent: {extents}")
    console.print(f"  Max extent: {max_extent:.4f} (local units)")
    console.print(f"  Expected: {expected_extent_m}m → scale_factor = {scale:.2f}")
    return round(scale, 2)


def run_mesh_phase(
    point_cloud_path: Path,
    output_dir: Path,
    target_faces: int,
    expected_extent_m: float = 30.0,
    mesh_method: str = "poisson",
) -> dict:
    """Phase 2: Run mesh processing pipeline."""
    from src.reconstruction.mesh_processor import GeoReference, process_reconstruction

    console.print("\n[bold blue]═══ Phase 2: Mesh Processing ═══")
    console.print(f"  Point cloud: {point_cloud_path}")
    console.print(f"  Target faces: {target_faces}")
    console.print(f"  Method: {mesh_method}")

    # Auto-estimate scale factor from point cloud bounding box
    scale_factor = estimate_scale_factor(point_cloud_path, expected_extent_m)

    # Osaka geo-reference with auto-estimated scale
    geo_ref = GeoReference(
        latitude=34.69,
        longitude=135.50,
        altitude=10.0,
        scale_factor=scale_factor,
    )

    mesh_output = output_dir / "mesh.ply"

    t0 = time.perf_counter()
    result = process_reconstruction(
        point_cloud_path=point_cloud_path,
        output_path=mesh_output,
        method=mesh_method,
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
        "scale_factor": scale_factor,
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


def run_vlm_phase(
    mesh_path: Path,
    solar_output_dir: Path,
    output_dir: Path,
) -> dict:
    """Phase 4: Run VLM analysis on solar simulation results."""
    import torch
    from PIL import Image

    from src.vlm.inference import InferenceRequest, VLMPipeline

    console.print("\n[bold blue]═══ Phase 4: VLM Analysis (Qwen2.5-VL) ═══")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect input images for VLM
    images: list[Image.Image] = []
    image_paths: list[str] = []

    # Look for heatmap from solar simulation
    heatmap_candidates = [
        solar_output_dir / "irradiance_heatmap.png",
        solar_output_dir / "heatmap.png",
        solar_output_dir / "solar_heatmap.png",
    ]
    for candidate in heatmap_candidates:
        if candidate.exists():
            images.append(Image.open(candidate).convert("RGB"))
            image_paths.append(str(candidate))
            console.print(f"  Heatmap: {candidate}")
            break

    # If no heatmap available, create a placeholder test image
    if not images:
        console.print("  [yellow]No heatmap found, creating test image")
        test_img = Image.new("RGB", (640, 480), color=(100, 150, 200))
        images.append(test_img)
        image_paths.append("synthetic_test_image")

    # Load VLM pipeline
    torch.cuda.reset_peak_memory_stats()
    vram_before = torch.cuda.memory_allocated() / 1e9

    t0 = time.perf_counter()
    pipeline = VLMPipeline()
    pipeline.load()
    load_time = time.perf_counter() - t0

    vram_after_load = torch.cuda.memory_allocated() / 1e9
    console.print(f"  Model loaded in {load_time:.1f}s")
    console.print(f"  VRAM for model: {vram_after_load - vram_before:.1f} GB")

    # Run inference
    context_data = json.dumps({
        "location": "Osaka, Japan (34.69°N, 135.50°E)",
        "building_type": "Industrial factory",
        "note": "E2E pipeline test",
    }, ensure_ascii=False)

    request = InferenceRequest(
        images=images,
        prompt_template="panel_placement",
        context_data=context_data,
        max_new_tokens=1024,
        temperature=0.7,
    )

    t1 = time.perf_counter()
    result = pipeline.infer(request)
    infer_time = time.perf_counter() - t1

    peak_vram = torch.cuda.max_memory_allocated() / 1e9

    console.print(f"  Inference: {infer_time:.1f}s")
    console.print(f"  Output tokens: {result.output_tokens}")
    console.print(f"  Peak VRAM: {peak_vram:.1f} GB")
    console.print(f"  Throughput: {result.output_tokens / infer_time:.0f} tok/s")

    # Save VLM output
    report_path = output_dir / "vlm_report.md"
    report_path.write_text(result.text, encoding="utf-8")
    console.print(f"  Report saved: {report_path}")

    # Preview first 200 chars
    preview = result.text[:200].replace("\n", " ")
    console.print(f"  Preview: {preview}...")

    # Cleanup to free VRAM for potential further use
    del pipeline
    torch.cuda.empty_cache()

    return {
        "model_id": result.model_id,
        "load_time_s": round(load_time, 2),
        "inference_time_s": round(infer_time, 2),
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "throughput_tok_s": round(result.output_tokens / infer_time, 1),
        "vram_model_gb": round(vram_after_load - vram_before, 2),
        "vram_peak_gb": round(peak_vram, 2),
        "n_images": result.images_count,
        "report_path": str(report_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="H100 E2E Pipeline Test: VGGT → Mesh → Solar → VLM"
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
        "--mesh-method",
        choices=["poisson", "kaolin_dmtet"],
        default="poisson",
        help="Mesh reconstruction method (default: poisson)",
    )
    parser.add_argument(
        "--expected-extent",
        type=float,
        default=30.0,
        help="Expected real-world extent in meters for scale estimation (default: 30m)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "e2e_results" / "h100",
        help="Output directory for results",
    )
    parser.add_argument(
        "--foreground-mask",
        choices=["depth", "semantic", "sam", "sam+depth", "both"],
        default=None,
        help="Foreground extraction method for VGGT point cloud filtering",
    )
    parser.add_argument(
        "--foreground-depth-sigma",
        type=float,
        default=2.0,
        help="Std dev threshold for depth-based foreground filtering (default: 2.0)",
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
    parser.add_argument(
        "--skip-vlm",
        action="store_true",
        help="Skip VLM analysis (Phase 4)",
    )
    args = parser.parse_args()

    console.print("[bold magenta]" + "=" * 60)
    console.print("[bold magenta]  ExaSense H100 E2E Pipeline Test")
    console.print("[bold magenta]  Phase 1→2→3→4 Full Pipeline")
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
    vlm_output = args.output_dir / "vlm"

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
            "expected_extent_m": args.expected_extent,
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
            foreground_mask=args.foreground_mask,
            foreground_depth_sigma=args.foreground_depth_sigma,
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
            expected_extent_m=args.expected_extent,
            mesh_method=args.mesh_method,
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

    # --- Phase 4: VLM Analysis ---
    if not args.skip_vlm:
        try:
            vlm_results = run_vlm_phase(
                mesh_path=mesh_path,
                solar_output_dir=solar_output,
                output_dir=vlm_output,
            )
            results["vlm"] = vlm_results
        except Exception as e:
            console.print(f"\n[red]Phase 4 VLM failed: {e}")
            results["vlm"] = {"error": str(e)}
    else:
        console.print("\n[yellow]Skipping VLM (--skip-vlm)")

    # --- Summary ---
    pipeline_total = time.perf_counter() - pipeline_t0
    results["total_pipeline_s"] = round(pipeline_total, 2)

    # Save results
    results_path = args.output_dir / "h100_e2e_results.json"
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    console.print(f"\n[green]Results saved: {results_path}")

    # Print comparison
    print_comparison_table(results)

    # VLM summary
    vlm = results.get("vlm", {})
    if vlm and "error" not in vlm:
        console.print(f"\n[bold]VLM Analysis:")
        console.print(f"  Model: {vlm.get('model_id', '?')}")
        console.print(f"  Load: {vlm.get('load_time_s', 0):.1f}s")
        console.print(f"  Inference: {vlm.get('inference_time_s', 0):.1f}s")
        console.print(f"  Throughput: {vlm.get('throughput_tok_s', 0):.0f} tok/s")
        console.print(f"  VRAM peak: {vlm.get('vram_peak_gb', 0):.1f} GB")

    # Summary banner
    console.print("\n[bold green]" + "=" * 60)
    console.print(f"[bold green]  Pipeline complete in {pipeline_total:.1f}s")
    console.print(f"[bold green]  T4 baseline: {T4_BASELINE['total_s']:.1f}s (Phase 1-3 only)")
    phases_13_total = (
        results.get("vggt", {}).get("total_time_s", 0)
        + results.get("mesh", {}).get("total_time_s", 0)
        + results.get("solar", {}).get("total_time_s", 0)
    )
    if phases_13_total > 0:
        console.print(f"[bold green]  Phase 1-3 speedup: {T4_BASELINE['total_s'] / phases_13_total:.1f}x")
    console.print("[bold green]" + "=" * 60)

    # Output files listing
    console.print("\n[bold]Output files:")
    for p in sorted(args.output_dir.rglob("*")):
        if p.is_file():
            size_mb = p.stat().st_size / (1024 * 1024)
            console.print(f"  {p.relative_to(args.output_dir)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
