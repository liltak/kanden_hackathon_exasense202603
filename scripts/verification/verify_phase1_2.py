#!/usr/bin/env python3
"""Phase 1-2 verification script for AWS GPU instance.

Validates the entire 3D reconstruction + mesh processing pipeline
with a small dataset. Designed to run on g5.xlarge (A10G 24GB) or similar.

Usage:
    python verify_phase1_2.py --dataset synthetic    # Quick test with synthetic data
    python verify_phase1_2.py --dataset mipnerf360    # Real test with garden scene
    python verify_phase1_2.py --all                   # Run all verification steps

Exit codes:
    0 = All checks passed
    1 = Some checks failed (see report)
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "verification"


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    metrics: dict = field(default_factory=dict)
    duration_s: float = 0.0


@dataclass
class VerificationReport:
    checks: list[CheckResult] = field(default_factory=list)
    gpu_info: dict = field(default_factory=dict)
    total_duration_s: float = 0.0

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "all_passed": self.all_passed,
            "gpu_info": self.gpu_info,
            "total_duration_s": round(self.total_duration_s, 1),
            "checks": [asdict(c) for c in self.checks],
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


# ── Check 0: GPU Environment ─────────────────────────────────────────────────

def check_gpu_environment() -> CheckResult:
    """Verify CUDA GPU is available and has sufficient VRAM."""
    t0 = time.time()
    try:
        import torch

        if not torch.cuda.is_available():
            return CheckResult("GPU Environment", False, "CUDA not available")

        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        cuda_ver = torch.version.cuda

        metrics = {
            "gpu_name": name,
            "vram_gb": round(vram_gb, 1),
            "cuda_version": cuda_ver,
            "pytorch_version": torch.__version__,
        }

        if vram_gb < 16:
            return CheckResult(
                "GPU Environment", False,
                f"{name} has {vram_gb:.1f}GB VRAM (need ≥16GB)",
                metrics, time.time() - t0,
            )

        return CheckResult(
            "GPU Environment", True,
            f"{name} ({vram_gb:.0f}GB VRAM, CUDA {cuda_ver})",
            metrics, time.time() - t0,
        )
    except ImportError:
        return CheckResult("GPU Environment", False, "PyTorch not installed",
                           duration_s=time.time() - t0)


# ── Check 1: COLMAP Installation ─────────────────────────────────────────────

def check_colmap() -> CheckResult:
    """Verify COLMAP is installed and working."""
    t0 = time.time()
    try:
        result = subprocess.run(
            ["colmap", "help"], capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return CheckResult("COLMAP", True, "colmap available on PATH",
                               duration_s=time.time() - t0)
        return CheckResult("COLMAP", False, f"colmap returned {result.returncode}",
                           duration_s=time.time() - t0)
    except FileNotFoundError:
        return CheckResult("COLMAP", False, "colmap not found on PATH",
                           duration_s=time.time() - t0)
    except subprocess.TimeoutExpired:
        return CheckResult("COLMAP", False, "colmap timed out",
                           duration_s=time.time() - t0)


# ── Check 2: VGGT Model Load ─────────────────────────────────────────────────

def check_vggt_load() -> CheckResult:
    """Verify VGGT model can be downloaded and loaded."""
    t0 = time.time()
    try:
        import torch
        from transformers import AutoModel

        console.print("  Downloading VGGT model (this may take a few minutes)...")

        torch.cuda.reset_peak_memory_stats()
        vram_before = torch.cuda.memory_allocated() / 1e9

        model = AutoModel.from_pretrained(
            "facebook/VGGT-1B", trust_remote_code=True,
        )
        model = model.to("cuda")

        vram_after = torch.cuda.memory_allocated() / 1e9
        peak_vram = torch.cuda.max_memory_allocated() / 1e9

        metrics = {
            "vram_model_gb": round(vram_after - vram_before, 2),
            "vram_peak_gb": round(peak_vram, 2),
            "load_time_s": round(time.time() - t0, 1),
        }

        # Cleanup
        del model
        torch.cuda.empty_cache()

        return CheckResult(
            "VGGT Model Load", True,
            f"Loaded in {metrics['load_time_s']}s, VRAM: {metrics['vram_model_gb']:.1f}GB",
            metrics, time.time() - t0,
        )
    except Exception as e:
        return CheckResult("VGGT Model Load", False, str(e),
                           duration_s=time.time() - t0)


# ── Check 3: VGGT Inference ──────────────────────────────────────────────────

def check_vggt_inference(image_dir: Path) -> CheckResult:
    """Run VGGT inference on test images."""
    t0 = time.time()
    try:
        import torch
        from PIL import Image
        from transformers import AutoModel

        images = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
        if len(images) < 2:
            return CheckResult("VGGT Inference", False,
                               f"Need ≥2 images, found {len(images)} in {image_dir}",
                               duration_s=time.time() - t0)

        # Limit to 10 images for verification
        images = images[:10]
        console.print(f"  Running VGGT on {len(images)} images...")

        torch.cuda.reset_peak_memory_stats()
        model = AutoModel.from_pretrained("facebook/VGGT-1B", trust_remote_code=True)
        model = model.to("cuda")

        # Load and preprocess images
        pil_images = [Image.open(p).convert("RGB") for p in images]

        # Run inference
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            predictions = model.infer(pil_images)

        peak_vram = torch.cuda.max_memory_allocated() / 1e9
        infer_time = time.time() - t0

        # Check outputs
        has_points = "world_points" in predictions or hasattr(predictions, "world_points")
        has_depth = "depth" in predictions or hasattr(predictions, "depth")

        metrics = {
            "n_images": len(images),
            "inference_time_s": round(infer_time, 1),
            "peak_vram_gb": round(peak_vram, 2),
            "has_point_cloud": has_points,
            "has_depth_maps": has_depth,
            "output_keys": list(predictions.keys()) if isinstance(predictions, dict) else "N/A",
        }

        del model, predictions
        torch.cuda.empty_cache()

        return CheckResult(
            "VGGT Inference", True,
            f"{len(images)} images in {infer_time:.1f}s, peak VRAM: {peak_vram:.1f}GB",
            metrics, time.time() - t0,
        )
    except Exception as e:
        return CheckResult("VGGT Inference", False, str(e),
                           duration_s=time.time() - t0)


# ── Check 4: COLMAP SfM ──────────────────────────────────────────────────────

def check_colmap_sfm(image_dir: Path) -> CheckResult:
    """Run COLMAP SfM pipeline on test images."""
    t0 = time.time()
    try:
        from src.reconstruction.colmap_runner import run_colmap

        output_dir = RESULTS_DIR / "colmap_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"  Running COLMAP SfM on {image_dir}...")
        result = run_colmap(
            image_dir=image_dir,
            workspace_dir=output_dir,
            camera_model="OPENCV",
            matcher="exhaustive",
        )

        metrics = {
            "n_cameras": result.n_cameras,
            "n_images": result.n_images,
            "n_points3d": result.n_points3d,
            "total_time_s": round(result.total_time_s, 1),
            "step_timings": {s.name: round(s.duration_s, 1) for s in result.steps},
        }

        if result.n_points3d < 10:
            return CheckResult(
                "COLMAP SfM", False,
                f"Only {result.n_points3d} 3D points reconstructed",
                metrics, time.time() - t0,
            )

        return CheckResult(
            "COLMAP SfM", True,
            f"{result.n_images} images → {result.n_points3d} 3D points in {result.total_time_s:.0f}s",
            metrics, time.time() - t0,
        )
    except Exception as e:
        return CheckResult("COLMAP SfM", False, str(e),
                           duration_s=time.time() - t0)


# ── Check 5: Mesh Processing ─────────────────────────────────────────────────

def check_mesh_processing() -> CheckResult:
    """Verify mesh processing pipeline with synthetic point cloud."""
    t0 = time.time()
    try:
        import trimesh

        from src.reconstruction.mesh_processor import MeshProcessor

        # Create a synthetic point cloud (hemisphere)
        n_points = 5000
        phi = np.random.uniform(0, 2 * np.pi, n_points)
        theta = np.random.uniform(0, np.pi / 2, n_points)
        r = 10 + np.random.normal(0, 0.1, n_points)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        points = np.column_stack([x, y, z])
        normals = points / np.linalg.norm(points, axis=1, keepdims=True)

        # Save as PLY
        cloud_path = RESULTS_DIR / "synthetic_cloud.ply"
        cloud_path.parent.mkdir(parents=True, exist_ok=True)
        cloud = trimesh.PointCloud(points)
        cloud.export(str(cloud_path))

        console.print("  Processing synthetic point cloud → mesh...")

        processor = MeshProcessor()
        processor.load_point_cloud(str(cloud_path))
        processor.extract_mesh(method="poisson")

        mesh = processor.mesh
        if mesh is None:
            return CheckResult("Mesh Processing", False, "No mesh produced",
                               duration_s=time.time() - t0)

        processor.fix_normals()
        processor.segment_faces()

        output_path = RESULTS_DIR / "synthetic_mesh.obj"
        processor.save(str(output_path))

        metrics = {
            "input_points": n_points,
            "output_vertices": len(mesh.vertices),
            "output_faces": len(mesh.faces),
            "is_watertight": mesh.is_watertight,
            "has_labels": processor.labels is not None,
        }

        return CheckResult(
            "Mesh Processing", True,
            f"{n_points} points → {len(mesh.faces)} faces",
            metrics, time.time() - t0,
        )
    except Exception as e:
        return CheckResult("Mesh Processing", False, str(e),
                           duration_s=time.time() - t0)


# ── Check 6: Phase 2→3 Pipeline Connection ───────────────────────────────────

def check_pipeline_connection() -> CheckResult:
    """Verify Phase 2 mesh output can be consumed by Phase 3 simulation."""
    t0 = time.time()
    try:
        import trimesh

        from src.simulation.runner import load_config, run_simulation

        # Use demo factory mesh as stand-in for reconstructed mesh
        from src.simulation.demo_factory import create_factory_complex

        mesh = create_factory_complex()
        config = load_config(PROJECT_ROOT / "configs" / "solar_params.yaml")

        # Run with coarse time resolution for speed
        config["simulation"]["time_resolution_minutes"] = 180  # 3-hour steps
        output_dir = RESULTS_DIR / "pipeline_test"

        console.print("  Running Phase 3 simulation on test mesh...")
        irr, roi = run_simulation(mesh, config, output_dir)

        metrics = {
            "mesh_faces": len(mesh.faces),
            "irradiance_faces": len(irr),
            "suitable_panels": len(roi.proposals),
            "total_capacity_kw": roi.total_capacity_kw,
            "payback_years": roi.overall_payback_years,
        }

        if len(irr) != len(mesh.faces):
            return CheckResult(
                "Pipeline Connection", False,
                f"Face count mismatch: mesh={len(mesh.faces)}, irradiance={len(irr)}",
                metrics, time.time() - t0,
            )

        return CheckResult(
            "Pipeline Connection", True,
            f"Phase 2→3 OK: {len(mesh.faces)} faces → {len(roi.proposals)} panel proposals",
            metrics, time.time() - t0,
        )
    except Exception as e:
        return CheckResult("Pipeline Connection", False, str(e),
                           duration_s=time.time() - t0)


# ── Synthetic Test Data ───────────────────────────────────────────────────────

def create_synthetic_images(output_dir: Path, n_images: int = 8) -> Path:
    """Create synthetic test images (checkerboard patterns from different angles)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        # Fallback: create simple numpy arrays saved as raw
        console.print("[yellow]PIL not available, creating minimal test images")
        for i in range(n_images):
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            # Save as PPM (no PIL needed)
            path = output_dir / f"test_{i:03d}.ppm"
            with open(path, "wb") as f:
                f.write(f"P6\n640 480\n255\n".encode())
                f.write(img.tobytes())
        return output_dir

    for i in range(n_images):
        img = Image.new("RGB", (640, 480), "white")
        draw = ImageDraw.Draw(img)

        # Draw a 3D-looking checkerboard with perspective shift
        offset = i * 20
        for y in range(0, 480, 40):
            for x in range(0, 640, 40):
                if (x // 40 + y // 40) % 2 == 0:
                    draw.rectangle(
                        [x + offset % 40, y, x + 40 + offset % 40, y + 40],
                        fill=(50, 50, 200),
                    )

        # Add some distinctive features
        cx, cy = 320 + offset, 240
        draw.ellipse([cx - 30, cy - 30, cx + 30, cy + 30], fill="red")
        draw.rectangle([100 + offset, 100, 200 + offset, 200], fill="green")

        img.save(output_dir / f"test_{i:03d}.jpg", quality=95)

    console.print(f"  Created {n_images} synthetic test images in {output_dir}")
    return output_dir


# ── Main ──────────────────────────────────────────────────────────────────────

def run_verification(
    dataset: str = "synthetic",
    run_all: bool = False,
    skip_gpu: bool = False,
) -> VerificationReport:
    """Run Phase 1-2 verification suite."""
    report = VerificationReport()
    t_total = time.time()

    console.print(Panel.fit(
        "[bold blue]ExaSense Phase 1-2 Verification[/]\n"
        f"Dataset: {dataset} | Skip GPU: {skip_gpu}",
        border_style="blue",
    ))

    # GPU check
    gpu_result = check_gpu_environment()
    report.checks.append(gpu_result)
    report.gpu_info = gpu_result.metrics
    _print_check(gpu_result)

    has_gpu = gpu_result.passed

    # COLMAP check
    colmap_result = check_colmap()
    report.checks.append(colmap_result)
    _print_check(colmap_result)

    # Prepare test images
    if dataset == "synthetic":
        image_dir = create_synthetic_images(RESULTS_DIR / "synthetic_images")
    elif dataset == "mipnerf360":
        image_dir = PROJECT_ROOT / "data" / "raw" / "mipnerf360" / "garden" / "images"
        if not image_dir.exists():
            console.print("[yellow]Mip-NeRF 360 not found. Run scripts/download_data.sh first")
            console.print("[yellow]Falling back to synthetic data")
            image_dir = create_synthetic_images(RESULTS_DIR / "synthetic_images")
    else:
        image_dir = Path(dataset)

    # VGGT checks (GPU required)
    if has_gpu and not skip_gpu:
        vggt_load = check_vggt_load()
        report.checks.append(vggt_load)
        _print_check(vggt_load)

        if vggt_load.passed and run_all:
            vggt_infer = check_vggt_inference(image_dir)
            report.checks.append(vggt_infer)
            _print_check(vggt_infer)
    else:
        report.checks.append(CheckResult(
            "VGGT Model Load", False, "Skipped (no GPU or --skip-gpu)",
        ))

    # COLMAP SfM (needs real images + colmap)
    if colmap_result.passed and dataset != "synthetic":
        sfm_result = check_colmap_sfm(image_dir)
        report.checks.append(sfm_result)
        _print_check(sfm_result)
    else:
        report.checks.append(CheckResult(
            "COLMAP SfM", False,
            "Skipped (no COLMAP or synthetic dataset)",
        ))

    # Mesh processing (can run without GPU using Open3D)
    mesh_result = check_mesh_processing()
    report.checks.append(mesh_result)
    _print_check(mesh_result)

    # Pipeline connection test
    pipe_result = check_pipeline_connection()
    report.checks.append(pipe_result)
    _print_check(pipe_result)

    report.total_duration_s = time.time() - t_total

    # Summary
    _print_summary(report)

    # Save report
    report_path = RESULTS_DIR / "verification_report.json"
    report.save(report_path)
    console.print(f"\nReport saved to: {report_path}")

    return report


def _print_check(result: CheckResult):
    status = "[green]PASS" if result.passed else "[red]FAIL"
    console.print(f"  {status}[/] {result.name}: {result.message}")
    if result.duration_s > 0:
        console.print(f"        ({result.duration_s:.1f}s)")


def _print_summary(report: VerificationReport):
    console.print()
    table = Table(title="Verification Summary")
    table.add_column("Check", style="bold")
    table.add_column("Status")
    table.add_column("Time")
    table.add_column("Details")

    for c in report.checks:
        status = "[green]PASS" if c.passed else "[red]FAIL"
        table.add_row(
            c.name, status,
            f"{c.duration_s:.1f}s" if c.duration_s > 0 else "-",
            c.message,
        )

    console.print(table)

    passed = sum(1 for c in report.checks if c.passed)
    total = len(report.checks)
    color = "green" if report.all_passed else "yellow"
    console.print(f"\n[{color}]{passed}/{total} checks passed "
                  f"({report.total_duration_s:.0f}s total)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ExaSense Phase 1-2 Verification")
    parser.add_argument(
        "--dataset", default="synthetic",
        help="Test dataset: synthetic, mipnerf360, or path to image directory",
    )
    parser.add_argument("--all", action="store_true", help="Run all checks including heavy inference")
    parser.add_argument("--skip-gpu", action="store_true", help="Skip GPU-dependent checks")
    args = parser.parse_args()

    report = run_verification(
        dataset=args.dataset,
        run_all=args.all,
        skip_gpu=args.skip_gpu,
    )

    sys.exit(0 if report.all_passed else 1)
