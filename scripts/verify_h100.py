#!/usr/bin/env python3
"""H100 GPU Quick Verification — check all components before E2E run.

Verifies GPU environment, VGGT, Open3D, VLM (Qwen2.5-VL), and Phase 3
in ~2-5 minutes. Run this first before the full E2E pipeline.

Usage:
    uv run python scripts/verify_h100.py
    uv run python scripts/verify_h100.py --skip-vlm    # Skip VLM (faster)
    uv run python scripts/verify_h100.py --all          # Include inference tests
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

console = Console()


@dataclass
class Check:
    name: str
    passed: bool
    message: str
    metrics: dict = field(default_factory=dict)
    duration_s: float = 0.0


def _status(c: Check) -> str:
    return "[green]PASS" if c.passed else "[red]FAIL"


# ── 1. GPU Environment ─────────────────────────────────────────────────────

def check_gpu() -> Check:
    t0 = time.time()
    try:
        import torch

        if not torch.cuda.is_available():
            return Check("GPU", False, "CUDA not available", duration_s=time.time() - t0)

        n_gpus = torch.cuda.device_count()
        gpus = []
        for i in range(n_gpus):
            name = torch.cuda.get_device_name(i)
            vram = torch.cuda.get_device_properties(i).total_memory / 1e9
            gpus.append({"id": i, "name": name, "vram_gb": round(vram, 1)})

        metrics = {
            "n_gpus": n_gpus,
            "gpus": gpus,
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
        }

        gpu_list = ", ".join(f"{g['name']} ({g['vram_gb']}GB)" for g in gpus)
        return Check("GPU", True, f"{n_gpus}x {gpu_list}, CUDA {torch.version.cuda}",
                      metrics, time.time() - t0)
    except ImportError:
        return Check("GPU", False, "PyTorch not installed", duration_s=time.time() - t0)


# ── 2. VGGT Model ──────────────────────────────────────────────────────────

def check_vggt_load() -> Check:
    t0 = time.time()
    try:
        import torch
        from transformers import AutoModel

        torch.cuda.reset_peak_memory_stats()
        vram_before = torch.cuda.memory_allocated() / 1e9

        model = AutoModel.from_pretrained(
            "facebook/VGGT-1B-Commercial", trust_remote_code=True,
        )
        model = model.to("cuda")

        vram_model = torch.cuda.memory_allocated() / 1e9 - vram_before
        peak_vram = torch.cuda.max_memory_allocated() / 1e9

        metrics = {
            "vram_model_gb": round(vram_model, 2),
            "vram_peak_gb": round(peak_vram, 2),
            "load_time_s": round(time.time() - t0, 1),
        }

        del model
        torch.cuda.empty_cache()

        return Check("VGGT Load", True,
                      f"Loaded in {metrics['load_time_s']}s, VRAM {vram_model:.1f}GB",
                      metrics, time.time() - t0)
    except Exception as e:
        return Check("VGGT Load", False, str(e), duration_s=time.time() - t0)


# ── 3. Open3D ──────────────────────────────────────────────────────────────

def check_open3d() -> Check:
    t0 = time.time()
    try:
        import open3d as o3d

        # Create synthetic point cloud and run Poisson
        n = 2000
        theta = np.random.uniform(0, 2 * np.pi, n)
        phi = np.random.uniform(0, np.pi / 2, n)
        r = 5.0
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        points = np.column_stack([x, y, z])
        normals = points / np.linalg.norm(points, axis=1, keepdims=True)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)

        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=5)

        metrics = {
            "open3d_version": o3d.__version__,
            "vertices": len(mesh.vertices),
            "faces": len(mesh.triangles),
        }

        return Check("Open3D", True,
                      f"v{o3d.__version__}, Poisson OK ({len(mesh.triangles)} faces)",
                      metrics, time.time() - t0)
    except ImportError:
        return Check("Open3D", False, "open3d not installed", duration_s=time.time() - t0)
    except Exception as e:
        return Check("Open3D", False, str(e), duration_s=time.time() - t0)


# ── 4. VLM (Qwen2.5-VL) ──────────────────────────────────────────────────

def check_vlm_load() -> Check:
    t0 = time.time()
    try:
        import torch

        torch.cuda.reset_peak_memory_stats()
        vram_before = torch.cuda.memory_allocated() / 1e9

        from src.vlm.model_loader import ModelConfig, load_model

        config = ModelConfig(torch_dtype="bfloat16")
        model, processor = load_model(config)

        vram_model = torch.cuda.memory_allocated() / 1e9 - vram_before
        peak_vram = torch.cuda.max_memory_allocated() / 1e9

        metrics = {
            "model_id": config.model_id,
            "vram_model_gb": round(vram_model, 2),
            "vram_peak_gb": round(peak_vram, 2),
            "load_time_s": round(time.time() - t0, 1),
        }

        del model, processor
        torch.cuda.empty_cache()

        return Check("VLM Load", True,
                      f"Qwen2.5-VL loaded in {metrics['load_time_s']}s, VRAM {vram_model:.1f}GB",
                      metrics, time.time() - t0)
    except Exception as e:
        return Check("VLM Load", False, str(e), duration_s=time.time() - t0)


def check_vlm_inference() -> Check:
    """Quick VLM inference test with a synthetic image."""
    t0 = time.time()
    try:
        from PIL import Image

        from src.vlm.inference import InferenceRequest, VLMPipeline

        pipeline = VLMPipeline()
        pipeline.load()

        # Create a simple test image
        test_img = Image.new("RGB", (640, 480), color=(100, 150, 200))

        request = InferenceRequest(
            images=[test_img],
            prompt_template="general_qa",
            context_data="テスト: この画像を説明してください",
            custom_prompt="この画像を一文で説明してください。",
            max_new_tokens=128,
            temperature=0.3,
        )

        result = pipeline.infer(request)

        import torch
        peak_vram = torch.cuda.max_memory_allocated() / 1e9

        metrics = {
            "output_tokens": result.output_tokens,
            "latency_s": round(result.latency_seconds, 2),
            "throughput_tok_s": round(result.output_tokens / max(result.latency_seconds, 0.01), 1),
            "vram_peak_gb": round(peak_vram, 2),
        }

        del pipeline
        torch.cuda.empty_cache()

        return Check("VLM Inference", True,
                      f"{result.output_tokens} tokens in {result.latency_seconds:.1f}s "
                      f"({metrics['throughput_tok_s']} tok/s)",
                      metrics, time.time() - t0)
    except Exception as e:
        return Check("VLM Inference", False, str(e), duration_s=time.time() - t0)


# ── 5. Phase 3 Simulation ─────────────────────────────────────────────────

def check_simulation() -> Check:
    t0 = time.time()
    try:
        from src.simulation.demo_factory import create_factory_complex
        from src.simulation.runner import load_config, run_simulation

        mesh = create_factory_complex()
        config = load_config(PROJECT_ROOT / "configs" / "solar_params.yaml")
        config["simulation"]["time_resolution_minutes"] = 180  # fast

        output_dir = PROJECT_ROOT / "data" / "verification" / "sim_test"
        irr, roi = run_simulation(mesh, config, output_dir)

        metrics = {
            "n_faces": len(mesh.faces),
            "n_proposals": len(roi.proposals),
            "capacity_kw": round(roi.total_capacity_kw, 1),
            "payback_years": round(roi.overall_payback_years, 1),
        }

        return Check("Solar Sim", True,
                      f"{len(roi.proposals)} proposals, {roi.total_capacity_kw:.0f}kW, "
                      f"payback {roi.overall_payback_years:.1f}y",
                      metrics, time.time() - t0)
    except Exception as e:
        return Check("Solar Sim", False, str(e), duration_s=time.time() - t0)


# ── 6. Embree Ray Backend ─────────────────────────────────────────────────

def check_embree() -> Check:
    t0 = time.time()
    try:
        from trimesh.ray import ray_pyembree  # noqa: F401
        return Check("Embree", True, "Accelerated ray backend available",
                      duration_s=time.time() - t0)
    except ImportError:
        return Check("Embree", False, "Not available (using trimesh native backend)",
                      duration_s=time.time() - t0)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="H100 Quick Verification")
    parser.add_argument("--skip-vlm", action="store_true", help="Skip VLM checks")
    parser.add_argument("--all", action="store_true", help="Include inference tests")
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold blue]ExaSense H100 Quick Verification[/]\n"
        f"Date: {datetime.now().isoformat()}\n"
        f"Skip VLM: {args.skip_vlm} | Full: {args.all}",
        border_style="blue",
    ))

    checks: list[Check] = []

    # GPU
    gpu = check_gpu()
    checks.append(gpu)
    console.print(f"  {_status(gpu)}[/] {gpu.name}: {gpu.message}")

    if not gpu.passed:
        console.print("[red]GPU check failed. Cannot continue.")
        sys.exit(1)

    # Open3D
    o3d = check_open3d()
    checks.append(o3d)
    console.print(f"  {_status(o3d)}[/] {o3d.name}: {o3d.message}")

    # Embree
    embree = check_embree()
    checks.append(embree)
    console.print(f"  {_status(embree)}[/] {embree.name}: {embree.message}")

    # Solar simulation
    sim = check_simulation()
    checks.append(sim)
    console.print(f"  {_status(sim)}[/] {sim.name}: {sim.message}")

    # VGGT
    vggt = check_vggt_load()
    checks.append(vggt)
    console.print(f"  {_status(vggt)}[/] {vggt.name}: {vggt.message}")

    # VLM
    if not args.skip_vlm:
        vlm = check_vlm_load()
        checks.append(vlm)
        console.print(f"  {_status(vlm)}[/] {vlm.name}: {vlm.message}")

        if args.all and vlm.passed:
            vlm_infer = check_vlm_inference()
            checks.append(vlm_infer)
            console.print(f"  {_status(vlm_infer)}[/] {vlm_infer.name}: {vlm_infer.message}")
    else:
        checks.append(Check("VLM Load", False, "Skipped (--skip-vlm)"))

    # Summary table
    console.print()
    table = Table(title="H100 Verification Summary")
    table.add_column("Check", style="bold")
    table.add_column("Status")
    table.add_column("Time")
    table.add_column("Details")

    for c in checks:
        table.add_row(
            c.name,
            "[green]PASS" if c.passed else "[red]FAIL",
            f"{c.duration_s:.1f}s" if c.duration_s > 0 else "-",
            c.message,
        )

    console.print(table)

    passed = sum(1 for c in checks if c.passed)
    total = len(checks)
    color = "green" if passed == total else "yellow"
    console.print(f"\n[{color}]{passed}/{total} checks passed")

    # Save report
    report_dir = PROJECT_ROOT / "data" / "verification"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "h100_verification.json"
    report = {
        "timestamp": datetime.now().isoformat(),
        "all_passed": passed == total,
        "checks": [asdict(c) for c in checks],
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    console.print(f"\nReport saved: {report_path}")

    # Next steps
    if passed == total:
        console.print("\n[bold green]All checks passed! Run full E2E pipeline:")
        console.print("  bash scripts/run_h100_e2e.sh --max-images 20")
    else:
        console.print("\n[bold yellow]Some checks failed. Fix issues before running E2E.")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
