"""OpenSplat 3D Gaussian Splatting trainer.

Runs OpenSplat training on COLMAP sparse reconstruction output,
producing a 3D Gaussian Splatting .ply file suitable for real-time rendering.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from typing import TYPE_CHECKING
from rich.console import Console

if TYPE_CHECKING:
    import torch
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class OpenSplatResult:
    """Structured result from OpenSplat training."""

    output_ply: Path | None = None
    num_gaussians: int = 0
    training_time_s: float = 0.0
    peak_vram_gb: float = 0.0
    final_loss: float = 0.0
    num_iterations: int = 0
    checkpoint_dir: Path | None = None
    success: bool = False
    step_timings: list[dict] = field(default_factory=list)


def _get_peak_vram_gb() -> float:
    """Return peak VRAM usage in GB if CUDA is available."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**3)
    return 0.0


def _check_opensplat_available() -> str:
    """Check that opensplat is available on PATH."""
    result = subprocess.run(
        ["opensplat", "--help"],
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(
            "opensplat not found on PATH. Install OpenSplat and ensure it is accessible.\n"
            "See: https://github.com/pierotofy/OpenSplat"
        )
    return "available"


def _find_colmap_model(colmap_dir: Path) -> Path:
    """Find the COLMAP sparse model directory within a workspace.

    Searches for the standard COLMAP output structure:
    colmap_dir/sparse/0/ or colmap_dir/0/ etc.

    Args:
        colmap_dir: COLMAP workspace or sparse model directory.

    Returns:
        Path to the model directory containing cameras.bin/txt.

    Raises:
        FileNotFoundError: If no valid COLMAP model is found.
    """
    candidates = [
        colmap_dir,
        colmap_dir / "sparse" / "0",
        colmap_dir / "sparse",
        colmap_dir / "0",
    ]

    for d in candidates:
        if not d.is_dir():
            continue
        has_cameras = (d / "cameras.bin").exists() or (d / "cameras.txt").exists()
        has_images = (d / "images.bin").exists() or (d / "images.txt").exists()
        if has_cameras and has_images:
            return d

    # Search recursively for sparse model directories
    for d in sorted(colmap_dir.rglob("cameras.bin")):
        return d.parent
    for d in sorted(colmap_dir.rglob("cameras.txt")):
        return d.parent

    raise FileNotFoundError(
        f"No valid COLMAP model found in {colmap_dir}. "
        "Expected cameras.bin/txt and images.bin/txt files."
    )


def _find_images_dir(colmap_dir: Path) -> Path:
    """Find the images directory associated with a COLMAP workspace.

    Args:
        colmap_dir: COLMAP workspace directory.

    Returns:
        Path to the images directory.

    Raises:
        FileNotFoundError: If images directory is not found.
    """
    candidates = [
        colmap_dir / "images",
        colmap_dir / ".." / "images",
        colmap_dir.parent / "images",
        colmap_dir.parent.parent / "images",
    ]

    for d in candidates:
        if d.exists() and d.is_dir():
            return d.resolve()

    raise FileNotFoundError(
        f"No images directory found near {colmap_dir}. "
        "Expected an 'images' directory in the workspace."
    )


def _count_gaussians_in_ply(ply_path: Path) -> int:
    """Count the number of Gaussians in a splat PLY file by reading its header."""
    try:
        with open(ply_path, "rb") as f:
            for line in f:
                decoded = line.decode("ascii", errors="replace").strip()
                if decoded.startswith("element vertex"):
                    return int(decoded.split()[-1])
                if decoded == "end_header":
                    break
    except Exception:
        pass
    return 0


def _manage_checkpoints(
    checkpoint_dir: Path,
    max_checkpoints: int = 3,
) -> None:
    """Keep only the most recent checkpoints, removing older ones.

    Args:
        checkpoint_dir: Directory containing checkpoint files.
        max_checkpoints: Maximum number of checkpoints to keep.
    """
    if not checkpoint_dir.exists():
        return

    ckpts = sorted(
        checkpoint_dir.glob("*.ply"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    for old_ckpt in ckpts[max_checkpoints:]:
        old_ckpt.unlink()
        logger.info(f"Removed old checkpoint: {old_ckpt.name}")


def run_opensplat(
    colmap_dir: Path,
    output_dir: Path,
    num_iterations: int = 30000,
    num_downscales: int = 2,
    output_ply_name: str = "splat.ply",
    extra_args: list[str] | None = None,
    save_checkpoints: bool = True,
    checkpoint_interval: int = 5000,
    max_checkpoints: int = 3,
) -> OpenSplatResult:
    """Run OpenSplat 3D Gaussian Splatting training.

    Args:
        colmap_dir: Path to COLMAP workspace or sparse model directory.
        output_dir: Directory for output files.
        num_iterations: Number of training iterations.
        num_downscales: Number of image downscale levels.
        output_ply_name: Name of the output PLY file.
        extra_args: Additional command-line arguments for opensplat.
        save_checkpoints: Whether to save intermediate checkpoints.
        checkpoint_interval: Iterations between checkpoints.
        max_checkpoints: Maximum number of checkpoints to keep.

    Returns:
        OpenSplatResult with training statistics.
    """
    colmap_dir = Path(colmap_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = OpenSplatResult()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # --- Check OpenSplat ---
        task = progress.add_task("Checking OpenSplat installation...", total=None)
        try:
            _check_opensplat_available()
            progress.update(task, description="OpenSplat: available")
        except RuntimeError as e:
            console.print(f"[red]{e}")
            result.success = False
            return result
        progress.stop_task(task)

        # --- Locate COLMAP model ---
        task = progress.add_task("Locating COLMAP model...", total=None)
        try:
            model_dir = _find_colmap_model(colmap_dir)
            progress.update(task, description=f"Model: {model_dir}")
        except FileNotFoundError as e:
            console.print(f"[red]{e}")
            result.success = False
            return result
        progress.stop_task(task)

        # --- Checkpoint directory ---
        checkpoint_dir = output_dir / "checkpoints"
        if save_checkpoints:
            checkpoint_dir.mkdir(exist_ok=True)
            result.checkpoint_dir = checkpoint_dir

        # --- Build command ---
        output_ply = output_dir / output_ply_name
        cmd = [
            "opensplat",
            str(colmap_dir),
            "--output", str(output_ply),
            "--num-iters", str(num_iterations),
            "--num-downscales", str(num_downscales),
        ]

        if save_checkpoints:
            cmd.extend([
                "--save-every", str(checkpoint_interval),
            ])

        if extra_args:
            cmd.extend(extra_args)

        # --- Run training ---
        task = progress.add_task(
            f"Training ({num_iterations} iterations)...", total=None
        )
        t0 = time.perf_counter()

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=86400,  # 24 hour max
            )

            result.training_time_s = time.perf_counter() - t0
            result.success = proc.returncode == 0

            if not result.success:
                logger.error(f"OpenSplat failed (rc={proc.returncode})")
                logger.error(f"stderr: {proc.stderr[:3000]}")
                progress.update(
                    task,
                    description=f"Training FAILED ({result.training_time_s:.1f}s)",
                )
                progress.stop_task(task)
                console.print(f"[red]OpenSplat training failed")
                console.print(f"[red]stderr: {proc.stderr[:1000]}")
                return result

            # Parse training output for loss values
            lines = proc.stdout.strip().split("\n")
            for line in reversed(lines):
                if "loss" in line.lower():
                    try:
                        parts = line.split()
                        for j, part in enumerate(parts):
                            if "loss" in part.lower() and j + 1 < len(parts):
                                loss_str = parts[j + 1].strip(",").strip(":")
                                result.final_loss = float(loss_str)
                                break
                    except (ValueError, IndexError):
                        pass
                    if result.final_loss > 0:
                        break

            progress.update(
                task,
                description=f"Training done ({result.training_time_s:.1f}s)",
            )

        except subprocess.TimeoutExpired:
            result.training_time_s = time.perf_counter() - t0
            result.success = False
            logger.error("OpenSplat training timed out")
            progress.update(task, description="Training TIMED OUT")
            progress.stop_task(task)
            return result

        progress.stop_task(task)

        # --- Verify output ---
        task = progress.add_task("Verifying output...", total=None)
        if output_ply.exists():
            result.output_ply = output_ply
            result.num_gaussians = _count_gaussians_in_ply(output_ply)
            file_size_mb = output_ply.stat().st_size / (1024**2)
            progress.update(
                task,
                description=f"Output: {result.num_gaussians:,} gaussians ({file_size_mb:.1f} MB)",
            )
        else:
            console.print("[yellow]Warning: Output PLY not found at expected path")
            # Check if opensplat saved it somewhere else
            for ply in output_dir.rglob("*.ply"):
                if ply.name != "input.ply":
                    result.output_ply = ply
                    result.num_gaussians = _count_gaussians_in_ply(ply)
                    console.print(f"[yellow]Found output at: {ply}")
                    break
        progress.stop_task(task)

        # --- Manage checkpoints ---
        if save_checkpoints and checkpoint_dir.exists():
            _manage_checkpoints(checkpoint_dir, max_checkpoints)

    # --- Measure VRAM ---
    result.peak_vram_gb = _get_peak_vram_gb()
    result.num_iterations = num_iterations

    # Save metadata
    metadata = {
        "success": result.success,
        "training_time_s": round(result.training_time_s, 2),
        "peak_vram_gb": round(result.peak_vram_gb, 2),
        "num_gaussians": result.num_gaussians,
        "num_iterations": num_iterations,
        "final_loss": result.final_loss,
        "output_ply": str(result.output_ply) if result.output_ply else None,
        "output_size_mb": round(
            result.output_ply.stat().st_size / (1024**2), 2
        ) if result.output_ply and result.output_ply.exists() else 0,
        "settings": {
            "colmap_dir": str(colmap_dir),
            "num_iterations": num_iterations,
            "num_downscales": num_downscales,
        },
        "command": " ".join(cmd),
    }
    (output_dir / "opensplat_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)
    )

    # Summary
    console.print(f"\n[bold green]OpenSplat training complete")
    console.print(f"  Gaussians:    {result.num_gaussians:,}")
    console.print(f"  Training:     {result.training_time_s:.1f}s")
    console.print(f"  Final loss:   {result.final_loss:.6f}")
    console.print(f"  Peak VRAM:    {result.peak_vram_gb:.1f} GB")
    if result.output_ply:
        size_mb = result.output_ply.stat().st_size / (1024**2)
        console.print(f"  Output:       {result.output_ply} ({size_mb:.1f} MB)")
    console.print(f"  Workspace:    {output_dir}")

    return result


def resume_training(
    colmap_dir: Path,
    output_dir: Path,
    checkpoint_ply: Path,
    additional_iterations: int = 10000,
    **kwargs,
) -> OpenSplatResult:
    """Resume OpenSplat training from a checkpoint.

    Args:
        colmap_dir: Path to COLMAP workspace.
        output_dir: Output directory.
        checkpoint_ply: Path to the checkpoint PLY to resume from.
        additional_iterations: Number of additional iterations.
        **kwargs: Additional arguments passed to run_opensplat.

    Returns:
        OpenSplatResult from the resumed training.
    """
    if not checkpoint_ply.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_ply}")

    console.print(f"[blue]Resuming from checkpoint: {checkpoint_ply}")

    extra_args = kwargs.pop("extra_args", []) or []
    extra_args.extend(["--input-ply", str(checkpoint_ply)])

    return run_opensplat(
        colmap_dir=colmap_dir,
        output_dir=output_dir,
        num_iterations=additional_iterations,
        extra_args=extra_args,
        **kwargs,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run OpenSplat 3D Gaussian Splatting training."
    )
    parser.add_argument(
        "colmap_dir",
        type=Path,
        help="COLMAP workspace or sparse model directory",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: data/opensplat_output)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=30000,
        help="Number of training iterations (default: 30000)",
    )
    parser.add_argument(
        "--num-downscales",
        type=int,
        default=2,
        help="Number of image downscale levels (default: 2)",
    )
    parser.add_argument(
        "--output-ply",
        type=str,
        default="splat.ply",
        help="Output PLY filename (default: splat.ply)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume training from checkpoint PLY file",
    )
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Disable intermediate checkpoints",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5000,
        help="Iterations between checkpoints (default: 5000)",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = Path(__file__).parent.parent.parent / "data" / "opensplat_output"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.resume:
        resume_training(
            colmap_dir=args.colmap_dir,
            output_dir=args.output_dir,
            checkpoint_ply=args.resume,
            additional_iterations=args.num_iterations,
            num_downscales=args.num_downscales,
            output_ply_name=args.output_ply,
            save_checkpoints=not args.no_checkpoints,
            checkpoint_interval=args.checkpoint_interval,
        )
    else:
        run_opensplat(
            colmap_dir=args.colmap_dir,
            output_dir=args.output_dir,
            num_iterations=args.num_iterations,
            num_downscales=args.num_downscales,
            output_ply_name=args.output_ply,
            save_checkpoints=not args.no_checkpoints,
            checkpoint_interval=args.checkpoint_interval,
        )


if __name__ == "__main__":
    main()
