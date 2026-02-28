"""COLMAP Structure-from-Motion pipeline runner.

Executes the full COLMAP SfM pipeline via subprocess:
feature extraction -> feature matching -> bundle adjustment (mapper).
Produces a sparse reconstruction with camera poses, images, and 3D points.
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

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class StepTiming:
    """Timing for a single pipeline step."""

    name: str
    duration_s: float
    success: bool
    command: str = ""
    returncode: int = 0


@dataclass
class ColmapResult:
    """Structured result from COLMAP SfM pipeline."""

    workspace_dir: Path
    sparse_dir: Path
    num_cameras: int = 0
    num_images: int = 0
    num_points3d: int = 0
    step_timings: list[StepTiming] = field(default_factory=list)
    total_time_s: float = 0.0
    success: bool = False


def _check_colmap_available() -> str:
    """Check that COLMAP is on PATH and return its version."""
    result = subprocess.run(
        ["colmap", "help"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "COLMAP not found on PATH. Install COLMAP and ensure it is accessible.\n"
            "See: https://colmap.github.io/install.html"
        )
    # Try to get version
    version_result = subprocess.run(
        ["colmap", "version"],
        capture_output=True,
        text=True,
    )
    version = version_result.stdout.strip() or "unknown"
    return version


def _run_colmap_command(
    args: list[str],
    step_name: str,
    timeout: int = 3600,
) -> StepTiming:
    """Run a COLMAP command and return timing info.

    Args:
        args: Full command-line arguments (including 'colmap').
        step_name: Human-readable step name.
        timeout: Maximum execution time in seconds.

    Returns:
        StepTiming with duration and success status.

    Raises:
        subprocess.TimeoutExpired: If the command exceeds the timeout.
        RuntimeError: If the command fails.
    """
    cmd_str = " ".join(args)
    logger.info(f"Running: {cmd_str}")

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        duration = time.perf_counter() - t0
        success = result.returncode == 0

        if not success:
            logger.error(f"COLMAP {step_name} failed (rc={result.returncode})")
            logger.error(f"stderr: {result.stderr[:2000]}")

        return StepTiming(
            name=step_name,
            duration_s=duration,
            success=success,
            command=cmd_str,
            returncode=result.returncode,
        )

    except subprocess.TimeoutExpired:
        duration = time.perf_counter() - t0
        logger.error(f"COLMAP {step_name} timed out after {timeout}s")
        return StepTiming(
            name=step_name,
            duration_s=duration,
            success=False,
            command=cmd_str,
            returncode=-1,
        )


def _setup_workspace(workspace_dir: Path, image_dir: Path) -> Path:
    """Set up COLMAP workspace directory structure.

    Args:
        workspace_dir: Root workspace directory.
        image_dir: Source image directory.

    Returns:
        Path to the database file.
    """
    workspace_dir.mkdir(parents=True, exist_ok=True)
    (workspace_dir / "sparse").mkdir(exist_ok=True)
    (workspace_dir / "dense").mkdir(exist_ok=True)

    # Symlink or copy images into workspace
    ws_images = workspace_dir / "images"
    if ws_images.exists():
        if ws_images.is_symlink():
            ws_images.unlink()
        else:
            shutil.rmtree(ws_images)

    ws_images.symlink_to(image_dir.resolve())

    db_path = workspace_dir / "database.db"
    return db_path


def _count_sparse_model(sparse_dir: Path) -> tuple[int, int, int]:
    """Count cameras, images, and points in a COLMAP sparse model.

    Handles both binary (.bin) and text (.txt) formats.

    Returns:
        Tuple of (num_cameras, num_images, num_points3d).
    """
    num_cameras = 0
    num_images = 0
    num_points = 0

    # Try to find the model directory (COLMAP creates 0/, 1/, etc.)
    model_dirs = sorted(sparse_dir.iterdir()) if sparse_dir.exists() else []
    model_dir = None
    for d in model_dirs:
        if d.is_dir():
            model_dir = d
            break

    if model_dir is None:
        return 0, 0, 0

    # Count from binary files
    cameras_bin = model_dir / "cameras.bin"
    images_bin = model_dir / "images.bin"
    points_bin = model_dir / "points3D.bin"
    cameras_txt = model_dir / "cameras.txt"
    images_txt = model_dir / "images.txt"
    points_txt = model_dir / "points3D.txt"

    if cameras_txt.exists():
        with open(cameras_txt) as f:
            for line in f:
                if not line.startswith("#") and line.strip():
                    num_cameras += 1
    elif cameras_bin.exists():
        num_cameras = -1  # binary, count unknown without parsing

    if images_txt.exists():
        with open(images_txt) as f:
            for line in f:
                if not line.startswith("#") and line.strip():
                    num_images += 1
        num_images //= 2  # images.txt has 2 lines per image
    elif images_bin.exists():
        num_images = -1

    if points_txt.exists():
        with open(points_txt) as f:
            for line in f:
                if not line.startswith("#") and line.strip():
                    num_points += 1
    elif points_bin.exists():
        num_points = -1

    return num_cameras, num_images, num_points


def run_colmap(
    image_dir: Path,
    workspace_dir: Path,
    camera_model: str = "OPENCV",
    use_gpu: bool = True,
    matcher_type: str = "exhaustive",
    max_image_size: int = 3200,
    num_threads: int = -1,
) -> ColmapResult:
    """Run the full COLMAP SfM pipeline.

    Args:
        image_dir: Directory containing input images.
        workspace_dir: COLMAP workspace directory.
        camera_model: Camera model type (SIMPLE_PINHOLE, PINHOLE, OPENCV, etc.).
        use_gpu: Whether to use GPU for feature extraction/matching.
        matcher_type: Matching strategy ('exhaustive', 'sequential', 'vocab_tree').
        max_image_size: Maximum image size in pixels (longer edge).
        num_threads: Number of CPU threads (-1 for auto).

    Returns:
        ColmapResult with reconstruction statistics and timing.
    """
    image_dir = Path(image_dir)
    workspace_dir = Path(workspace_dir)
    total_t0 = time.perf_counter()

    result = ColmapResult(
        workspace_dir=workspace_dir,
        sparse_dir=workspace_dir / "sparse",
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # --- Check COLMAP ---
        task = progress.add_task("Checking COLMAP installation...", total=None)
        try:
            version = _check_colmap_available()
            progress.update(task, description=f"COLMAP version: {version}")
        except RuntimeError as e:
            console.print(f"[red]{e}")
            result.success = False
            return result
        progress.stop_task(task)

        # --- Setup workspace ---
        task = progress.add_task("Setting up workspace...", total=None)
        db_path = _setup_workspace(workspace_dir, image_dir)
        progress.update(task, description=f"Workspace: {workspace_dir}")
        progress.stop_task(task)

        gpu_flag = "1" if use_gpu else "0"

        # --- Step 1: Feature Extraction ---
        task = progress.add_task("Feature extraction...", total=None)
        step = _run_colmap_command(
            [
                "colmap", "feature_extractor",
                "--database_path", str(db_path),
                "--image_path", str(workspace_dir / "images"),
                "--ImageReader.camera_model", camera_model,
                "--ImageReader.single_camera", "1",
                "--SiftExtraction.use_gpu", gpu_flag,
                "--SiftExtraction.max_image_size", str(max_image_size),
                *(["--SiftExtraction.num_threads", str(num_threads)] if num_threads > 0 else []),
            ],
            step_name="feature_extraction",
        )
        result.step_timings.append(step)
        progress.update(
            task,
            description=f"Feature extraction: {'OK' if step.success else 'FAILED'} ({step.duration_s:.1f}s)",
        )
        progress.stop_task(task)

        if not step.success:
            console.print("[red]Feature extraction failed, aborting pipeline")
            result.success = False
            result.total_time_s = time.perf_counter() - total_t0
            return result

        # --- Step 2: Feature Matching ---
        task = progress.add_task(f"Feature matching ({matcher_type})...", total=None)
        matcher_cmd = f"{matcher_type}_matcher"
        step = _run_colmap_command(
            [
                "colmap", matcher_cmd,
                "--database_path", str(db_path),
                "--SiftMatching.use_gpu", gpu_flag,
                *(["--SiftMatching.num_threads", str(num_threads)] if num_threads > 0 else []),
            ],
            step_name="feature_matching",
        )
        result.step_timings.append(step)
        progress.update(
            task,
            description=f"Feature matching: {'OK' if step.success else 'FAILED'} ({step.duration_s:.1f}s)",
        )
        progress.stop_task(task)

        if not step.success:
            console.print("[red]Feature matching failed, aborting pipeline")
            result.success = False
            result.total_time_s = time.perf_counter() - total_t0
            return result

        # --- Step 3: Bundle Adjustment (Mapper) ---
        task = progress.add_task("Bundle adjustment (mapper)...", total=None)
        sparse_dir = workspace_dir / "sparse"
        step = _run_colmap_command(
            [
                "colmap", "mapper",
                "--database_path", str(db_path),
                "--image_path", str(workspace_dir / "images"),
                "--output_path", str(sparse_dir),
            ],
            step_name="mapper",
            timeout=7200,
        )
        result.step_timings.append(step)
        progress.update(
            task,
            description=f"Mapper: {'OK' if step.success else 'FAILED'} ({step.duration_s:.1f}s)",
        )
        progress.stop_task(task)

        if not step.success:
            console.print("[red]Mapper failed")
            result.success = False
            result.total_time_s = time.perf_counter() - total_t0
            return result

        # --- Step 4: Convert model to TXT for easier parsing ---
        task = progress.add_task("Converting model to text format...", total=None)
        model_dirs = sorted(d for d in sparse_dir.iterdir() if d.is_dir())
        if model_dirs:
            model_dir = model_dirs[0]
            txt_dir = sparse_dir / f"{model_dir.name}_txt"
            txt_dir.mkdir(exist_ok=True)
            step = _run_colmap_command(
                [
                    "colmap", "model_converter",
                    "--input_path", str(model_dir),
                    "--output_path", str(txt_dir),
                    "--output_type", "TXT",
                ],
                step_name="model_converter",
            )
            result.step_timings.append(step)
            progress.update(
                task,
                description=f"Model converted: {'OK' if step.success else 'FAILED'} ({step.duration_s:.1f}s)",
            )
        progress.stop_task(task)

    # --- Count results ---
    num_cameras, num_images, num_points = _count_sparse_model(sparse_dir)
    result.num_cameras = num_cameras
    result.num_images = num_images
    result.num_points3d = num_points
    result.success = True
    result.total_time_s = time.perf_counter() - total_t0

    # Save metadata
    metadata = {
        "success": result.success,
        "total_time_s": round(result.total_time_s, 2),
        "num_cameras": result.num_cameras,
        "num_images": result.num_images,
        "num_points3d": result.num_points3d,
        "settings": {
            "camera_model": camera_model,
            "use_gpu": use_gpu,
            "matcher_type": matcher_type,
            "max_image_size": max_image_size,
        },
        "steps": [
            {
                "name": s.name,
                "duration_s": round(s.duration_s, 2),
                "success": s.success,
            }
            for s in result.step_timings
        ],
    }
    (workspace_dir / "colmap_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)
    )

    # Summary
    console.print(f"\n[bold green]COLMAP SfM complete")
    console.print(f"  Cameras:    {num_cameras}")
    console.print(f"  Images:     {num_images}")
    console.print(f"  3D points:  {num_points}")
    console.print(f"  Total time: {result.total_time_s:.1f}s")
    for s in result.step_timings:
        status = "[green]OK[/green]" if s.success else "[red]FAIL[/red]"
        console.print(f"    {s.name}: {s.duration_s:.1f}s {status}")
    console.print(f"  Output:     {workspace_dir}")

    return result


def clean_workspace(workspace_dir: Path, keep_sparse: bool = True) -> None:
    """Clean a COLMAP workspace.

    Args:
        workspace_dir: Workspace directory to clean.
        keep_sparse: If True, keep the sparse reconstruction.
    """
    workspace_dir = Path(workspace_dir)
    if not workspace_dir.exists():
        return

    db_path = workspace_dir / "database.db"
    if db_path.exists():
        db_path.unlink()
        console.print(f"  Removed {db_path}")

    dense_dir = workspace_dir / "dense"
    if dense_dir.exists():
        shutil.rmtree(dense_dir)
        console.print(f"  Removed {dense_dir}")

    if not keep_sparse:
        sparse_dir = workspace_dir / "sparse"
        if sparse_dir.exists():
            shutil.rmtree(sparse_dir)
            console.print(f"  Removed {sparse_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run COLMAP SfM pipeline on a directory of images."
    )
    parser.add_argument(
        "image_dir",
        type=Path,
        help="Directory containing input images",
    )
    parser.add_argument(
        "-o", "--workspace-dir",
        type=Path,
        default=None,
        help="COLMAP workspace directory (default: data/colmap_workspace)",
    )
    parser.add_argument(
        "--camera-model",
        type=str,
        default="OPENCV",
        choices=["SIMPLE_PINHOLE", "PINHOLE", "OPENCV", "SIMPLE_RADIAL", "RADIAL"],
        help="Camera model (default: OPENCV)",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU for feature extraction/matching",
    )
    parser.add_argument(
        "--matcher",
        type=str,
        default="exhaustive",
        choices=["exhaustive", "sequential", "vocab_tree"],
        help="Feature matching strategy (default: exhaustive)",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=3200,
        help="Maximum image size in pixels (default: 3200)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean workspace before running",
    )
    args = parser.parse_args()

    if args.workspace_dir is None:
        args.workspace_dir = Path(__file__).parent.parent.parent / "data" / "colmap_workspace"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.clean:
        console.print("[yellow]Cleaning workspace...")
        clean_workspace(args.workspace_dir, keep_sparse=False)

    run_colmap(
        image_dir=args.image_dir,
        workspace_dir=args.workspace_dir,
        camera_model=args.camera_model,
        use_gpu=not args.no_gpu,
        matcher_type=args.matcher,
        max_image_size=args.max_image_size,
    )


if __name__ == "__main__":
    main()
