"""VGGT (Visual Geometry Grounded Transformer) inference runner.

Loads the facebookresearch/vggt model from HuggingFace and performs
3D reconstruction from a directory of images, producing point clouds,
camera poses, and depth maps.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from typing import TYPE_CHECKING
from rich.console import Console

if TYPE_CHECKING:
    import torch
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

logger = logging.getLogger(__name__)
console = Console()

VGGT_MODEL_ID = "facebook/VGGT-1B-Commercial"


@dataclass
class VGGTResult:
    """Structured result from VGGT inference."""

    point_cloud: np.ndarray  # (N, 3) world-coordinate points
    point_colors: np.ndarray  # (N, 3) RGB in [0, 1]
    camera_poses: list[dict]  # per-image extrinsics + intrinsics
    depth_maps: list[np.ndarray]  # per-image (H, W) depth
    confidence_maps: list[np.ndarray]  # per-image (H, W) confidence
    image_names: list[str]
    num_points: int = 0
    inference_time_s: float = 0.0
    peak_vram_gb: float = 0.0

    def __post_init__(self):
        self.num_points = len(self.point_cloud)


@dataclass
class TimingInfo:
    """Timing breakdown for pipeline stages."""

    model_load_s: float = 0.0
    image_load_s: float = 0.0
    inference_s: float = 0.0
    postprocess_s: float = 0.0
    save_s: float = 0.0
    total_s: float = 0.0
    peak_vram_gb: float = 0.0


def _get_peak_vram_gb() -> float:
    """Return peak VRAM usage in GB if CUDA is available."""
    import torch as _torch
    if _torch.cuda.is_available():
        return _torch.cuda.max_memory_allocated() / (1024**3)
    return 0.0


def _reset_vram_tracking():
    """Reset CUDA peak memory tracking."""
    import torch as _torch
    if _torch.cuda.is_available():
        _torch.cuda.reset_peak_memory_stats()


def _load_images(image_dir: Path, max_images: int | None = None) -> tuple[list[Image.Image], list[str]]:
    """Load images from a directory, sorted by name.

    Args:
        image_dir: Directory containing images.
        max_images: Maximum number of images to load.

    Returns:
        Tuple of (PIL images, filenames).
    """
    from PIL import Image as _Image

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in extensions and not p.name.startswith(".")
    )

    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    if max_images is not None:
        image_paths = image_paths[:max_images]

    images = []
    names = []
    for p in image_paths:
        img = _Image.open(p).convert("RGB")
        images.append(img)
        names.append(p.name)

    return images, names


def _load_model(device: torch.device) -> tuple:
    """Load VGGT model and processor from HuggingFace.

    Returns:
        Tuple of (model, processor).
    """
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images

    model = VGGT.from_pretrained(VGGT_MODEL_ID)
    model = model.to(device)
    model.eval()

    return model, load_and_preprocess_images


def _camera_pose_to_dict(
    extrinsic: np.ndarray,
    intrinsic: np.ndarray | None,
    image_name: str,
    image_size: tuple[int, int],
) -> dict:
    """Convert camera parameters to a serializable dictionary."""
    pose = {
        "image_name": image_name,
        "width": image_size[0],
        "height": image_size[1],
        "extrinsic": extrinsic.tolist(),
    }
    if intrinsic is not None:
        pose["intrinsic"] = intrinsic.tolist()
    return pose


def _save_point_cloud_ply(
    points: np.ndarray,
    colors: np.ndarray,
    output_path: Path,
) -> None:
    """Save point cloud as PLY file.

    Args:
        points: (N, 3) float array of xyz coordinates.
        colors: (N, 3) float array of RGB in [0, 1].
        output_path: Path for the output PLY file.
    """
    n = len(points)
    colors_uint8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)

    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    with open(output_path, "w") as f:
        f.write(header)
        for i in range(n):
            x, y, z = points[i]
            r, g, b = colors_uint8[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


def run_vggt(
    image_dir: Path,
    output_dir: Path,
    device: str = "cuda",
    max_images: int | None = None,
    confidence_threshold: float = 0.5,
    dtype: str = "float16",
) -> VGGTResult:
    """Run VGGT inference on a directory of images.

    Args:
        image_dir: Directory containing input images.
        output_dir: Directory for output files (PLY, JSON, depth maps).
        device: Torch device string.
        max_images: Maximum number of images to process.
        confidence_threshold: Minimum confidence for point cloud filtering.
        dtype: Model dtype ("float16" or "float32").

    Returns:
        VGGTResult with point cloud, camera poses, and depth maps.
    """
    import torch
    from PIL import Image

    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if dtype == "float16" else torch.float32

    if torch_device.type == "cpu":
        console.print("[yellow]CUDA not available, running on CPU (will be slow)")

    timing = TimingInfo()
    _reset_vram_tracking()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # --- Load model ---
        task = progress.add_task("Loading VGGT model...", total=None)
        t0 = time.perf_counter()
        model, load_and_preprocess_images = _load_model(torch_device)
        timing.model_load_s = time.perf_counter() - t0
        progress.update(task, description=f"Model loaded ({timing.model_load_s:.1f}s)")
        progress.stop_task(task)

        # --- Load images ---
        task = progress.add_task("Loading images...", total=None)
        t0 = time.perf_counter()
        images, image_names = _load_images(image_dir, max_images)
        timing.image_load_s = time.perf_counter() - t0
        progress.update(
            task,
            description=f"Loaded {len(images)} images ({timing.image_load_s:.1f}s)",
        )
        progress.stop_task(task)

        console.print(f"  Images: {len(images)}, size: {images[0].size}")

        # --- Preprocess and run inference ---
        task = progress.add_task("Running VGGT inference...", total=None)
        t0 = time.perf_counter()

        image_paths = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
            and not p.name.startswith(".")
        )
        if max_images is not None:
            image_paths = image_paths[:max_images]

        images_tensor = load_and_preprocess_images([str(p) for p in image_paths])
        images_tensor = images_tensor.to(torch_device)

        with torch.no_grad(), torch.autocast(device_type=torch_device.type, dtype=torch_dtype):
            predictions = model(images_tensor)

        timing.inference_s = time.perf_counter() - t0
        timing.peak_vram_gb = _get_peak_vram_gb()
        progress.update(
            task,
            description=f"Inference done ({timing.inference_s:.1f}s, VRAM: {timing.peak_vram_gb:.1f}GB)",
        )
        progress.stop_task(task)

        # --- Post-processing ---
        task = progress.add_task("Post-processing results...", total=None)
        t0 = time.perf_counter()

        # Extract point cloud from predictions
        # VGGT returns per-frame 3D points, we aggregate them
        pred_world_points = predictions["world_points"].cpu().numpy()  # (1, N_frames, H, W, 3)
        pred_world_points = pred_world_points[0]  # (N_frames, H, W, 3)

        # Confidence maps
        if "world_points_conf" in predictions:
            pred_conf = predictions["world_points_conf"].cpu().numpy()[0]  # (N_frames, H, W)
        else:
            pred_conf = np.ones(pred_world_points.shape[:3], dtype=np.float32)

        # Depth maps from predictions
        if "depth" in predictions:
            raw_depths = predictions["depth"].cpu().numpy()[0]  # (N_frames, H, W)
        else:
            raw_depths = np.linalg.norm(pred_world_points, axis=-1)

        # Camera extrinsics
        if "extrinsic" in predictions:
            pred_extrinsics = predictions["extrinsic"].cpu().numpy()[0]  # (N_frames, 3, 4) or (N_frames, 4, 4)
        else:
            pred_extrinsics = [np.eye(4) for _ in range(len(image_names))]

        # Camera intrinsics
        if "intrinsic" in predictions:
            pred_intrinsics = predictions["intrinsic"].cpu().numpy()[0]
        else:
            pred_intrinsics = [None] * len(image_names)

        # Collect per-image results
        all_points = []
        all_colors = []
        depth_maps = []
        confidence_maps = []
        camera_poses = []

        for i, name in enumerate(image_names):
            frame_points = pred_world_points[i]  # (H, W, 3)
            frame_conf = pred_conf[i]  # (H, W)
            frame_depth = raw_depths[i]  # (H, W)

            mask = frame_conf > confidence_threshold
            valid_points = frame_points[mask]  # (M, 3)

            # Get colors from the original image, resized to match prediction resolution
            img = images[i]
            img_resized = img.resize((frame_points.shape[1], frame_points.shape[0]), Image.BILINEAR)
            img_array = np.array(img_resized, dtype=np.float32) / 255.0  # (H, W, 3)
            valid_colors = img_array[mask]  # (M, 3)

            all_points.append(valid_points)
            all_colors.append(valid_colors)
            depth_maps.append(frame_depth)
            confidence_maps.append(frame_conf)

            extrinsic = pred_extrinsics[i] if isinstance(pred_extrinsics, np.ndarray) else pred_extrinsics[i]
            intrinsic = pred_intrinsics[i] if isinstance(pred_intrinsics, np.ndarray) else pred_intrinsics[i]

            camera_poses.append(
                _camera_pose_to_dict(extrinsic, intrinsic, name, img.size)
            )

        point_cloud = np.concatenate(all_points, axis=0)
        point_colors = np.concatenate(all_colors, axis=0)

        timing.postprocess_s = time.perf_counter() - t0
        progress.update(
            task,
            description=f"Post-processed ({timing.postprocess_s:.1f}s, {len(point_cloud):,} points)",
        )
        progress.stop_task(task)

        # --- Save results ---
        task = progress.add_task("Saving results...", total=None)
        t0 = time.perf_counter()

        ply_path = output_dir / "point_cloud.ply"
        _save_point_cloud_ply(point_cloud, point_colors, ply_path)

        poses_path = output_dir / "camera_poses.json"
        poses_data = {
            "num_cameras": len(camera_poses),
            "cameras": camera_poses,
        }
        poses_path.write_text(json.dumps(poses_data, indent=2, ensure_ascii=False))

        depth_dir = output_dir / "depth_maps"
        depth_dir.mkdir(exist_ok=True)
        for i, (name, dm) in enumerate(zip(image_names, depth_maps)):
            np.save(depth_dir / f"{Path(name).stem}_depth.npy", dm)

        conf_dir = output_dir / "confidence_maps"
        conf_dir.mkdir(exist_ok=True)
        for i, (name, cm) in enumerate(zip(image_names, confidence_maps)):
            np.save(conf_dir / f"{Path(name).stem}_conf.npy", cm)

        timing.save_s = time.perf_counter() - t0
        timing.total_s = timing.model_load_s + timing.image_load_s + timing.inference_s + timing.postprocess_s + timing.save_s
        progress.update(task, description=f"Results saved ({timing.save_s:.1f}s)")
        progress.stop_task(task)

    # Save timing metadata
    metadata = {
        "timing": {
            "model_load_s": round(timing.model_load_s, 2),
            "image_load_s": round(timing.image_load_s, 2),
            "inference_s": round(timing.inference_s, 2),
            "postprocess_s": round(timing.postprocess_s, 2),
            "save_s": round(timing.save_s, 2),
            "total_s": round(timing.total_s, 2),
        },
        "peak_vram_gb": round(timing.peak_vram_gb, 2),
        "num_images": len(image_names),
        "num_points": len(point_cloud),
        "confidence_threshold": confidence_threshold,
        "device": str(torch_device),
        "dtype": dtype,
        "output_files": {
            "point_cloud": str(ply_path),
            "camera_poses": str(poses_path),
            "depth_maps": str(depth_dir),
        },
    }
    (output_dir / "vggt_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)
    )

    # Summary
    console.print(f"\n[bold green]VGGT reconstruction complete")
    console.print(f"  Points:     {len(point_cloud):,}")
    console.print(f"  Cameras:    {len(camera_poses)}")
    console.print(f"  Time:       {timing.total_s:.1f}s")
    console.print(f"  Peak VRAM:  {timing.peak_vram_gb:.1f} GB")
    console.print(f"  Output:     {output_dir}")

    result = VGGTResult(
        point_cloud=point_cloud,
        point_colors=point_colors,
        camera_poses=camera_poses,
        depth_maps=depth_maps,
        confidence_maps=confidence_maps,
        image_names=image_names,
        inference_time_s=timing.inference_s,
        peak_vram_gb=timing.peak_vram_gb,
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run VGGT 3D reconstruction on a directory of images."
    )
    parser.add_argument(
        "image_dir",
        type=Path,
        help="Directory containing input images",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: data/vggt_output)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device (default: cuda)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum confidence for point cloud filtering (default: 0.5)",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Model dtype (default: float16)",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = Path(__file__).parent.parent.parent / "data" / "vggt_output"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    run_vggt(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        device=args.device,
        max_images=args.max_images,
        confidence_threshold=args.confidence_threshold,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
