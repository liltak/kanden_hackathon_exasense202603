"""Foreground extraction for VGGT reconstruction quality improvement.

Provides per-image binary masks that separate foreground (building/structure)
from background (sky, distant vegetation, etc.). These masks are applied
during VGGT post-processing to exclude background points from the point cloud.

Methods available:
- "depth": Statistical depth filtering (fast, no extra model)
- "semantic": DeepLabV3 segmentation to remove sky/vegetation
- "sam": SAM2 automatic mask generation — union of all detected segments
  as foreground, uncovered regions treated as background (sky etc.)
  When SAM3 access is approved, will be upgraded automatically.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from rich.console import Console

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)
console = Console()

# DeepLabV3 COCO class indices to exclude as background
# 0=background is ambiguous so not excluded by default
_SKY_VEGETATION_CLASSES = frozenset({
    # No explicit "sky" class in COCO/VOC — handled by "background" (0)
    # but we include classes that are typically NOT the target structure:
})

# ADE20K class indices for sky/vegetation (used by DeepLabV3 with ADE20K)
# We use VOC classes: 0=background is kept, but we look at specific exclusions
_EXCLUDE_VOC_CLASSES: frozenset[int] = frozenset()  # No VOC classes to exclude by default

# For ADE20K backbone (if used): sky=2, tree=4, grass=9, plant=17, ceiling=5
_EXCLUDE_ADE20K_CLASSES = frozenset({2, 4, 9, 17})


def compute_depth_masks(
    depth_maps: list[np.ndarray],
    confidence_maps: list[np.ndarray],
    depth_sigma: float = 2.0,
    min_confidence: float = 0.0,
) -> list[np.ndarray]:
    """Compute foreground masks from depth statistics.

    For each image, marks pixels as foreground if their depth is within
    `depth_sigma` standard deviations of the median depth (computed over
    high-confidence pixels only).

    Args:
        depth_maps: Per-image depth arrays (H, W).
        confidence_maps: Per-image confidence arrays (H, W).
        depth_sigma: Number of std devs from median to keep (default 2.0).
        min_confidence: Minimum confidence for depth statistics computation.

    Returns:
        List of boolean masks (H, W), True = foreground.
    """
    masks = []

    # Compute global depth statistics from all high-confidence pixels
    all_valid_depths = []
    for depth, conf in zip(depth_maps, confidence_maps):
        valid = conf > min_confidence
        if valid.any():
            all_valid_depths.append(depth[valid])

    if not all_valid_depths:
        return [np.ones(d.shape, dtype=bool) for d in depth_maps]

    global_depths = np.concatenate(all_valid_depths)
    median_depth = float(np.median(global_depths))
    std_depth = float(np.std(global_depths))

    depth_min = median_depth - depth_sigma * std_depth
    depth_max = median_depth + depth_sigma * std_depth

    console.print(
        f"    Depth filter: median={median_depth:.3f}, "
        f"std={std_depth:.3f}, range=[{depth_min:.3f}, {depth_max:.3f}]"
    )

    for depth in depth_maps:
        mask = (depth >= depth_min) & (depth <= depth_max) & (depth > 0)
        masks.append(mask)

    total_pixels = sum(m.size for m in masks)
    fg_pixels = sum(m.sum() for m in masks)
    console.print(f"    Depth mask: {fg_pixels:,}/{total_pixels:,} pixels kept ({100*fg_pixels/total_pixels:.1f}%)")

    return masks


def compute_semantic_masks(
    images: list[Image.Image],
    device: str = "cuda:0",
    exclude_background: bool = True,
) -> list[np.ndarray]:
    """Compute foreground masks using DeepLabV3 semantic segmentation.

    Uses torchvision's DeepLabV3-ResNet50 (BSD license) to identify
    and exclude sky, vegetation, and other background classes.

    The model segments each image into 21 VOC classes. We keep pixels
    that belong to structural classes (building, ground near building)
    and remove sky/distant background.

    Args:
        images: List of PIL images.
        device: Torch device for inference.
        exclude_background: If True, exclude class 0 (background/sky/unknown).

    Returns:
        List of boolean masks (H, W), True = foreground.
    """
    import torch
    import torchvision.transforms as T
    from torchvision.models.segmentation import (
        deeplabv3_resnet50,
        DeepLabV3_ResNet50_Weights,
    )

    console.print("    Loading DeepLabV3-ResNet50 for semantic segmentation...")
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = deeplabv3_resnet50(weights=weights).to(device).eval()

    preprocess = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # VOC classes: 0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat,
    # 5=bottle, 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable,
    # 12=dog, 13=horse, 14=motorbike, 15=person, 16=pottedplant,
    # 17=sheep, 18=sofa, 19=train, 20=tvmonitor
    #
    # For factory/building images, class 0 (background) often corresponds to
    # sky and distant scenery. We exclude it to remove background.
    # All other detected classes are likely part of the scene of interest.
    exclude_classes = set()
    if exclude_background:
        exclude_classes.add(0)  # background/sky/unknown

    masks = []
    with torch.no_grad():
        for i, img in enumerate(images):
            input_tensor = preprocess(img).unsqueeze(0).to(device)
            output = model(input_tensor)["out"]  # (1, 21, H, W)
            pred = output.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)

            # Foreground = NOT in exclude set
            fg_mask = np.ones(pred.shape, dtype=bool)
            for cls_id in exclude_classes:
                fg_mask &= pred != cls_id

            # Resize mask to original image size if needed
            if fg_mask.shape != (img.height, img.width):
                from PIL import Image as PILImage
                mask_img = PILImage.fromarray(fg_mask.astype(np.uint8) * 255)
                mask_img = mask_img.resize((img.width, img.height), PILImage.NEAREST)
                fg_mask = np.array(mask_img) > 127

            masks.append(fg_mask)

    total_pixels = sum(m.size for m in masks)
    fg_pixels = sum(m.sum() for m in masks)
    console.print(
        f"    Semantic mask: {fg_pixels:,}/{total_pixels:,} pixels kept "
        f"({100*fg_pixels/total_pixels:.1f}%)"
    )

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return masks


def compute_sam_masks(
    images: list[Image.Image],
    device: str = "cuda:0",
    sam_checkpoint: str = "models/sam2.1_hiera_tiny.pt",
    sam_config: str = "configs/sam2.1/sam2.1_hiera_t.yaml",
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.7,
    stability_score_thresh: float = 0.9,
    min_mask_region_area: int = 1000,
    sky_area_pct: float = 30.0,
    sky_center_y_max: float = 0.4,
) -> list[np.ndarray]:
    """Compute foreground masks using SAM2 automatic mask generation.

    Strategy: generate all object masks via SAM2, take their union as
    foreground. Uncovered pixels are treated as background (sky, etc.).
    Additionally, large masks centered in the upper part of the image
    are removed as likely sky/canopy.

    When SAM3 becomes available, this will be upgraded to use text-prompted
    concept segmentation (e.g. "building", "factory roof").

    Args:
        images: List of PIL images.
        device: Torch device.
        sam_checkpoint: Path to SAM2 checkpoint.
        sam_config: SAM2 config YAML name.
        points_per_side: Grid density for automatic mask generation.
        pred_iou_thresh: Minimum predicted IoU to keep a mask.
        stability_score_thresh: Minimum stability score.
        min_mask_region_area: Minimum mask area in pixels.
        sky_area_pct: Masks covering more than this % of the image
            and centered in the upper portion are removed as sky.
        sky_center_y_max: Maximum normalized y-center (0=top, 1=bottom)
            for a mask to be considered sky.

    Returns:
        List of boolean masks (H, W), True = foreground.
    """
    import torch
    from pathlib import Path

    # Try SAM3 first, fall back to SAM2
    try:
        from sam3 import build_sam3_image_model
        from huggingface_hub import hf_hub_download

        console.print("    SAM3 available — loading model...")
        # Try official repo first, fall back to mirror
        try:
            ckpt = hf_hub_download(repo_id="facebook/sam3", filename="sam3.pt")
        except Exception:
            ckpt = hf_hub_download(repo_id="1038lab/sam3", filename="sam3.pt")
            console.print("    Using mirror checkpoint (1038lab/sam3)")

        # SAM3 model_builder only supports "cuda" or "cpu", not "cuda:0"
        sam3_device = "cuda" if "cuda" in device else "cpu"
        model = build_sam3_image_model(
            device=sam3_device, eval_mode=True,
            checkpoint_path=ckpt, load_from_HF=False,
        )
        console.print("    SAM3 loaded! Using text-prompted segmentation.")
        masks = _compute_sam3_masks(model, images, device)
        del model
        torch.cuda.empty_cache()
        return masks
    except Exception as e:
        console.print(f"    SAM3 unavailable ({e}), falling back to SAM2...")
        pass

    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    ckpt_path = Path(sam_checkpoint)
    if not ckpt_path.is_absolute():
        # Try relative to project root
        project_root = Path(__file__).resolve().parent.parent.parent
        ckpt_path = project_root / sam_checkpoint

    if not ckpt_path.exists():
        console.print(f"    [yellow]SAM2 checkpoint not found at {ckpt_path}")
        console.print("    [yellow]Downloading SAM2.1 Hiera-Tiny...")
        import urllib.request
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
            str(ckpt_path),
        )
        console.print(f"    Downloaded: {ckpt_path}")

    console.print("    Loading SAM2 Hiera-T for automatic mask generation...")
    sam2_model = build_sam2(sam_config, ckpt_path=str(ckpt_path), device=device)
    mask_gen = SAM2AutomaticMaskGenerator(
        sam2_model,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
    )

    result_masks = []
    for i, img in enumerate(images):
        img_np = np.array(img)
        H, W = img_np.shape[:2]
        total_pixels = H * W

        sam_masks = mask_gen.generate(img_np)
        sam_masks = sorted(sam_masks, key=lambda x: x["area"], reverse=True)

        # Union of all masks = foreground
        fg = np.zeros((H, W), dtype=bool)
        for m in sam_masks:
            fg |= m["segmentation"]

        # Remove sky-like masks: large + centered in upper image
        for m in sam_masks[:5]:
            area_pct = 100 * m["area"] / total_pixels
            bbox = m["bbox"]  # x, y, w, h
            center_y = (bbox[1] + bbox[3] / 2) / H
            if area_pct > sky_area_pct and center_y < sky_center_y_max:
                fg &= ~m["segmentation"]
                console.print(
                    f"    [{i}] Removed sky mask: "
                    f"area={area_pct:.0f}%, center_y={center_y:.2f}"
                )

        result_masks.append(fg)
        fg_pct = 100 * fg.sum() / total_pixels
        console.print(f"    [{i}] SAM2: {len(sam_masks)} masks, {fg_pct:.0f}% foreground")

    total_pixels = sum(m.size for m in result_masks)
    fg_pixels = sum(m.sum() for m in result_masks)
    console.print(
        f"    SAM mask total: {fg_pixels:,}/{total_pixels:,} pixels kept "
        f"({100*fg_pixels/total_pixels:.1f}%)"
    )

    del sam2_model, mask_gen
    torch.cuda.empty_cache()

    return result_masks


def _compute_sam3_masks(
    model,
    images: list[Image.Image],
    device: str = "cuda:0",
    fg_prompts: list[str] | None = None,
    bg_prompts: list[str] | None = None,
) -> list[np.ndarray]:
    """Compute foreground masks using SAM3 text-prompted segmentation.

    Uses text prompts to detect foreground concepts (building, structure, etc.)
    and background concepts (sky) to create per-image masks.

    Args:
        model: Loaded Sam3Image model.
        images: List of PIL images.
        device: Torch device.
        fg_prompts: Text prompts for foreground concepts.
        bg_prompts: Text prompts for background concepts.

    Returns:
        List of boolean masks (H, W), True = foreground.
    """
    import torch
    from sam3.model.sam3_image_processor import Sam3Processor

    if fg_prompts is None:
        fg_prompts = ["building", "structure", "roof", "ground", "wall", "factory"]
    if bg_prompts is None:
        bg_prompts = ["sky"]

    processor = Sam3Processor(model, device=device)
    result_masks = []

    for i, img in enumerate(images):
        img_np = np.array(img)
        H, W = img_np.shape[:2]

        state = processor.set_image(img)

        # Collect foreground masks
        fg_union = np.zeros((H, W), dtype=bool)
        for prompt in fg_prompts:
            output = processor.set_text_prompt(state=state, prompt=prompt)
            masks = output.get("masks")
            if masks is not None and len(masks) > 0:
                if isinstance(masks, torch.Tensor):
                    masks_np = masks.cpu().numpy()
                else:
                    masks_np = np.array(masks)
                if masks_np.ndim == 4:
                    masks_np = masks_np.squeeze(1)
                if masks_np.ndim == 3:
                    combined = masks_np.any(axis=0)
                else:
                    combined = masks_np > 0.5
                if combined.shape != (H, W):
                    from PIL import Image as PILImage
                    m_img = PILImage.fromarray(combined.astype(np.uint8) * 255)
                    m_img = m_img.resize((W, H), PILImage.NEAREST)
                    combined = np.array(m_img) > 127
                fg_union |= combined

        # Collect background masks
        bg_union = np.zeros((H, W), dtype=bool)
        for prompt in bg_prompts:
            output = processor.set_text_prompt(state=state, prompt=prompt)
            masks = output.get("masks")
            if masks is not None and len(masks) > 0:
                if isinstance(masks, torch.Tensor):
                    masks_np = masks.cpu().numpy()
                else:
                    masks_np = np.array(masks)
                if masks_np.ndim == 4:
                    masks_np = masks_np.squeeze(1)
                if masks_np.ndim == 3:
                    combined = masks_np.any(axis=0)
                else:
                    combined = masks_np > 0.5
                if combined.shape != (H, W):
                    from PIL import Image as PILImage
                    m_img = PILImage.fromarray(combined.astype(np.uint8) * 255)
                    m_img = m_img.resize((W, H), PILImage.NEAREST)
                    combined = np.array(m_img) > 127
                bg_union |= combined

        # Final mask: foreground minus background
        fg_mask = fg_union & ~bg_union
        result_masks.append(fg_mask)

        fg_pct = 100 * fg_mask.sum() / (H * W)
        console.print(f"    [{i}] SAM3: {fg_pct:.0f}% foreground")

    total_pixels = sum(m.size for m in result_masks)
    fg_pixels = sum(m.sum() for m in result_masks)
    console.print(
        f"    SAM3 mask total: {fg_pixels:,}/{total_pixels:,} pixels kept "
        f"({100*fg_pixels/total_pixels:.1f}%)"
    )

    return result_masks


def compute_foreground_masks(
    images: list[Image.Image],
    depth_maps: list[np.ndarray] | None = None,
    confidence_maps: list[np.ndarray] | None = None,
    method: str = "depth",
    device: str = "cuda:0",
    depth_sigma: float = 2.0,
) -> list[np.ndarray]:
    """Compute foreground masks using the specified method.

    Args:
        images: List of PIL images.
        depth_maps: Per-image depth arrays (H, W). Required for "depth" method.
        confidence_maps: Per-image confidence arrays (H, W). Used by "depth" method.
        method: "depth", "semantic", "sam", or "sam+depth".
        device: Torch device for segmentation models.
        depth_sigma: Std dev threshold for depth filtering.

    Returns:
        List of boolean masks (H, W), True = foreground.
    """
    console.print(f"  Computing foreground masks (method={method})...")

    if method == "depth":
        if depth_maps is None:
            raise ValueError("depth_maps required for method='depth'")
        if confidence_maps is None:
            confidence_maps = [np.ones_like(d) for d in depth_maps]
        return compute_depth_masks(depth_maps, confidence_maps, depth_sigma=depth_sigma)

    elif method == "semantic":
        return compute_semantic_masks(images, device=device)

    elif method == "sam":
        return compute_sam_masks(images, device=device)

    elif method == "sam+depth":
        if depth_maps is None:
            raise ValueError("depth_maps required for method='sam+depth'")
        if confidence_maps is None:
            confidence_maps = [np.ones_like(d) for d in depth_maps]

        sam_masks = compute_sam_masks(images, device=device)
        depth_masks = compute_depth_masks(
            depth_maps, confidence_maps, depth_sigma=depth_sigma
        )

        combined = _combine_masks(sam_masks, depth_masks)
        return combined

    elif method == "both":
        if depth_maps is None:
            raise ValueError("depth_maps required for method='both'")
        if confidence_maps is None:
            confidence_maps = [np.ones_like(d) for d in depth_maps]

        depth_masks = compute_depth_masks(
            depth_maps, confidence_maps, depth_sigma=depth_sigma
        )
        semantic_masks = compute_semantic_masks(images, device=device)

        combined = _combine_masks(depth_masks, semantic_masks)
        return combined

    else:
        raise ValueError(
            f"Unknown foreground method: {method!r}. "
            "Use 'depth', 'semantic', 'sam', 'sam+depth', or 'both'."
        )


def _combine_masks(
    masks_a: list[np.ndarray],
    masks_b: list[np.ndarray],
) -> list[np.ndarray]:
    """Combine two mask lists with AND, resizing if needed."""
    combined = []
    for ma, mb in zip(masks_a, masks_b):
        if mb.shape != ma.shape:
            from PIL import Image as PILImage
            mb_img = PILImage.fromarray(mb.astype(np.uint8) * 255)
            mb_img = mb_img.resize((ma.shape[1], ma.shape[0]), PILImage.NEAREST)
            mb = np.array(mb_img) > 127
        combined.append(ma & mb)

    total_pixels = sum(m.size for m in combined)
    fg_pixels = sum(m.sum() for m in combined)
    console.print(
        f"    Combined mask: {fg_pixels:,}/{total_pixels:,} pixels kept "
        f"({100*fg_pixels/total_pixels:.1f}%)"
    )
    return combined
