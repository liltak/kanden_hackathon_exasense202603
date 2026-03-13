"""Waypoint-1-Small frame generator.

Wraps the Overworld/Waypoint-1-Small diffusion world model for
generating interactive video frames from seed images.

GPU-only inference; uses TYPE_CHECKING + lazy imports for macOS compatibility.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Set, Tuple

import numpy as np
from rich.console import Console

if TYPE_CHECKING:
    import torch
    from PIL import Image

logger = logging.getLogger(__name__)
console = Console()

MODEL_ID = "Overworld/Waypoint-1-Small"


@dataclass
class ControlInput:
    """Single-frame control input for Waypoint.

    Attributes:
        button: Set of pressed button IDs (0-255). Mapping is model-internal
                and not publicly documented. See docs/waypoint-1-benchmark-report.md.
        mouse: (x, y) velocity. x>0 = look right, x<0 = look left.
        scroll: Scroll wheel direction (-1, 0, 1).
    """

    button: Set[int] = field(default_factory=set)
    mouse: Tuple[float, float] = (0.0, 0.0)
    scroll: int = 0


@dataclass
class GenerationResult:
    """Result from frame generation."""

    frames: list  # list of PIL Images
    num_frames: int = 0
    total_time_s: float = 0.0
    avg_fps: float = 0.0
    peak_vram_gb: float = 0.0

    def __post_init__(self):
        self.num_frames = len(self.frames)


class WaypointGenerator:
    """Waypoint-1-Small frame generator.

    Usage::

        gen = WaypointGenerator(device="cuda:0")
        gen.load_model()

        # Generate frames with orbit controls
        controls = [ControlInput(mouse=(0.15, 0.0)) for _ in range(60)]
        result = gen.generate(
            seed_image=Image.open("factory.jpg"),
            prompt="Factory rooftop with solar panels",
            controls=controls,
        )

        gen.export_video(result.frames, "output.mp4", fps=30)
    """

    def __init__(self, device: str = "cuda:0", compile: bool = True):
        self.device = device
        self.compile = compile
        self._pipe = None
        self._loaded = False

    def load_model(self) -> float:
        """Load Waypoint-1-Small model. Returns load time in seconds."""
        import torch
        from diffusers.modular_pipelines import ModularPipeline

        t0 = time.perf_counter()

        self._pipe = ModularPipeline.from_pretrained(MODEL_ID, trust_remote_code=True)
        self._pipe.load_components(
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self._pipe.transformer.apply_inference_patches()

        if self.compile:
            logger.info("Applying torch.compile (first run may take minutes)...")
            self._pipe.transformer.compile(
                fullgraph=True, mode="max-autotune", dynamic=False
            )
            self._pipe.vae.bake_weight_norm()
            self._pipe.vae.compile(fullgraph=True, mode="max-autotune")

        load_time = time.perf_counter() - t0
        self._loaded = True
        logger.info(f"Model loaded in {load_time:.1f}s on {self.device}")
        return load_time

    def generate(
        self,
        seed_image: Image,
        prompt: str,
        controls: list[ControlInput],
    ) -> GenerationResult:
        """Generate frames from a seed image with control inputs.

        Args:
            seed_image: PIL Image to start generation from.
            prompt: Text prompt to guide generation.
            controls: List of ControlInput, one per frame to generate.

        Returns:
            GenerationResult with generated frames.
        """
        import torch

        if not self._loaded:
            raise RuntimeError("Call load_model() before generate()")

        torch.cuda.reset_peak_memory_stats()
        frames = []
        t_start = time.perf_counter()

        # First frame (with seed image)
        ctrl = controls[0] if controls else ControlInput()
        state = self._pipe(
            prompt=prompt,
            image=seed_image,
            button=ctrl.button,
            mouse=ctrl.mouse,
            scroll=ctrl.scroll,
        )
        frames.append(state.values["images"])

        # Subsequent frames
        state.values["image"] = None
        for i in range(1, len(controls)):
            ctrl = controls[i]
            state = self._pipe(
                state,
                prompt=prompt,
                button=ctrl.button,
                mouse=ctrl.mouse,
                scroll=ctrl.scroll,
                output_type="pil",
            )
            frames.append(state.values["images"])

        total_time = time.perf_counter() - t_start
        peak_vram = torch.cuda.max_memory_allocated() / 1e9

        return GenerationResult(
            frames=frames,
            total_time_s=round(total_time, 1),
            avg_fps=round(len(frames) / total_time, 1) if total_time > 0 else 0,
            peak_vram_gb=round(peak_vram, 2),
        )

    @staticmethod
    def export_video(frames: list, output_path: str, fps: int = 30) -> Path:
        """Export frames to MP4 video."""
        from diffusers.utils import export_to_video

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(frames, str(path), fps=fps)
        return path

    @staticmethod
    def orbit_controls(
        num_frames: int,
        turn_speed: float = 0.15,
        walk_button: int | None = None,
        wobble: float = 0.02,
    ) -> list[ControlInput]:
        """Generate orbit (circular) control inputs.

        Args:
            num_frames: Number of frames to generate controls for.
            turn_speed: Mouse x velocity for turning (0.05=slow, 0.3=fast).
            walk_button: Optional button ID to press for forward movement.
            wobble: Vertical camera wobble amplitude.
        """
        import math

        controls = []
        for i in range(num_frames):
            mouse_x = turn_speed
            mouse_y = wobble * math.sin(i * 0.3)
            buttons = {walk_button} if walk_button is not None else set()
            controls.append(ControlInput(button=buttons, mouse=(mouse_x, mouse_y)))
        return controls

    @staticmethod
    def static_controls(num_frames: int) -> list[ControlInput]:
        """Generate static (no movement) control inputs."""
        return [ControlInput() for _ in range(num_frames)]
