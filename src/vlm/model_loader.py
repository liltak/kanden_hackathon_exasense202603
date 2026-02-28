"""Qwen3.5-VL model loader for solar panel analysis.

Loads Qwen/Qwen3.5-VL-7B-Instruct from HuggingFace with optional 4-bit quantization.
Targets H100 GPU server for inference and fine-tuning.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)
console = Console()

MODEL_ID = "Qwen/Qwen3.5-VL-7B-Instruct"


@dataclass
class ModelConfig:
    """Configuration for Qwen3.5-VL model loading."""

    model_id: str = MODEL_ID
    quantize_4bit: bool = False
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    max_new_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.05
    trust_remote_code: bool = True
    attn_implementation: str = "flash_attention_2"
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 1280 * 28 * 28
    extra_kwargs: dict = field(default_factory=dict)


def get_torch_dtype(dtype_str: str) -> "torch.dtype":
    """Convert string dtype to torch.dtype."""
    import torch

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Use one of {list(mapping.keys())}")
    return mapping[dtype_str]


def get_vram_usage_mb() -> float:
    """Get current GPU VRAM usage in MB."""
    import torch

    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024 / 1024


def get_vram_summary() -> dict[str, float]:
    """Get detailed VRAM usage summary."""
    import torch

    if not torch.cuda.is_available():
        return {"allocated_mb": 0, "reserved_mb": 0, "total_mb": 0}
    torch.cuda.synchronize()
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
        "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
        "total_mb": torch.cuda.get_device_properties(0).total_mem / 1024 / 1024,
    }


def _build_quantization_config():
    """Build BitsAndBytes 4-bit quantization config."""
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=get_torch_dtype("bfloat16"),
        bnb_4bit_use_double_quant=True,
    )


def load_model(
    config: ModelConfig | None = None,
) -> tuple:
    """Load Qwen3.5-VL model and processor.

    Args:
        config: Model configuration. Uses defaults if None.

    Returns:
        Tuple of (model, processor).
    """
    if config is None:
        config = ModelConfig()

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    console.print(f"[bold blue]Loading model: {config.model_id}")
    console.print(f"  Quantize 4-bit: {config.quantize_4bit}")
    console.print(f"  Dtype: {config.torch_dtype}")
    console.print(f"  Attention: {config.attn_implementation}")

    vram_before = get_vram_usage_mb()
    t0 = time.perf_counter()

    model_kwargs: dict = {
        "torch_dtype": get_torch_dtype(config.torch_dtype),
        "device_map": config.device_map,
        "trust_remote_code": config.trust_remote_code,
        "attn_implementation": config.attn_implementation,
        **config.extra_kwargs,
    }

    if config.quantize_4bit:
        model_kwargs["quantization_config"] = _build_quantization_config()
        console.print("  [yellow]4-bit quantization enabled (BitsAndBytes NF4)")

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model_id,
            **model_kwargs,
        )
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        console.print(f"[red]Model load failed: {e}")
        raise

    processor = AutoProcessor.from_pretrained(
        config.model_id,
        trust_remote_code=config.trust_remote_code,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
    )

    elapsed = time.perf_counter() - t0
    vram_after = get_vram_usage_mb()
    vram_used = vram_after - vram_before

    console.print(f"[green]Model loaded in {elapsed:.1f}s")
    console.print(f"  VRAM used: {vram_used:.0f} MB ({vram_used / 1024:.1f} GB)")

    try:
        import torch as _torch
        _cuda = _torch.cuda.is_available()
    except ImportError:
        _cuda = False
    if _cuda:
        summary = get_vram_summary()
        console.print(
            f"  VRAM total: {summary['total_mb']:.0f} MB, "
            f"allocated: {summary['allocated_mb']:.0f} MB, "
            f"reserved: {summary['reserved_mb']:.0f} MB"
        )

    logger.info(
        "Model loaded: %s (%.1fs, %.0f MB VRAM)",
        config.model_id,
        elapsed,
        vram_used,
    )

    return model, processor


def load_model_simple(
    quantize: bool = False,
    model_id: str = MODEL_ID,
) -> tuple:
    """Simplified model loader with minimal configuration.

    Args:
        quantize: Enable 4-bit quantization.
        model_id: HuggingFace model ID.

    Returns:
        Tuple of (model, processor).
    """
    config = ModelConfig(model_id=model_id, quantize_4bit=quantize)
    return load_model(config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load Qwen3.5-VL model")
    parser.add_argument("--model-id", default=MODEL_ID, help="HuggingFace model ID")
    parser.add_argument("--quantize", action="store_true", help="Enable 4-bit quantization")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()

    cfg = ModelConfig(
        model_id=args.model_id,
        quantize_4bit=args.quantize,
        torch_dtype=args.dtype,
        max_new_tokens=args.max_tokens,
    )

    model, processor = load_model(cfg)
    console.print("[bold green]Model loaded successfully!")

    summary = get_vram_summary()
    console.print(f"\n[bold]VRAM Summary:")
    for k, v in summary.items():
        console.print(f"  {k}: {v:.0f} MB")
