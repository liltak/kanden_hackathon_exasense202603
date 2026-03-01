"""Unsloth fine-tuning pipeline for Qwen3.5-VL.

LoRA-based fine-tuning with VRAM monitoring, checkpoint saving,
and optional Wandb logging. Targets H100 GPU server.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from typing import TYPE_CHECKING

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.vlm.model_loader import get_vram_summary, get_vram_usage_mb

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class LoRAConfig:
    """LoRA configuration for Qwen3.5-VL fine-tuning."""

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    modules_to_save: list[str] | None = None


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    output_dir: str = "outputs/vlm-finetune"
    num_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = True
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    dataloader_num_workers: int = 4
    max_seq_length: int = 2048
    seed: int = 42
    report_to: str = "none"  # "wandb" or "none"
    wandb_project: str = "exasense-vlm"
    wandb_run_name: str | None = None
    resume_from_checkpoint: str | None = None


@dataclass
class TrainingResult:
    """Fine-tuning result summary."""

    output_dir: str
    total_steps: int
    train_loss: float
    eval_loss: float | None
    training_time_seconds: float
    vram_peak_mb: float
    best_checkpoint: str | None = None
    metrics: dict = field(default_factory=dict)


def setup_wandb(config: TrainingConfig) -> None:
    """Initialize Wandb logging if configured."""
    if config.report_to != "wandb":
        return

    try:
        import wandb

        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config={
                "lora_r": config.gradient_accumulation_steps,
                "learning_rate": config.learning_rate,
                "epochs": config.num_epochs,
                "batch_size": config.per_device_train_batch_size,
            },
        )
        console.print("[green]Wandb initialized")
    except ImportError:
        logger.warning("wandb not installed. Disabling wandb logging.")
        config.report_to = "none"


def load_model_for_training(
    model_id: str = "Qwen/Qwen3.5-VL-7B-Instruct",
    lora_config: LoRAConfig | None = None,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
) -> tuple:
    """Load model with Unsloth optimizations for fine-tuning.

    Args:
        model_id: HuggingFace model ID.
        lora_config: LoRA configuration.
        max_seq_length: Maximum sequence length.
        load_in_4bit: Use 4-bit quantization.

    Returns:
        Tuple of (model, tokenizer) with LoRA adapters applied.
    """
    if lora_config is None:
        lora_config = LoRAConfig()

    from unsloth import FastVisionModel

    console.print(f"[bold blue]Loading model for fine-tuning: {model_id}")
    console.print(f"  LoRA rank: {lora_config.r}, alpha: {lora_config.lora_alpha}")
    console.print(f"  4-bit: {load_in_4bit}, max_seq_length: {max_seq_length}")

    vram_before = get_vram_usage_mb()
    t0 = time.perf_counter()

    model, tokenizer = FastVisionModel.from_pretrained(
        model_id,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=torch.bfloat16,
    )

    model = FastVisionModel.get_peft_model(
        model,
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=lora_config.target_modules,
        bias=lora_config.bias,
        use_gradient_checkpointing="unsloth",
        random_state=42,
        modules_to_save=lora_config.modules_to_save,
    )

    elapsed = time.perf_counter() - t0
    vram_after = get_vram_usage_mb()

    console.print(f"[green]Model ready for training in {elapsed:.1f}s")
    console.print(f"  VRAM: {vram_after - vram_before:.0f} MB used")

    _print_trainable_parameters(model)

    return model, tokenizer


def _print_trainable_parameters(model) -> None:
    """Print trainable vs total parameters."""
    trainable = 0
    total = 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()

    pct = 100 * trainable / total if total > 0 else 0

    table = Table(title="Model Parameters")
    table.add_column("Type", style="cyan")
    table.add_column("Count", justify="right")
    table.add_row("Total", f"{total:,}")
    table.add_row("Trainable", f"{trainable:,}")
    table.add_row("Trainable %", f"{pct:.2f}%")
    console.print(table)


def load_dataset_from_jsonl(
    train_path: str | Path,
    eval_path: str | Path | None = None,
) -> tuple:
    """Load training and evaluation datasets from JSONL files.

    Each line should be a JSON object with ChatML conversation format:
    {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

    Args:
        train_path: Path to training JSONL file.
        eval_path: Path to evaluation JSONL file (optional).

    Returns:
        Tuple of (train_dataset, eval_dataset). eval_dataset may be None.
    """
    from datasets import Dataset

    def _load_jsonl(path: Path) -> list[dict]:
        data = []
        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning("Skipping invalid JSON at line %d: %s", line_num, e)
        return data

    train_data = _load_jsonl(Path(train_path))
    console.print(f"[blue]Loaded {len(train_data)} training samples from {train_path}")

    train_dataset = Dataset.from_list(train_data)

    eval_dataset = None
    if eval_path is not None:
        eval_data = _load_jsonl(Path(eval_path))
        console.print(f"[blue]Loaded {len(eval_data)} eval samples from {eval_path}")
        eval_dataset = Dataset.from_list(eval_data)

    return train_dataset, eval_dataset


def run_training(
    model,
    tokenizer,
    train_dataset,
    eval_dataset=None,
    training_config: TrainingConfig | None = None,
) -> TrainingResult:
    """Run the fine-tuning training loop.

    Args:
        model: Model with LoRA adapters.
        tokenizer: Tokenizer/processor.
        train_dataset: Training dataset.
        eval_dataset: Evaluation dataset (optional).
        training_config: Training hyperparameters.

    Returns:
        TrainingResult with metrics.
    """
    if training_config is None:
        training_config = TrainingConfig()

    from trl import SFTConfig, SFTTrainer
    from unsloth import is_bfloat16_supported

    output_dir = Path(training_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_wandb(training_config)

    console.print("[bold blue]Starting fine-tuning")
    console.print(f"  Output: {output_dir}")
    console.print(f"  Epochs: {training_config.num_epochs}")
    console.print(f"  Batch size: {training_config.per_device_train_batch_size}")
    console.print(f"  Gradient accumulation: {training_config.gradient_accumulation_steps}")
    console.print(f"  Learning rate: {training_config.learning_rate}")
    console.print(f"  Train samples: {len(train_dataset)}")
    if eval_dataset:
        console.print(f"  Eval samples: {len(eval_dataset)}")

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=training_config.num_epochs,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        per_device_eval_batch_size=training_config.per_device_eval_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        warmup_ratio=training_config.warmup_ratio,
        lr_scheduler_type=training_config.lr_scheduler_type,
        max_grad_norm=training_config.max_grad_norm,
        fp16=training_config.fp16,
        bf16=is_bfloat16_supported() if training_config.bf16 else False,
        logging_steps=training_config.logging_steps,
        save_steps=training_config.save_steps,
        eval_steps=training_config.eval_steps if eval_dataset else None,
        eval_strategy="steps" if eval_dataset else "no",
        save_total_limit=training_config.save_total_limit,
        dataloader_num_workers=training_config.dataloader_num_workers,
        max_seq_length=training_config.max_seq_length,
        seed=training_config.seed,
        report_to=training_config.report_to,
        load_best_model_at_end=eval_dataset is not None,
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
    )

    vram_before = get_vram_usage_mb()
    t0 = time.perf_counter()

    train_result = trainer.train(
        resume_from_checkpoint=training_config.resume_from_checkpoint,
    )

    elapsed = time.perf_counter() - t0
    vram_peak = get_vram_usage_mb()

    train_loss = train_result.training_loss
    eval_loss = None
    if eval_dataset:
        eval_metrics = trainer.evaluate()
        eval_loss = eval_metrics.get("eval_loss")

    # Save final model
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    console.print(f"[green]Final model saved to {final_dir}")

    # Log VRAM summary
    summary = get_vram_summary()
    console.print(f"\n[bold]Training Complete")
    console.print(f"  Time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    console.print(f"  Train loss: {train_loss:.4f}")
    if eval_loss is not None:
        console.print(f"  Eval loss: {eval_loss:.4f}")
    console.print(f"  VRAM peak: {vram_peak:.0f} MB")
    console.print(f"  Steps: {train_result.global_step}")

    return TrainingResult(
        output_dir=str(output_dir),
        total_steps=train_result.global_step,
        train_loss=train_loss,
        eval_loss=eval_loss,
        training_time_seconds=round(elapsed, 1),
        vram_peak_mb=round(vram_peak, 0),
        best_checkpoint=str(final_dir),
        metrics=train_result.metrics,
    )


def save_adapter(model, output_dir: str | Path) -> None:
    """Save LoRA adapter weights only.

    Args:
        model: Model with LoRA adapters.
        output_dir: Directory to save adapter weights.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(output_dir))
    console.print(f"[green]Adapter weights saved to {output_dir}")
    logger.info("Adapter saved to %s", output_dir)


def save_merged_model(
    model,
    tokenizer,
    output_dir: str | Path,
    quantization_method: str = "q4_k_m",
) -> None:
    """Merge LoRA adapters and save as GGUF for deployment.

    Args:
        model: Model with LoRA adapters.
        tokenizer: Tokenizer.
        output_dir: Directory to save merged model.
        quantization_method: GGUF quantization method.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as prog:
        prog.add_task("Merging LoRA adapters and saving...", total=None)
        model.save_pretrained_merged(
            str(output_dir),
            tokenizer,
            save_method="merged_16bit",
        )

    console.print(f"[green]Merged model saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Qwen3.5-VL with Unsloth + LoRA")
    parser.add_argument("--train-data", required=True, help="Path to training JSONL")
    parser.add_argument("--eval-data", default=None, help="Path to eval JSONL")
    parser.add_argument("--output-dir", default="outputs/vlm-finetune")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-VL-7B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--wandb", action="store_true", help="Enable Wandb logging")
    parser.add_argument("--save-merged", action="store_true", help="Save merged model after training")
    args = parser.parse_args()

    lora_cfg = LoRAConfig(r=args.lora_r, lora_alpha=args.lora_alpha)
    train_cfg = TrainingConfig(
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
        report_to="wandb" if args.wandb else "none",
    )

    console.print("[bold blue]ExaSense VLM Fine-tuning")
    console.print(f"  Model: {args.model_id}")
    console.print(f"  Train data: {args.train_data}")

    model, tokenizer = load_model_for_training(
        model_id=args.model_id,
        lora_config=lora_cfg,
        max_seq_length=args.max_seq_length,
    )

    train_dataset, eval_dataset = load_dataset_from_jsonl(
        args.train_data,
        args.eval_data,
    )

    result = run_training(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_config=train_cfg,
    )

    save_adapter(model, Path(args.output_dir) / "adapter")

    if args.save_merged:
        save_merged_model(model, tokenizer, Path(args.output_dir) / "merged")

    console.print(f"\n[bold green]Training complete!")
    console.print(f"  Steps: {result.total_steps}")
    console.print(f"  Train loss: {result.train_loss:.4f}")
    if result.eval_loss is not None:
        console.print(f"  Eval loss: {result.eval_loss:.4f}")
    console.print(f"  Time: {result.training_time_seconds:.0f}s")
    console.print(f"  Output: {result.output_dir}")
