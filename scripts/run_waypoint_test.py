#!/usr/bin/env python3
"""Waypoint-1-Small 推論テスト — H100 GPU.

シード画像1枚からWaypoint-1-Smallで動画を生成し、
推論速度・VRAM使用量を計測する。

Usage:
    # デフォルト（HuggingFaceサンプル画像、60フレーム生成）
    uv run python scripts/run_waypoint_test.py

    # カスタム画像 & フレーム数
    uv run python scripts/run_waypoint_test.py --image path/to/seed.png --frames 120

    # 高品質（デノイズ8ステップ + compile）
    uv run python scripts/run_waypoint_test.py --steps 8 --frames 60

    # 量子化テスト（fp8, nvfp4）
    uv run python scripts/run_waypoint_test.py --quantize fp8

    # torch.compile スキップ（初回テスト用、コンパイル時間を省略）
    uv run python scripts/run_waypoint_test.py --no-compile
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Set, Tuple

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

console = Console()

MODEL_ID = "Overworld/Waypoint-1-Small"
DEFAULT_SEED_URL = (
    "https://gist.github.com/user-attachments/assets/"
    "4adc5a3d-6980-4d1e-b6e8-9033cdf61c66"
)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "waypoint_results"


@dataclass
class CtrlInput:
    """Waypoint control input — keyboard buttons + mouse velocity."""

    button: Set[int] = field(default_factory=set)
    mouse: Tuple[float, float] = (0.0, 0.0)


def random_ctrl() -> CtrlInput:
    """ランダムな操作入力を生成."""
    return random.choice([
        CtrlInput(button={48, 42}, mouse=(0.4, 0.3)),  # W+Shift + mouse
        CtrlInput(mouse=(0.1, 0.2)),                    # mouse only
        CtrlInput(button={48}, mouse=(0.0, 0.0)),       # W key (forward)
        CtrlInput(button={95, 32, 105}),                 # complex input
        CtrlInput(button={48}, mouse=(-0.3, 0.1)),       # forward + look left
    ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Waypoint-1-Small inference test")
    parser.add_argument("--image", type=str, default=None, help="シード画像パス（未指定でHFサンプル使用）")
    parser.add_argument("--prompt", type=str, default="A factory rooftop with industrial buildings",
                        help="テキストプロンプト")
    parser.add_argument("--frames", type=int, default=60, help="生成フレーム数")
    parser.add_argument("--steps", type=int, default=None,
                        help="デノイズステップ数（デフォルト4、モデル固有スケジュール使用）")
    parser.add_argument("--quantize", type=str, default=None, choices=["fp8", "nvfp4"],
                        help="量子化モード")
    parser.add_argument("--no-compile", action="store_true", help="torch.compileをスキップ")
    parser.add_argument("--gpu", type=int, default=0, help="使用するGPU ID")
    parser.add_argument("--output", type=str, default=None, help="出力動画パス")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")
    return parser.parse_args()


def load_seed_image(image_path: str | None):
    """シード画像を読み込む."""
    from diffusers.utils import load_image

    if image_path:
        console.print(f"  シード画像: {image_path}")
        return load_image(image_path)
    else:
        console.print(f"  シード画像: HuggingFace サンプル（デフォルト）")
        return load_image(DEFAULT_SEED_URL)


def get_gpu_info() -> dict:
    """GPU情報を取得."""
    dev = torch.cuda.current_device()
    info = {
        "gpu_id": dev,
        "gpu_name": torch.cuda.get_device_name(dev),
        "vram_total_gb": round(torch.cuda.get_device_properties(dev).total_memory / 1e9, 1),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
    }
    return info


def measure_vram() -> float:
    """現在のVRAM使用量 (GB) を返す."""
    return torch.cuda.max_memory_allocated() / 1e9


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    console.rule("[bold blue]Waypoint-1-Small 推論テスト")

    # --- GPU確認 ---
    if not torch.cuda.is_available():
        console.print("[red]ERROR: CUDA が利用できません。GPU環境で実行してください。[/red]")
        sys.exit(1)

    torch.cuda.set_device(args.gpu)
    gpu_info = get_gpu_info()
    console.print(f"  GPU: {gpu_info['gpu_name']}")
    console.print(f"  VRAM: {gpu_info['vram_total_gb']} GB")
    console.print(f"  PyTorch: {gpu_info['torch_version']}")
    console.print(f"  CUDA: {gpu_info['cuda_version']}")
    console.print()

    # --- モデルロード ---
    console.rule("[bold]Phase 1: モデルロード")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    from diffusers.modular_pipelines import ModularPipeline

    pipe = ModularPipeline.from_pretrained(MODEL_ID, trust_remote_code=True)
    pipe.load_components(
        device_map=f"cuda:{args.gpu}",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    pipe.transformer.apply_inference_patches()

    # 量子化
    if args.quantize:
        console.print(f"  量子化: {args.quantize}")
        pipe.transformer.quantize(args.quantize)

    # torch.compile
    if not args.no_compile:
        console.print("  torch.compile 適用中（初回は数分かかります）...")
        pipe.transformer.compile(fullgraph=True, mode="max-autotune", dynamic=False)
        pipe.vae.bake_weight_norm()
        pipe.vae.compile(fullgraph=True, mode="max-autotune")
    else:
        console.print("  torch.compile: スキップ")

    model_load_time = time.perf_counter() - t0
    model_vram = measure_vram()
    console.print(f"  ロード時間: {model_load_time:.1f}s")
    console.print(f"  VRAM使用量: {model_vram:.2f} GB")
    console.print()

    # --- シード画像読み込み ---
    console.rule("[bold]Phase 2: シード画像")
    image = load_seed_image(args.image)
    console.print(f"  画像サイズ: {image.size}")
    console.print(f"  プロンプト: {args.prompt}")
    console.print(f"  生成フレーム数: {args.frames}")
    console.print()

    # --- sigmas設定 ---
    sigmas_kwarg = {}
    if args.steps:
        # デフォルト: [1.0, 0.949, 0.840, 0.0] (3ステップ)
        # ステップ数に応じて1.0→0.0を均等分割
        sigmas_list = np.linspace(1.0, 0.0, args.steps + 1).tolist()
        sigmas_kwarg["scheduler_sigmas"] = torch.tensor(
            sigmas_list, dtype=torch.bfloat16, device=f"cuda:{args.gpu}"
        )
        console.print(f"  デノイズステップ: {args.steps} ({len(sigmas_list)} sigmas)")
    else:
        console.print("  デノイズステップ: 3（デフォルト）")
    console.print()

    # --- 推論 ---
    console.rule("[bold]Phase 3: フレーム生成")
    torch.cuda.reset_peak_memory_stats()

    outputs = []
    frame_times = []

    # 初回フレーム（ウォームアップ含む）
    t_start = time.perf_counter()
    ctrl = random_ctrl()
    state = pipe(prompt=args.prompt, image=image, button=ctrl.button, mouse=ctrl.mouse,
                 **sigmas_kwarg)
    outputs.append(state.values["images"])
    first_frame_time = time.perf_counter() - t_start
    frame_times.append(first_frame_time)
    console.print(f"  フレーム 1/{args.frames} — {first_frame_time:.3f}s（ウォームアップ含む）")

    # 後続フレーム
    state.values["image"] = None
    for i in range(1, args.frames):
        t_frame = time.perf_counter()
        ctrl = random_ctrl()
        state = pipe(
            state,
            prompt=args.prompt,
            button=ctrl.button,
            mouse=ctrl.mouse,
            output_type="pil",
            **sigmas_kwarg,
        )
        outputs.append(state.values["images"])
        dt = time.perf_counter() - t_frame
        frame_times.append(dt)

        if (i + 1) % 10 == 0:
            avg_recent = sum(frame_times[-10:]) / 10
            console.print(
                f"  フレーム {i+1}/{args.frames} — "
                f"直近10f平均: {avg_recent:.3f}s ({1/avg_recent:.1f} FPS)"
            )

    total_gen_time = time.perf_counter() - t_start
    inference_vram = measure_vram()
    console.print()

    # --- 動画保存 ---
    console.rule("[bold]Phase 4: 動画保存")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        steps_tag = f"_steps{args.steps}" if args.steps else ""
        quantize_tag = f"_{args.quantize}" if args.quantize else ""
        compile_tag = "_nocompile" if args.no_compile else ""
        gpu_tag = f"_gpu{args.gpu}" if args.gpu != 0 else ""
        output_path = OUTPUT_DIR / f"waypoint_test_{timestamp}{steps_tag}{quantize_tag}{compile_tag}{gpu_tag}.mp4"

    from diffusers.utils import export_to_video
    export_to_video(outputs, str(output_path), fps=30)
    console.print(f"  保存先: {output_path}")
    console.print(f"  ファイルサイズ: {output_path.stat().st_size / 1e6:.1f} MB")

    # シード画像も保存
    seed_img_path = output_path.with_suffix(".seed.png")
    image.save(str(seed_img_path))
    console.print(f"  シード画像: {seed_img_path}")
    console.print()

    # --- ベンチマーク結果 ---
    console.rule("[bold green]ベンチマーク結果")

    # フレーム時間の統計（初回ウォームアップを除外）
    steady_times = frame_times[1:] if len(frame_times) > 1 else frame_times
    avg_frame_time = sum(steady_times) / len(steady_times)
    min_frame_time = min(steady_times)
    max_frame_time = max(steady_times)

    results = {
        **gpu_info,
        "model_id": MODEL_ID,
        "prompt": args.prompt,
        "denoise_steps": args.steps or 3,
        "quantize": args.quantize,
        "torch_compile": not args.no_compile,
        "num_frames": args.frames,
        "model_load_s": round(model_load_time, 1),
        "model_vram_gb": round(model_vram, 2),
        "inference_peak_vram_gb": round(inference_vram, 2),
        "total_generation_s": round(total_gen_time, 1),
        "first_frame_s": round(first_frame_time, 3),
        "avg_frame_s": round(avg_frame_time, 3),
        "min_frame_s": round(min_frame_time, 3),
        "max_frame_s": round(max_frame_time, 3),
        "avg_fps": round(1 / avg_frame_time, 1),
        "output_path": str(output_path),
    }

    table = Table(title="Waypoint-1-Small Benchmark")
    table.add_column("項目", style="cyan")
    table.add_column("値", style="green")

    table.add_row("GPU", gpu_info["gpu_name"])
    table.add_row("VRAM (total)", f"{gpu_info['vram_total_gb']} GB")
    table.add_row("デノイズステップ", str(args.steps or 3))
    table.add_row("量子化", args.quantize or "none (bf16)")
    table.add_row("torch.compile", "有効" if not args.no_compile else "無効")
    table.add_row("─" * 20, "─" * 20)
    table.add_row("モデルロード時間", f"{model_load_time:.1f}s")
    table.add_row("モデルVRAM", f"{model_vram:.2f} GB")
    table.add_row("推論ピークVRAM", f"{inference_vram:.2f} GB")
    table.add_row("─" * 20, "─" * 20)
    table.add_row("生成フレーム数", str(args.frames))
    table.add_row("合計生成時間", f"{total_gen_time:.1f}s")
    table.add_row("初回フレーム", f"{first_frame_time:.3f}s")
    table.add_row("平均フレーム時間", f"{avg_frame_time:.3f}s")
    table.add_row("平均FPS", f"{1/avg_frame_time:.1f}")
    table.add_row("最速/最遅", f"{min_frame_time:.3f}s / {max_frame_time:.3f}s")

    console.print(table)

    # JSON保存
    results_path = output_path.with_suffix(".json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    console.print(f"\n  ベンチマーク結果: {results_path}")

    console.rule("[bold green]完了")
    console.print(f"  動画: {output_path}")
    console.print(f"  結果: {results_path}")
    console.print()


if __name__ == "__main__":
    main()
