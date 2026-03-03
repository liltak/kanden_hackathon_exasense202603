#!/usr/bin/env python3
"""Waypoint-1-Small 周回動画生成 — コロッセオを一周する.

シード画像から、前進+旋回の制御入力で建物の周りをぐるっと回る動画を生成する。

Usage:
    uv run python scripts/run_waypoint_orbit.py \
        --image data/sample/colosseum_frames/frame_001.jpg \
        --prompt "Ancient Roman Colosseum exterior, cinematic view" \
        --frames 300 --fps 30
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from rich.console import Console

console = Console()

MODEL_ID = "Overworld/Waypoint-1-Small"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "waypoint_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Waypoint orbit video generator")
    parser.add_argument("--image", type=str, required=True, help="シード画像パス")
    parser.add_argument("--prompt", type=str,
                        default="Ancient Roman Colosseum exterior, cinematic golden hour lighting",
                        help="テキストプロンプト")
    parser.add_argument("--frames", type=int, default=300, help="生成フレーム数（300f=10秒@30fps）")
    parser.add_argument("--fps", type=int, default=30, help="出力FPS")
    parser.add_argument("--gpu", type=int, default=0, help="使用GPU")
    parser.add_argument("--no-compile", action="store_true", help="torch.compileスキップ")
    parser.add_argument("--turn-speed", type=float, default=0.15,
                        help="旋回速度（0.05=ゆっくり, 0.3=速い）")
    parser.add_argument("--walk-speed", type=float, default=0.5,
                        help="前進キー押下率（0=停止, 1=常時前進）")
    parser.add_argument("--wobble", type=float, default=0.03,
                        help="視線の上下揺れ（歩行感）")
    parser.add_argument("--output", type=str, default=None, help="出力パス")
    return parser.parse_args()


def orbit_controls(frame_idx: int, total_frames: int, args):
    """フレームごとの制御入力を生成 — 周回軌道."""
    t = frame_idx / total_frames  # 0.0 → 1.0 で一周

    # 常に一定方向に旋回（右回り）
    mouse_x = args.turn_speed

    # 上下に軽い揺れ（歩いてる感じ）
    mouse_y = args.wobble * math.sin(frame_idx * 0.3)

    # 前進キー（W=48）を一定確率で押す
    buttons = set()
    if (frame_idx % 3) < (args.walk_speed * 3):
        buttons.add(48)  # W key (forward)

    # たまにゆっくり見回す演出
    # 1/4周ごとに少し立ち止まって見る
    quarter = int(t * 4) % 4
    quarter_progress = (t * 4) % 1.0
    if quarter_progress < 0.1:
        # 各1/4周の最初で少しスローダウン
        mouse_x *= 0.3
        buttons.discard(48)

    return buttons, (mouse_x, mouse_y)


def main():
    args = parse_args()
    torch.manual_seed(42)

    console.rule("[bold blue]Waypoint-1-Small 周回動画生成")

    if not torch.cuda.is_available():
        console.print("[red]ERROR: CUDAが必要です[/red]")
        sys.exit(1)

    torch.cuda.set_device(args.gpu)
    dev = torch.cuda.current_device()
    console.print(f"  GPU{dev}: {torch.cuda.get_device_name(dev)}")
    console.print(f"  フレーム: {args.frames} ({args.frames / args.fps:.1f}秒 @{args.fps}fps)")
    console.print(f"  旋回速度: {args.turn_speed}")
    console.print()

    # --- モデルロード ---
    console.rule("[bold]モデルロード")
    t0 = time.perf_counter()

    from diffusers.modular_pipelines import ModularPipeline

    pipe = ModularPipeline.from_pretrained(MODEL_ID, trust_remote_code=True)
    pipe.load_components(
        device_map=f"cuda:{args.gpu}",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    pipe.transformer.apply_inference_patches()

    if not args.no_compile:
        console.print("  torch.compile 適用中...")
        pipe.transformer.compile(fullgraph=True, mode="max-autotune", dynamic=False)
        pipe.vae.bake_weight_norm()
        pipe.vae.compile(fullgraph=True, mode="max-autotune")

    load_time = time.perf_counter() - t0
    console.print(f"  ロード完了: {load_time:.0f}s")
    console.print()

    # --- シード画像 ---
    from diffusers.utils import load_image
    image = load_image(args.image)
    console.print(f"  シード: {args.image} ({image.size})")
    console.print(f"  プロンプト: {args.prompt}")
    console.print()

    # --- フレーム生成 ---
    console.rule("[bold]フレーム生成（周回モード）")
    torch.cuda.reset_peak_memory_stats()

    outputs = []
    frame_times = []

    # 初回フレーム
    t_start = time.perf_counter()
    buttons, mouse = orbit_controls(0, args.frames, args)
    state = pipe(prompt=args.prompt, image=image, button=buttons, mouse=mouse)
    outputs.append(state.values["images"])
    first_time = time.perf_counter() - t_start
    frame_times.append(first_time)
    console.print(f"  フレーム 1/{args.frames} — {first_time:.1f}s（ウォームアップ）")

    # 後続フレーム
    state.values["image"] = None
    for i in range(1, args.frames):
        t_f = time.perf_counter()
        buttons, mouse = orbit_controls(i, args.frames, args)
        state = pipe(
            state,
            prompt=args.prompt,
            button=buttons,
            mouse=mouse,
            output_type="pil",
        )
        outputs.append(state.values["images"])
        dt = time.perf_counter() - t_f
        frame_times.append(dt)

        if (i + 1) % 30 == 0:
            progress = (i + 1) / args.frames * 100
            avg = sum(frame_times[-30:]) / 30
            console.print(
                f"  フレーム {i+1}/{args.frames} [{progress:.0f}%] — "
                f"{1/avg:.1f} FPS"
            )

    total_time = time.perf_counter() - t_start
    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    console.print()

    # --- 保存 ---
    console.rule("[bold]動画保存")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.output:
        output_path = Path(args.output)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"waypoint_orbit_{ts}_{args.frames}f.mp4"

    from diffusers.utils import export_to_video
    export_to_video(outputs, str(output_path), fps=args.fps)

    file_size = output_path.stat().st_size / 1e6
    console.print(f"  保存先: {output_path}")
    console.print(f"  サイズ: {file_size:.1f} MB")
    console.print(f"  再生時間: {args.frames / args.fps:.1f}秒")

    # シード画像保存
    seed_path = output_path.with_suffix(".seed.png")
    image.save(str(seed_path))

    # --- 結果 ---
    steady = frame_times[1:] if len(frame_times) > 1 else frame_times
    avg_fps = 1 / (sum(steady) / len(steady))

    results = {
        "mode": "orbit",
        "gpu": torch.cuda.get_device_name(dev),
        "frames": args.frames,
        "duration_s": round(args.frames / args.fps, 1),
        "fps_output": args.fps,
        "turn_speed": args.turn_speed,
        "total_generation_s": round(total_time, 1),
        "avg_fps_generation": round(avg_fps, 1),
        "peak_vram_gb": round(peak_vram, 2),
        "file_size_mb": round(file_size, 1),
        "prompt": args.prompt,
        "seed_image": args.image,
        "output": str(output_path),
    }

    results_path = output_path.with_suffix(".json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    console.rule("[bold green]完了")
    console.print(f"  動画: {output_path}")
    console.print(f"  {args.frames}フレーム / {args.frames/args.fps:.1f}秒 / 平均 {avg_fps:.1f} FPS生成")
    console.print(f"  ピークVRAM: {peak_vram:.1f} GB")
    console.print()


if __name__ == "__main__":
    main()
