#!/usr/bin/env python3
"""Waypoint-1-Small ボタンIDスキャン — 各ボタンの効果を自動判定.

0〜255の各ボタンIDで短いフレーム列を生成し、
シード画像からの変化量（ピクセル差分）で「動いたかどうか」を判定する。

Usage:
    uv run python scripts/run_waypoint_button_scan.py \
        --image data/sample/colosseum_frames/frame_001.jpg \
        --gpu 0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

console = Console()

MODEL_ID = "Overworld/Waypoint-1-Small"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "waypoint_results"


def parse_args():
    parser = argparse.ArgumentParser(description="Waypoint button ID scanner")
    parser.add_argument("--image", type=str, required=True, help="シード画像パス")
    parser.add_argument("--prompt", type=str, default="A 3D world to explore")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--frames-per-button", type=int, default=10,
                        help="各ボタンで生成するフレーム数")
    parser.add_argument("--button-range", type=str, default="0-255",
                        help="スキャン範囲 (例: 0-255, 30-60)")
    parser.add_argument("--no-compile", action="store_true")
    return parser.parse_args()


def frames_to_arrays(frames):
    """PIL frames → numpy arrays."""
    arrays = []
    for f in frames:
        if hasattr(f, "convert"):
            arrays.append(np.array(f.convert("RGB"), dtype=np.float32))
        else:
            arrays.append(np.array(f, dtype=np.float32))
    return arrays


def measure_motion(arrays):
    """フレーム間の平均ピクセル変化量を計算."""
    if len(arrays) < 2:
        return 0.0, 0.0, (0.0, 0.0)

    diffs = []
    for i in range(1, len(arrays)):
        diff = np.abs(arrays[i] - arrays[i - 1]).mean()
        diffs.append(diff)

    # 全体の変化（最初と最後）
    total_diff = np.abs(arrays[-1] - arrays[0]).mean()

    # 左右の変化（水平方向の動き検出）
    h = arrays[0].shape[1]
    left_diff = np.abs(arrays[-1][:, :h//2] - arrays[0][:, :h//2]).mean()
    right_diff = np.abs(arrays[-1][:, h//2:] - arrays[0][:, h//2:]).mean()

    return float(np.mean(diffs)), float(total_diff), (float(left_diff), float(right_diff))


def main():
    args = parse_args()
    torch.manual_seed(42)
    torch.cuda.set_device(args.gpu)

    # ボタン範囲パース
    start, end = map(int, args.button_range.split("-"))

    console.rule("[bold blue]Waypoint ボタンIDスキャン")
    console.print(f"  GPU{args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    console.print(f"  スキャン範囲: {start}-{end} ({end - start + 1} buttons)")
    console.print(f"  フレーム/ボタン: {args.frames_per_button}")
    console.print()

    # --- モデルロード ---
    console.rule("[bold]モデルロード")
    from diffusers.modular_pipelines import ModularPipeline
    from diffusers.utils import load_image

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

    console.print("  ロード完了")
    console.print()

    image = load_image(args.image)

    # --- ベースライン（ボタンなし） ---
    console.rule("[bold]ベースライン（操作なし）")
    baseline_frames = []
    state = pipe(prompt=args.prompt, image=image, button=set(), mouse=(0.0, 0.0))
    baseline_frames.append(state.values["images"])
    state.values["image"] = None
    for _ in range(args.frames_per_button - 1):
        state = pipe(state, prompt=args.prompt, button=set(), mouse=(0.0, 0.0),
                     output_type="pil")
        baseline_frames.append(state.values["images"])

    baseline_arrays = frames_to_arrays(baseline_frames)
    base_avg, base_total, _ = measure_motion(baseline_arrays)
    console.print(f"  ベースライン変化量: avg={base_avg:.2f}, total={base_total:.2f}")
    console.print()

    # --- 各ボタンスキャン ---
    console.rule("[bold]ボタンスキャン")
    results = []

    for btn_id in range(start, end + 1):
        torch.manual_seed(42)  # 毎回同じシードで比較

        frames = []
        state = pipe(prompt=args.prompt, image=image, button={btn_id}, mouse=(0.0, 0.0))
        frames.append(state.values["images"])
        state.values["image"] = None
        for _ in range(args.frames_per_button - 1):
            state = pipe(state, prompt=args.prompt, button={btn_id}, mouse=(0.0, 0.0),
                         output_type="pil")
            frames.append(state.values["images"])

        arrays = frames_to_arrays(frames)
        avg_diff, total_diff, (left_d, right_d) = measure_motion(arrays)

        # ベースラインからの差分（ボタンの効果）
        effect = total_diff - base_total

        entry = {
            "button_id": btn_id,
            "avg_frame_diff": round(avg_diff, 2),
            "total_diff": round(total_diff, 2),
            "effect_vs_baseline": round(effect, 2),
            "left_diff": round(left_d, 2),
            "right_diff": round(right_d, 2),
            "lr_bias": round(right_d - left_d, 2),
        }
        results.append(entry)

        marker = ""
        if abs(effect) > 3.0:
            marker = " ★★★"
        elif abs(effect) > 1.5:
            marker = " ★★"
        elif abs(effect) > 0.5:
            marker = " ★"

        if (btn_id - start + 1) % 16 == 0 or btn_id == end or marker:
            console.print(
                f"  Button {btn_id:3d}: "
                f"effect={effect:+.2f}, total={total_diff:.2f}, "
                f"LR_bias={right_d - left_d:+.2f}{marker}"
            )

    # --- 結果集計 ---
    console.rule("[bold green]結果サマリー")

    # 効果が大きいボタンをソート
    sorted_results = sorted(results, key=lambda x: abs(x["effect_vs_baseline"]), reverse=True)

    table = Table(title="効果の大きいボタンID TOP 20")
    table.add_column("Button ID", style="cyan")
    table.add_column("Effect", style="green")
    table.add_column("Total Diff", style="yellow")
    table.add_column("LR Bias", style="magenta")
    table.add_column("推定動作", style="white")

    for entry in sorted_results[:20]:
        # 推定動作
        effect = entry["effect_vs_baseline"]
        lr = entry["lr_bias"]
        guess = ""
        if abs(effect) > 1.0:
            if lr > 2.0:
                guess = "左移動?"
            elif lr < -2.0:
                guess = "右移動?"
            elif effect > 0:
                guess = "前進/動き?"
            else:
                guess = "後退/停止?"

        table.add_row(
            str(entry["button_id"]),
            f"{entry['effect_vs_baseline']:+.2f}",
            f"{entry['total_diff']:.2f}",
            f"{entry['lr_bias']:+.2f}",
            guess,
        )

    console.print(table)

    # JSON保存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    results_path = OUTPUT_DIR / f"button_scan_{ts}.json"
    with open(results_path, "w") as f:
        json.dump({
            "baseline": {"avg_diff": base_avg, "total_diff": base_total},
            "buttons": results,
            "config": {
                "frames_per_button": args.frames_per_button,
                "prompt": args.prompt,
                "image": args.image,
                "range": f"{start}-{end}",
            },
        }, f, indent=2, ensure_ascii=False)

    console.print(f"\n  結果: {results_path}")
    console.rule("[bold green]完了")


if __name__ == "__main__":
    main()
