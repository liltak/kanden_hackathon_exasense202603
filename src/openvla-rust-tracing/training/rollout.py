"""
自然画像ロールアウトスクリプト

学習済み LoRA モデルを使って、自然画像上でクラック追従を実行する。
起点座標を手動で指定し、モデルが次の移動量を繰り返し予測する。

使い方:
  python training/rollout.py \
    --ckpt_dir checkpoints/crack_openvla/best \
    --image    path/to/crack_image.png \
    --x 300 --y 450 \
    --max_steps 30 \
    --output_dir results/rollout
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

sys.path.insert(0, os.path.dirname(__file__))
from action_tokenizer import ActionTokenizer

INSTRUCTION = "クラックを追従してください"
PATCH_SIZE = 224


def crop_patch(img_arr: np.ndarray, x: int, y: int) -> PILImage.Image:
    """(x, y) を中心とした 224×224 パッチを切り出す（ゼロパディング）。"""
    h, w = img_arr.shape[:2]
    half = PATCH_SIZE // 2
    canvas = np.zeros((h + PATCH_SIZE, w + PATCH_SIZE, 3), dtype=np.uint8)
    canvas[half:half + h, half:half + w] = img_arr
    cx, cy = x + half, y + half
    patch = canvas[cy - half:cy + half, cx - half:cx + half]
    return PILImage.fromarray(patch)


def save_trajectory_image(
    img_arr: np.ndarray,
    trajectory: list[tuple[int, int]],
    output_path: Path,
) -> None:
    """軌跡を元画像上に描画して保存する。"""
    import cv2
    vis = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    half = PATCH_SIZE // 2

    for i, (x, y) in enumerate(trajectory):
        # ROI矩形
        color = (0, 200, 0) if i == 0 else (0, 0, 220) if i == len(trajectory) - 1 else (255, 180, 0)
        cv2.rectangle(vis, (x - half, y - half), (x + half, y + half), color, 2)
        cv2.circle(vis, (x, y), 5, color, -1)
        cv2.putText(vis, str(i), (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # 経路を矢印で接続
    for i in range(len(trajectory) - 1):
        cv2.arrowedLine(vis, trajectory[i], trajectory[i + 1],
                        (0, 220, 255), 2, tipLength=0.15)

    cv2.imwrite(str(output_path), vis)
    print(f"[保存] 軌跡画像: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="自然画像ロールアウト（手動起点指定）")
    parser.add_argument("--ckpt_dir",   required=True, help="LoRAチェックポイントディレクトリ")
    parser.add_argument("--image",      required=True, help="入力画像パス")
    parser.add_argument("--x",          type=int, required=True, help="起点 x 座標（ピクセル）")
    parser.add_argument("--y",          type=int, required=True, help="起点 y 座標（ピクセル）")
    parser.add_argument("--max_steps",  type=int, default=30, help="最大ステップ数（デフォルト: 30）")
    parser.add_argument("--output_dir", default="results/rollout", help="結果の出力先")
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("[ERROR] PyTorch が必要です。")
        return

    from infer import load_model, predict_action

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[device] {device}")

    # モデル読み込み
    model, processor, action_tokenizer = load_model(args.ckpt_dir, device)

    # 画像読み込み
    img_pil = PILImage.open(args.image).convert("RGB")
    img_arr = np.array(img_pil)
    h, w = img_arr.shape[:2]
    print(f"[画像] {args.image}  ({w}×{h}px)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ロールアウト
    x, y = args.x, args.y
    trajectory = [(x, y)]
    results = []

    print(f"\n起点: ({x}, {y})  最大{args.max_steps}ステップ")
    print(f"{'Step':>5}  {'pos(x,y)':>16}  {'delta(dx,dy)':>20}  {'距離':>8}")
    print("-" * 56)

    for step in range(args.max_steps):
        patch = crop_patch(img_arr, x, y)

        # パッチ保存
        patch_path = output_dir / f"patch_{step:03d}.png"
        patch.save(patch_path)

        # 推論
        action = predict_action(model, processor, patch, action_tokenizer, device)
        dx, dy = float(action[0]), float(action[1])
        dist = math.sqrt(dx**2 + dy**2)

        print(f"  {step:3d}  ({x:6d},{y:6d})  ({dx:+8.1f},{dy:+8.1f})  {dist:7.1f}px")

        results.append({
            "step": step,
            "x": x, "y": y,
            "delta_x": dx, "delta_y": dy,
            "distance": dist,
        })

        # 次の位置へ移動
        x = int(round(x + dx))
        y = int(round(y + dy))

        # 画像範囲外に出たら終了
        if not (0 <= x < w and 0 <= y < h):
            print(f"  → 画像範囲外に出たため終了 ({x}, {y})")
            break

        trajectory.append((x, y))

    # 結果保存
    result_path = output_dir / "trajectory.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({"start": {"x": args.x, "y": args.y}, "steps": results}, f, indent=2)
    print(f"\n[保存] 軌跡データ: {result_path}")

    # 軌跡の可視化
    save_trajectory_image(img_arr, trajectory, output_dir / "trajectory.png")
    print(f"\n完了: {len(trajectory)} ステップ")


if __name__ == "__main__":
    main()
