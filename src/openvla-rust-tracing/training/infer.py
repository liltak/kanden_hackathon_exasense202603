"""
クラック追従推論スクリプト (infer.py) ― OpenVLA 7B LoRA

学習済み LoRA モデルを使って、クラック追従推論を実行する。

使い方:
  python infer.py --ckpt_dir checkpoints/crack_openvla/best \\
                  --image path/to/crack_image.png \\
                  --x 300 --y 450
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

# action_tokenizer は同一ディレクトリにある
sys.path.insert(0, os.path.dirname(__file__))
from action_tokenizer import ActionTokenizer

# 命令文 (学習時と同一)
INSTRUCTION = "Follow the rust trace. Navigate to continue tracking the corrosion path."
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
        color = (0, 200, 0) if i == 0 else (0, 0, 220) if i == len(trajectory) - 1 else (255, 180, 0)
        cv2.rectangle(vis, (x - half, y - half), (x + half, y + half), color, 2)
        cv2.circle(vis, (x, y), 5, color, -1)
        cv2.putText(vis, str(i), (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    for i in range(len(trajectory) - 1):
        cv2.arrowedLine(vis, trajectory[i], trajectory[i + 1],
                        (0, 220, 255), 2, tipLength=0.15)

    cv2.imwrite(str(output_path), vis)
    print(f"[保存] 軌跡画像: {output_path}")


def load_model(ckpt_dir: str, device):
    """LoRA アダプターと ActionTokenizer を読み込んで返す。"""
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from peft import PeftModel

    processor = AutoProcessor.from_pretrained(ckpt_dir, trust_remote_code=True)
    base_model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=device.type == "cuda" and __import__("torch").bfloat16 or __import__("torch").float32,
        trust_remote_code=True,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, ckpt_dir)
    model.eval()

    stats_path = os.path.join(ckpt_dir, "action_stats.npz")
    action_tokenizer = ActionTokenizer.load(stats_path)
    print(f"モデル読み込み完了: {ckpt_dir}")
    print(f"ActionTokenizer: {action_tokenizer}")
    return model, processor, action_tokenizer


def predict_action(
    model,
    processor,
    pil_image: PILImage.Image,
    action_tokenizer: ActionTokenizer,
    device,
) -> np.ndarray:
    """
    224×224 パッチ画像から次の移動量を予測する。

    Returns:
        np.ndarray: [delta_x, delta_y] (ピクセル単位)
    """
    import torch

    inputs = processor(
        text=INSTRUCTION,
        images=pil_image,
        return_tensors="pt",
    ).to(device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=8,   # 2次元: "XXX YYY" = 最大7文字
            do_sample=False,
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    action = action_tokenizer.decode(generated_text)
    return action  # [delta_x, delta_y]


def infer_rollout(args: argparse.Namespace) -> None:
    """元画像 + 起点座標でクラックを自律追跡する。"""
    import torch

    device = (
        torch.device("cuda:1") if torch.cuda.device_count() > 1
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"[device] {device}")

    model, processor, action_tokenizer = load_model(args.ckpt_dir, device)

    img_pil = PILImage.open(args.image).convert("RGB")
    img_arr = np.array(img_pil)
    h, w = img_arr.shape[:2]
    print(f"[画像] {args.image}  ({w}×{h}px)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x, y = args.x, args.y
    trajectory = [(x, y)]
    results = []

    print(f"\n起点: ({x}, {y})  最大{args.max_steps}ステップ")
    print(f"{'Step':>5}  {'pos(x,y)':>16}  {'delta(dx,dy)':>20}  {'距離':>8}")
    print("-" * 56)

    for step in range(args.max_steps):
        patch = crop_patch(img_arr, x, y)
        action = predict_action(model, processor, patch, action_tokenizer, device)
        dx, dy = float(action[0]), float(action[1])
        dist = math.sqrt(dx**2 + dy**2)

        print(f"  {step:3d}  ({x:6d},{y:6d})  ({dx:+8.1f},{dy:+8.1f})  {dist:7.1f}px")
        results.append({"step": step, "x": x, "y": y, "delta_x": dx, "delta_y": dy, "distance": dist})

        x = int(round(x + dx))
        y = int(round(y + dy))

        if not (0 <= x < w and 0 <= y < h):
            print(f"  → 画像範囲外に出たため終了 ({x}, {y})")
            break

        trajectory.append((x, y))

    result_path = output_dir / "trajectory.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({"start": {"x": args.x, "y": args.y}, "steps": results}, f, indent=2)
    print(f"\n[保存] 軌跡データ: {result_path}")

    save_trajectory_image(img_arr, trajectory, output_dir / "trajectory.png")
    print(f"\n完了: {len(trajectory)} ステップ")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="クラック追従 OpenVLA 7B LoRA 推論スクリプト"
    )
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="LoRA チェックポイントディレクトリ (action_stats.npz を含む)")
    parser.add_argument("--image", type=str, required=True,
                        help="入力画像パス（元画像）")
    parser.add_argument("--x", type=int, required=True, help="ロールアウト起点 x 座標（ピクセル）")
    parser.add_argument("--y", type=int, required=True, help="ロールアウト起点 y 座標（ピクセル）")
    parser.add_argument("--max_steps", type=int, default=100, help="ロールアウト最大ステップ数（デフォルト: 30）")
    parser.add_argument("--output_dir", type=str, default="results/rollout",
                        help="結果の出力先")
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("[ERROR] PyTorch が必要です。H100 上で実行してください。")
        return

    infer_rollout(args)


if __name__ == "__main__":
    main()
