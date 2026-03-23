"""
クラック追従推論スクリプト (infer.py) ― OpenVLA 7B LoRA

学習済み LoRA モデルを使って、クラック画像パッチから
次の移動量 [delta_x, delta_y] を予測する。

使い方:
  # 単一パッチ画像から予測
  python infer.py --ckpt_dir checkpoints/crack_openvla/best \\
                  --image data/auto_raw/patches/episode_0000_step_00.png

  # ディレクトリ内の全パッチを一括予測
  python infer.py --ckpt_dir checkpoints/crack_openvla/best \\
                  --image_dir data/auto_raw/patches/

  # エピソード JSON と照合して精度検証
  python infer.py --ckpt_dir checkpoints/crack_openvla/best \\
                  --episode data/auto_raw/episodes/episode_0000.json \\
                  --image_dir data/auto_raw/patches/
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
INSTRUCTION = "クラックを追従してください"


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


def infer_single(args: argparse.Namespace) -> None:
    """単一画像から予測して結果を表示する。"""
    import torch

    device = (
        torch.device("cuda:1") if torch.cuda.device_count() > 1
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"[device] {device}")

    model, processor, action_tokenizer = load_model(args.ckpt_dir, device)

    image = PILImage.open(args.image).convert("RGB")
    action = predict_action(model, processor, image, action_tokenizer, device)

    print(f"\n予測アクション:")
    print(f"  delta_x = {action[0]:+.2f} px")
    print(f"  delta_y = {action[1]:+.2f} px")
    dist = math.sqrt(action[0]**2 + action[1]**2)
    angle = math.degrees(math.atan2(action[1], action[0]))
    print(f"  距離    = {dist:.2f} px")
    print(f"  方向    = {angle:.1f}°")


def infer_episode(args: argparse.Namespace) -> None:
    """エピソード JSON と照合して予測精度を検証する。"""
    import torch

    device = (
        torch.device("cuda:1") if torch.cuda.device_count() > 1
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"[device] {device}")

    model, processor, action_tokenizer = load_model(args.ckpt_dir, device)

    with open(args.episode, encoding="utf-8") as f:
        episode = json.load(f)

    steps = [s for s in episode.get("steps", []) if s.get("patch_path") is not None]
    image_dir = Path(args.image_dir)

    # 出力ディレクトリ
    out_dir = None
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    errors = []
    patch_imgs = []
    preds_all = []  # 予測軌跡用
    print(f"\nエピソード {episode['episode_id']} の検証 ({len(steps)} ステップ)")
    print(f"{'Step':>5}  {'GT(dx,dy)':>20}  {'Pred(dx,dy)':>20}  {'Error':>8}")
    print("-" * 60)

    for i, step in enumerate(steps):
        patch_name = Path(step["patch_path"]).name
        patch_path = image_dir / patch_name
        if not patch_path.exists():
            print(f"  [{i}] スキップ: {patch_path} が見つかりません")
            continue

        image = PILImage.open(patch_path).convert("RGB")
        pred = predict_action(model, processor, image, action_tokenizer, device)
        gt = np.array(step["action_vector"][:2], dtype=np.float32)

        err = math.sqrt((pred[0]-gt[0])**2 + (pred[1]-gt[1])**2)
        errors.append(err)
        preds_all.append(pred)
        print(f"  {i:3d}  ({gt[0]:+7.1f},{gt[1]:+7.1f})  "
              f"({pred[0]:+7.1f},{pred[1]:+7.1f})  {err:7.2f}px")

        if out_dir:
            import cv2
            img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            cx, cy = 112, 112
            scale = 3.0
            cv2.arrowedLine(img_bgr, (cx, cy),
                            (int(cx + gt[0] * scale), int(cy + gt[1] * scale)),
                            (0, 200, 0), 2, tipLength=0.3)
            cv2.arrowedLine(img_bgr, (cx, cy),
                            (int(cx + pred[0] * scale), int(cy + pred[1] * scale)),
                            (0, 0, 220), 2, tipLength=0.3)
            cv2.putText(img_bgr, f"GT({gt[0]:+.0f},{gt[1]:+.0f})",
                        (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
            cv2.putText(img_bgr, f"Pred({pred[0]:+.0f},{pred[1]:+.0f}) err={err:.1f}px",
                        (4, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 220), 1)
            # ステップ番号
            cv2.putText(img_bgr, f"#{i}", (4, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imwrite(str(out_dir / f"step_{i:03d}.png"), img_bgr)
            patch_imgs.append(img_bgr)

    if errors:
        print(f"\n統計: 平均誤差={sum(errors)/len(errors):.2f}px | "
              f"最大={max(errors):.2f}px | 最小={min(errors):.2f}px")

    # 軌跡まとめ画像（元画像 + ROIを順番に描画）
    if out_dir and errors:
        import cv2

        # 元画像を探す
        src_img_name = episode.get("source_image", "")
        src_img_path = image_dir.parent / src_img_name  # patches/ の親 = auto_raw/
        if not src_img_path.exists():
            # crack_generated/ も試す
            src_img_path = Path("crack_generated") / src_img_name

        if src_img_path.exists():
            canvas = cv2.imread(str(src_img_path))
        else:
            # 元画像がない場合は黒背景に描画
            canvas = np.zeros((2000, 2000, 3), dtype=np.uint8)

        half = 112  # PATCH_SIZE // 2
        valid_steps = [s for s in steps if s.get("pixel_x") is not None]

        # GT軌跡（緑）
        for i, step in enumerate(valid_steps):
            x, y = step["pixel_x"], step["pixel_y"]
            color = (0, 200, 0) if i == 0 else (0, 0, 220) if i == len(valid_steps)-1 else (255, 180, 0)
            cv2.rectangle(canvas, (x - half, y - half), (x + half, y + half), color, 2)
            cv2.circle(canvas, (x, y), 4, color, -1)
            cv2.putText(canvas, str(i), (x - half + 4, y - half + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        for i in range(len(valid_steps) - 1):
            x0, y0 = valid_steps[i]["pixel_x"], valid_steps[i]["pixel_y"]
            x1, y1 = valid_steps[i+1]["pixel_x"], valid_steps[i+1]["pixel_y"]
            cv2.arrowedLine(canvas, (x0, y0), (x1, y1), (0, 220, 255), 2, tipLength=0.15)

        # 予測軌跡（赤）: 起点から予測deltaを累積
        if preds_all and valid_steps:
            px, py = float(valid_steps[0]["pixel_x"]), float(valid_steps[0]["pixel_y"])
            pred_positions = [(int(px), int(py))]
            for pred in preds_all:
                px += pred[0]
                py += pred[1]
                pred_positions.append((int(px), int(py)))

            for i, (x, y) in enumerate(pred_positions):
                cv2.rectangle(canvas, (x - half, y - half), (x + half, y + half), (0, 0, 200), 1)
                cv2.circle(canvas, (x, y), 4, (0, 0, 200), -1)
            for i in range(len(pred_positions) - 1):
                cv2.arrowedLine(canvas, pred_positions[i], pred_positions[i+1],
                                (80, 80, 255), 2, tipLength=0.15)

        # 凡例
        cv2.putText(canvas, "GT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
        cv2.putText(canvas, "Pred", (60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 255), 2)

        traj_path = out_dir / "trajectory.png"
        cv2.imwrite(str(traj_path), canvas)
        print(f"[保存] 軌跡画像 → {traj_path}")
        print(f"[保存] 個別パッチ {len(patch_imgs)} 枚 → {out_dir}/")


def infer_batch(args: argparse.Namespace) -> None:
    """ディレクトリ内の全パッチ画像を一括予測する。"""
    import torch

    device = (
        torch.device("cuda:1") if torch.cuda.device_count() > 1
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"[device] {device}")

    model, processor, action_tokenizer = load_model(args.ckpt_dir, device)

    image_dir = Path(args.image_dir)
    patches = sorted(image_dir.glob("*.png"))
    print(f"\n{len(patches)} 枚のパッチ画像を予測します...\n")

    results = []
    for patch_path in patches:
        image = PILImage.open(patch_path).convert("RGB")
        action = predict_action(model, processor, image, action_tokenizer, device)
        dist = math.sqrt(action[0]**2 + action[1]**2)
        results.append({
            "image": patch_path.name,
            "delta_x": float(action[0]),
            "delta_y": float(action[1]),
            "distance": float(dist),
        })
        print(f"  {patch_path.name}: dx={action[0]:+.1f}px  dy={action[1]:+.1f}px  dist={dist:.1f}px")

    # 結果を JSON で保存
    out_path = Path(args.ckpt_dir) / "infer_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n結果を保存: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="クラック追従 OpenVLA 7B LoRA 推論スクリプト"
    )
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="LoRA チェックポイントディレクトリ (action_stats.npz を含む)")
    parser.add_argument("--image", type=str, default=None,
                        help="単一パッチ画像パス (224×224 PNG)")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="パッチ画像ディレクトリ (一括処理 or エピソード検証)")
    parser.add_argument("--episode", type=str, default=None,
                        help="エピソード JSON パス (精度検証モード)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="パッチ画像の保存先（GT・予測矢印付き）")
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("[ERROR] PyTorch が必要です。H100 上で実行してください。")
        return

    if args.episode and args.image_dir:
        infer_episode(args)
    elif args.image_dir:
        infer_batch(args)
    elif args.image:
        infer_single(args)
    else:
        parser.error("--image か --image_dir を指定してください。")


if __name__ == "__main__":
    main()
