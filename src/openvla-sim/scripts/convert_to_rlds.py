"""
RLDS/TFRecord 変換スクリプト

collect.py が出力した JSON + JPG データを
OpenVLA 互換の TFRecord (GZIP) に変換する。

【実行環境】Mac / H100 どちらでも動作
  Mac:   pip install tensorflow-macos tensorflow-metal
  H100:  pip install tensorflow

RLDS フォーマット:
  各 step:
    steps/observation/image    : (224, 224, 3) uint8  JPEG encoded
    steps/action               : (7,) float32  [vx, vy, vz, yaw_rate, 0, 0, 0]
    steps/language_instruction : string
    steps/is_first             : int64 (0/1)
    steps/is_last              : int64 (0/1)
    steps/is_terminal          : int64 (0/1)

アクション値の意味:
  vx/vy/vz  : ドローンボディフレームの速度 [m/s]
  yaw_rate  : ヨー角速度 [rad/s]
  最終ステップ (is_last=1): そのステップのアクションをそのまま記録

使用方法:
  python convert_to_rlds.py \\
    --input_dir dataset \\
    --output_dir dataset_rlds \\
    --split_ratio 0.9

検証のみ実行:
  python convert_to_rlds.py --test --output_dir dataset_rlds
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[WARNING] TensorFlow が見つかりません。")
    print("  Mac:  pip install tensorflow-macos tensorflow-metal")
    print("  H100: pip install tensorflow")


# ─── 定数 ─────────────────────────────────────────────────────────────────
IMAGE_SIZE = 224
ACTION_DIM = 7   # [vx, vy, vz, yaw_rate, 0, 0, 0]


# ─── シリアライズ ─────────────────────────────────────────────────────────
def _bytes_feature(value: bytes) -> "tf.train.Feature":
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_list_feature(value: list[float]) -> "tf.train.Feature":
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value: int) -> "tf.train.Feature":
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def step_to_example(
    image_rgb: np.ndarray,
    action_vector: list[float],
    instruction: str,
    is_first: bool,
    is_last: bool,
    is_terminal: bool,
) -> "tf.train.Example":
    """1 ステップを TFRecord の Example に変換する。"""
    img_resized = cv2.resize(image_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
    _, img_bytes = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # 4D → 7D に拡張 (collect.py は 4D: [vx_body, vy_body, vz_body, yaw_rate])
    action_4d = list(action_vector)[:4]
    action_7d = action_4d + [0.0] * (ACTION_DIM - len(action_4d))

    feature = {
        "steps/observation/image":    _bytes_feature(img_bytes.tobytes()),
        "steps/action":               _float_list_feature([float(v) for v in action_7d]),
        "steps/language_instruction": _bytes_feature(instruction.encode("utf-8")),
        "steps/is_first":             _int64_feature(int(is_first)),
        "steps/is_last":              _int64_feature(int(is_last)),
        "steps/is_terminal":          _int64_feature(int(is_terminal)),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


# ─── エピソード読み込み ───────────────────────────────────────────────────
def load_episodes(input_dir: Path) -> list[dict[str, Any]]:
    """episodes/*.json を全て読み込む。"""
    episode_dir = input_dir / "episodes"
    if not episode_dir.exists():
        raise FileNotFoundError(f"エピソードディレクトリが見つかりません: {episode_dir}")

    episodes = []
    for ep_path in sorted(episode_dir.glob("episode_*.json")):
        with open(ep_path, encoding="utf-8") as f:
            episodes.append(json.load(f))

    print(f"[convert] {len(episodes)} エピソードを読み込みました: {input_dir}")
    return episodes


# ─── TFRecord 書き込み ─────────────────────────────────────────────────────
def write_tfrecord(
    episodes: list[dict[str, Any]],
    output_path: Path,
    dataset_root: Path,
    split_name: str,
) -> int:
    """episodes を 1 つの GZIP TFRecord ファイルに書き込む。ステップ数を返す。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_steps = 0
    options = tf.io.TFRecordOptions(compression_type="GZIP")

    with tf.io.TFRecordWriter(str(output_path), options=options) as writer:
        for ep in episodes:
            steps = ep.get("steps", [])
            if not steps:
                continue
            for step in steps:
                img_path = dataset_root / step["image_path"]
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"[WARNING] 画像が見つかりません: {img_path} — スキップ")
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                example = step_to_example(
                    image_rgb=img_rgb,
                    action_vector=step["action_vector"],
                    instruction=step["instruction"],
                    is_first=step.get("is_first", False),
                    is_last=step.get("is_last", False),
                    is_terminal=step.get("is_terminal", False),
                )
                writer.write(example.SerializeToString())
                n_steps += 1

    print(f"[convert] {split_name}: {len(episodes)} ep, {n_steps} steps → {output_path}")
    return n_steps


# ─── 検証 ─────────────────────────────────────────────────────────────────
def verify_tfrecord(tfrecord_path: Path, n_samples: int = 5) -> None:
    """TFRecord を読み込んで形式を確認する。"""
    feature_spec = {
        "steps/observation/image":    tf.io.FixedLenFeature([], tf.string),
        "steps/action":               tf.io.FixedLenFeature([ACTION_DIM], tf.float32),
        "steps/language_instruction": tf.io.FixedLenFeature([], tf.string),
        "steps/is_first":             tf.io.FixedLenFeature([], tf.int64),
        "steps/is_last":              tf.io.FixedLenFeature([], tf.int64),
        "steps/is_terminal":          tf.io.FixedLenFeature([], tf.int64),
    }
    dataset = tf.data.TFRecordDataset(str(tfrecord_path), compression_type="GZIP")
    errors = []
    for i, raw in enumerate(dataset.take(n_samples)):
        try:
            ex = tf.io.parse_single_example(raw, feature_spec)
            img = tf.image.decode_jpeg(ex["steps/observation/image"], channels=3)
            assert img.shape == (IMAGE_SIZE, IMAGE_SIZE, 3), f"shape: {img.shape}"
            assert ex["steps/action"].shape == (ACTION_DIM,)
            assert len(ex["steps/language_instruction"].numpy()) > 0
            # アクションの先頭4次元 (vx, vy, vz, yaw_rate) が有限値か確認
            action = ex["steps/action"].numpy()
            assert np.all(np.isfinite(action[:4])), f"action に非有限値: {action}"
        except Exception as e:
            errors.append(f"sample {i}: {e}")

    if errors:
        for err in errors:
            print(f"  [ERROR] {err}")
        raise RuntimeError("検証失敗")
    print(f"[verify] OK ({n_samples} samples): {tfrecord_path}")


# ─── メイン ──────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="JSON+JPG → RLDS TFRecord 変換")
    parser.add_argument("--input_dir",    default="dataset",
                        help="collect.py の出力ディレクトリ")
    parser.add_argument("--output_dir",   default="dataset_rlds",
                        help="TFRecord の出力ディレクトリ")
    parser.add_argument("--split_ratio",  type=float, default=0.9,
                        help="train 割合 (default: 0.9)")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--test",         action="store_true",
                        help="既存 TFRecord の検証のみ実行")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.test:
        if not TF_AVAILABLE:
            print("[ERROR] TensorFlow が必要です。")
            return
        for name in ["train", "val"]:
            p = output_dir / f"{name}.tfrecord.gz"
            if p.exists():
                verify_tfrecord(p)
            else:
                print(f"[test] 見つかりません: {p}")
        return

    if not TF_AVAILABLE:
        print("[ERROR] TensorFlow が必要です。")
        print("  Mac:  pip install tensorflow-macos tensorflow-metal")
        print("  H100: pip install tensorflow")
        return

    input_dir = Path(args.input_dir)
    episodes = load_episodes(input_dir)

    # シャッフルして train/val 分割
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(episodes)).tolist()
    n_train = max(1, int(len(episodes) * args.split_ratio))
    train_eps = [episodes[i] for i in indices[:n_train]]
    val_eps   = [episodes[i] for i in indices[n_train:]] if len(indices) > n_train else []

    print(f"[convert] train={len(train_eps)}, val={len(val_eps)}")

    n_train_steps = write_tfrecord(
        train_eps, output_dir / "train.tfrecord.gz", input_dir, "train"
    )
    n_val_steps = 0
    if val_eps:
        n_val_steps = write_tfrecord(
            val_eps, output_dir / "val.tfrecord.gz", input_dir, "val"
        )

    print("\n[検証]")
    verify_tfrecord(output_dir / "train.tfrecord.gz")
    if val_eps:
        verify_tfrecord(output_dir / "val.tfrecord.gz")

    with open(output_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_train_episodes": len(train_eps),
            "n_val_episodes":   len(val_eps),
            "n_train_steps":    n_train_steps,
            "n_val_steps":      n_val_steps,
            "image_size":       IMAGE_SIZE,
            "action_dim":       ACTION_DIM,
            "action_format":    "[vx, vy, vz, yaw_rate, 0, 0, 0]",
            "compression":      "GZIP",
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[convert] 完了 → {output_dir}")


if __name__ == "__main__":
    main()
