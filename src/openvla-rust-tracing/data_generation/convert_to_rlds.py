"""
タスク2: RLDS/TFRecord 変換スクリプト

【実行環境】Mac ローカル / H100 どちらでも動作
  依存: opencv-python, numpy, tensorflow
  - Mac (Apple Silicon):  pip install tensorflow-macos tensorflow-metal opencv-python
  - Linux / H100:         pip install tensorflow opencv-python

generate_dataset.py の出力を OpenVLA が読める
RLDS 互換の TFRecord 形式に変換する。

RLDS フォーマット:
  各 episode は steps のシーケンス。
  各 step:
    observation:
      image: (224, 224, 3) uint8
    action: (7,) float32  ← OpenVLA 標準は 7次元だが本実装では 3次元を左詰め
    language_instruction: string
    is_first: bool
    is_last: bool
    is_terminal: bool

使用方法:
  python convert_to_rlds.py \
    --input_dir data/rust_dataset \
    --output_dir data/rust_rlds \
    --split_ratio 0.9

テスト:
  python convert_to_rlds.py --test --output_dir data/rust_rlds
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

# TensorFlow: Mac では tensorflow-macos, Linux/H100 では tensorflow を使用。
# どちらも pip でインストール可能。未インストールでも import エラーにならないようにする。
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[WARNING] TensorFlow が見つかりません。")
    print("  Mac:   pip install tensorflow-macos tensorflow-metal")
    print("  H100:  pip install tensorflow")


# ─── 定数 ─────────────────────────────────────────────────────────────────
IMAGE_SIZE = 224
ACTION_DIM = 7      # OpenVLA 標準の出力次元
RUST_ACTION_DIM = 3  # 本実装の 3D ベクトル → 先頭3次元に入れ、残りはゼロ埋め

FEATURE_DESCRIPTION = {
    "steps/observation/image": tf.io.FixedLenFeature([], tf.string),
    "steps/action": tf.io.FixedLenFeature([ACTION_DIM], tf.float32),
    "steps/language_instruction": tf.io.FixedLenFeature([], tf.string),
    "steps/is_first": tf.io.FixedLenFeature([], tf.int64),
    "steps/is_last": tf.io.FixedLenFeature([], tf.int64),
    "steps/is_terminal": tf.io.FixedLenFeature([], tf.int64),
}


# ─── シリアライズ ─────────────────────────────────────────────────────────
def _bytes_feature(value: bytes) -> "tf.train.Feature":
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_list_feature(value: list[float]) -> "tf.train.Feature":
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value: int) -> "tf.train.Feature":
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def step_to_example(
    image_path: str,
    action_vector: list[float],
    instruction: str,
    is_first: bool,
    is_last: bool,
    is_terminal: bool,
    dataset_root: Path,
) -> "tf.train.Example":
    """
    1ステップを TFRecord の Example に変換する。

    action_vector は 3D → ACTION_DIM (7) にゼロ埋めして格納する。
    """
    # 画像を読み込み JPEG エンコード
    full_path = dataset_root / image_path
    img = cv2.imread(str(full_path))
    if img is None:
        raise FileNotFoundError(f"画像が見つかりません: {full_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    _, img_bytes = cv2.imencode(".jpg", img_resized)

    # アクションを 7D にゼロ埋め
    action_7d = [0.0] * ACTION_DIM
    for i, v in enumerate(action_vector[:RUST_ACTION_DIM]):
        action_7d[i] = float(v)

    feature = {
        "steps/observation/image": _bytes_feature(img_bytes.tobytes()),
        "steps/action": _float_list_feature(action_7d),
        "steps/language_instruction": _bytes_feature(instruction.encode("utf-8")),
        "steps/is_first": _int64_feature(int(is_first)),
        "steps/is_last": _int64_feature(int(is_last)),
        "steps/is_terminal": _int64_feature(int(is_terminal)),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


# ─── エピソード読み込み ───────────────────────────────────────────────────
def load_episodes(input_dir: Path) -> list[dict[str, Any]]:
    """episodes/ ディレクトリから全エピソードを読み込む。"""
    episodes = []
    episode_dir = input_dir / "episodes"
    if not episode_dir.exists():
        raise FileNotFoundError(f"エピソードディレクトリが見つかりません: {episode_dir}")

    for ep_path in sorted(episode_dir.glob("episode_*.json")):
        with open(ep_path) as f:
            episodes.append(json.load(f))

    print(f"[convert_to_rlds] {len(episodes)} エピソードを読み込みました。")
    return episodes


# ─── TFRecord 書き込み ─────────────────────────────────────────────────────
def write_split(
    episodes: list[dict[str, Any]],
    output_path: Path,
    dataset_root: Path,
    split_name: str,
) -> int:
    """
    episodes リストを 1 つの TFRecord ファイルに書き込む。
    Returns: 書き込んだステップ数
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_steps = 0

    options = tf.io.TFRecordOptions(compression_type="GZIP")
    with tf.io.TFRecordWriter(str(output_path), options=options) as writer:
        for ep in episodes:
            steps = ep.get("steps", [])
            if not steps:
                continue

            for i, step in enumerate(steps):
                is_first = (i == 0)
                is_last = (i == len(steps) - 1)
                is_terminal = is_last

                try:
                    example = step_to_example(
                        image_path=step["image_path"],
                        action_vector=step["action_vector"],
                        instruction=step["instruction"],
                        is_first=is_first,
                        is_last=is_last,
                        is_terminal=is_terminal,
                        dataset_root=dataset_root,
                    )
                    writer.write(example.SerializeToString())
                    n_steps += 1
                except FileNotFoundError as e:
                    print(f"[WARNING] {e} — スキップします")

    print(f"[convert_to_rlds] {split_name}: {len(episodes)} エピソード, "
          f"{n_steps} ステップ → {output_path}")
    return n_steps


# ─── データローダー検証 ───────────────────────────────────────────────────
def verify_tfrecord(tfrecord_path: Path, n_samples: int = 5) -> None:
    """
    TFRecord を OpenVLA 互換のデータローダーで読み込めることを確認する。
    """
    if not TF_AVAILABLE:
        print("[verify] TF が利用不可のためスキップします。")
        return

    dataset = tf.data.TFRecordDataset(
        str(tfrecord_path),
        compression_type="GZIP",
    )

    errors = []
    for i, raw_record in enumerate(dataset.take(n_samples)):
        try:
            example = tf.io.parse_single_example(raw_record, FEATURE_DESCRIPTION)

            # 画像デコード
            img_raw = example["steps/observation/image"]
            img = tf.image.decode_jpeg(img_raw, channels=3)
            assert img.shape == (IMAGE_SIZE, IMAGE_SIZE, 3), \
                f"画像サイズが不正: {img.shape}"

            # アクション
            action = example["steps/action"]
            assert action.shape == (ACTION_DIM,), \
                f"アクション次元が不正: {action.shape}"

            # 言語指示
            instruction = example["steps/language_instruction"].numpy().decode("utf-8")
            assert len(instruction) > 0, "instruction が空です"

        except Exception as e:
            errors.append(f"サンプル {i}: {e}")

    if errors:
        print(f"[verify] 検証エラー ({len(errors)} 件):")
        for err in errors:
            print(f"  - {err}")
        raise RuntimeError("TFRecord 検証に失敗しました。")

    print(f"[verify] ✓ {n_samples} サンプルの検証OK: {tfrecord_path}")


# ─── データセット統計 ─────────────────────────────────────────────────────
def print_dataset_stats(tfrecord_path: Path) -> None:
    """TFRecord の統計情報を表示する (アクション分布など)。"""
    if not TF_AVAILABLE:
        return

    dataset = tf.data.TFRecordDataset(str(tfrecord_path), compression_type="GZIP")
    action_counts: dict[str, int] = {}
    n_total = 0
    n_backtrack = 0

    for raw_record in dataset:
        example = tf.io.parse_single_example(raw_record, FEATURE_DESCRIPTION)
        action = example["steps/action"].numpy()
        z_val = action[2]  # z=1 ならバックトラック
        n_total += 1
        if z_val > 0.5:
            n_backtrack += 1

    print(f"\n[統計] TFRecord: {tfrecord_path.name}")
    print(f"  総ステップ数: {n_total}")
    print(f"  バックトラック: {n_backtrack} ({100*n_backtrack/max(1,n_total):.1f}%)")
    print(f"  通常移動: {n_total - n_backtrack}")


# ─── メイン ──────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="RLDS/TFRecord 変換")
    parser.add_argument("--input_dir", type=str, default="data/rust_dataset",
                        help="generate_dataset.py の出力ディレクトリ")
    parser.add_argument("--output_dir", type=str, default="data/rust_rlds",
                        help="TFRecord の出力ディレクトリ")
    parser.add_argument("--split_ratio", type=float, default=0.9,
                        help="train/val 分割比率 (train 割合)")
    parser.add_argument("--test", action="store_true",
                        help="変換済み TFRecord の検証のみ実行")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.test:
        # 検証モード
        train_path = output_dir / "train.tfrecord.gz"
        val_path = output_dir / "val.tfrecord.gz"
        for path in [train_path, val_path]:
            if path.exists():
                verify_tfrecord(path)
                print_dataset_stats(path)
            else:
                print(f"[test] ファイルが見つかりません: {path}")
        return

    if not TF_AVAILABLE:
        print("[ERROR] TensorFlow が必要です: pip install tensorflow")
        return

    input_dir = Path(args.input_dir)
    episodes = load_episodes(input_dir)

    # train/val 分割
    n_train = int(len(episodes) * args.split_ratio)
    train_episodes = episodes[:n_train]
    val_episodes = episodes[n_train:]

    print(f"[convert_to_rlds] Train: {len(train_episodes)} エピソード, "
          f"Val: {len(val_episodes)} エピソード")

    # TFRecord 書き込み
    train_path = output_dir / "train.tfrecord.gz"
    val_path = output_dir / "val.tfrecord.gz"

    n_train_steps = write_split(train_episodes, train_path, input_dir, "train")
    n_val_steps = write_split(val_episodes, val_path, input_dir, "val")

    # 検証
    print("\n[検証開始]")
    verify_tfrecord(train_path)
    verify_tfrecord(val_path)
    print_dataset_stats(train_path)

    # データセット情報を保存
    info = {
        "n_train_episodes": len(train_episodes),
        "n_val_episodes": len(val_episodes),
        "n_train_steps": n_train_steps,
        "n_val_steps": n_val_steps,
        "image_size": IMAGE_SIZE,
        "action_dim": ACTION_DIM,
        "rust_action_dim": RUST_ACTION_DIM,
        "compression": "GZIP",
        "format": "RLDS-compatible TFRecord",
    }
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n[convert_to_rlds] 完了! → {output_dir}")


if __name__ == "__main__":
    main()
