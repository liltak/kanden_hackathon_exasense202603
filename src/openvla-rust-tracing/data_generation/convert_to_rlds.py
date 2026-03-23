"""
データセット検証スクリプト

generate_crack.py が出力した JSON + PNG データが
training/train.py (CrackTraceDataset) の読み込み形式に
正しく合致しているか確認する。

使い方:
  python convert_to_rlds.py
  python convert_to_rlds.py --data data/auto_raw
  python convert_to_rlds.py --data data/auto_raw --show_stats
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def check_dataset(data_dir: Path, show_stats: bool = False) -> bool:
    """
    train.py (CrackTraceDataset) が期待するデータ形式を検証する。

    期待する構造:
      data_dir/
        episodes/episode_XXXX.json   ← episode_id, steps[], source_image
        patches/episode_XXXX_step_YY.png  ← 224×224 パッチ画像

    各 step に必要なフィールド:
      patch_path    : str  (patches/ 以下の相対パス)
      action_vector : list (最低2要素 = [delta_x, delta_y])
      pixel_x, pixel_y : int (ウェイポイント中心)
    """
    episode_dir = data_dir / "episodes"
    if not episode_dir.exists():
        print(f"[ERROR] エピソードディレクトリが見つかりません: {episode_dir}")
        return False

    episodes = sorted(episode_dir.glob("episode_*.json"))
    if not episodes:
        print(f"[ERROR] エピソード JSON が見つかりません: {episode_dir}")
        return False

    print(f"[check] {len(episodes)} エピソードを検証します: {data_dir}")

    n_steps_total = 0
    n_missing_patch = 0
    n_missing_action = 0
    all_actions: list[list[float]] = []
    errors: list[str] = []

    for ep_path in episodes:
        with open(ep_path, encoding="utf-8") as f:
            ep = json.load(f)

        ep_id = ep.get("episode_id", "?")
        steps = ep.get("steps", [])
        if not steps:
            errors.append(f"episode_{ep_id}: steps が空")
            continue

        for i, step in enumerate(steps):
            # patch_path チェック（最終ステップは None の場合あり）
            patch_path_str = step.get("patch_path")
            if patch_path_str is None:
                # 最終ステップはパッチなし（train.py でスキップ済み）
                continue

            n_steps_total += 1
            patch_path = data_dir / patch_path_str
            if not patch_path.exists():
                n_missing_patch += 1
                errors.append(f"episode_{ep_id} step_{i}: パッチ画像なし → {patch_path}")

            # action_vector チェック
            action_vector = step.get("action_vector")
            if action_vector is None or len(action_vector) < 2:
                n_missing_action += 1
                errors.append(f"episode_{ep_id} step_{i}: action_vector が不正 → {action_vector}")
            else:
                all_actions.append(action_vector[:2])

            # pixel_x, pixel_y チェック
            if "pixel_x" not in step or "pixel_y" not in step:
                errors.append(f"episode_{ep_id} step_{i}: pixel_x/y が欠落")

    # ─── 結果表示 ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"エピソード数  : {len(episodes)}")
    print(f"ステップ数    : {n_steps_total}  (train.py に渡るサンプル数)")
    print(f"欠損パッチ    : {n_missing_patch}")
    print(f"欠損アクション: {n_missing_action}")

    if all_actions and show_stats:
        arr = np.array(all_actions, dtype=np.float32)
        dx, dy = arr[:, 0], arr[:, 1]
        dists = np.sqrt(dx**2 + dy**2)
        print(f"\n── アクション統計 (delta_x, delta_y) ──────────────")
        print(f"  delta_x : min={dx.min():.1f}  max={dx.max():.1f}  mean={dx.mean():.1f}")
        print(f"  delta_y : min={dy.min():.1f}  max={dy.max():.1f}  mean={dy.mean():.1f}")
        print(f"  距離    : min={dists.min():.1f}  max={dists.max():.1f}  mean={dists.mean():.1f} px")
        print(f"\n  → ActionTokenizer の min/max 参考値:")
        print(f"       action_min = [{dx.min():.1f}, {dy.min():.1f}]")
        print(f"       action_max = [{dx.max():.1f}, {dy.max():.1f}]")

    if errors:
        print(f"\n── エラー ({len(errors)} 件) ─────────────────────────────")
        for err in errors[:20]:
            print(f"  [!] {err}")
        if len(errors) > 20:
            print(f"  ... 他 {len(errors)-20} 件")
        print(f"{'='*60}")
        return False

    print(f"\n[OK] データセットは train.py (CrackTraceDataset) と互換性があります。")
    print(f"{'='*60}\n")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="train.py 向けデータセット検証")
    parser.add_argument(
        "--data", type=str, default="data/auto_raw",
        help="検証するデータディレクトリ (default: data/auto_raw)",
    )
    parser.add_argument(
        "--show_stats", action="store_true",
        help="アクション統計を表示する",
    )
    args = parser.parse_args()

    data_dir = Path(args.data)
    ok = check_dataset(data_dir, show_stats=args.show_stats)
    exit(0 if ok else 1)


if __name__ == "__main__":
    main()
