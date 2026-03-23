"""
アノテーション結果の可視化ツール

episode_XXXX.json を読み込み、ウェイポイントを画像上に描画して保存する。

使用方法:
  python visualize_annotation.py --episode data/auto_raw/episodes/episode_0000.json \
                                 --image crack_generated/04_wood_w2.5.png

  # 全エピソードをまとめて可視化
  python visualize_annotation.py --episode_dir data/auto_raw/episodes \
                                 --image_dir crack_generated
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def draw_annotation(
    img: np.ndarray,
    waypoints: list[tuple[int, int]],
    patch_size: int = 224,
    *,
    line_thickness: int = 2,
    show_index: bool = True,
) -> np.ndarray:
    """
    ROIボックスと経路を画像上に描画する。
    - 各ウェイポイントを中心とした patch_size × patch_size のROI矩形
    - 矢印付き折れ線でトレース方向を表示
    - 始点=緑、終点=赤、中間=シアン
    """
    vis = img.copy()
    n = len(waypoints)
    half = patch_size // 2

    # スケールに応じてフォント・線幅を調整
    h = img.shape[0]
    font_scale = max(0.4, h / 2048)
    arrow_thickness = max(2, h // 1024)
    box_thickness = max(1, h // 2048)
    label_offset = max(6, h // 512)

    # 経路を矢印で描画（ROI中心を結ぶ）
    for i in range(n - 1):
        cv2.arrowedLine(vis, waypoints[i], waypoints[i + 1],
                        (0, 220, 255), arrow_thickness, tipLength=0.1)

    # 各ROIボックスを描画
    for i, (x, y) in enumerate(waypoints):
        if i == 0:
            color = (0, 200, 0)      # 始点: 緑
        elif i == n - 1:
            color = (0, 0, 220)      # 終点: 赤
        else:
            color = (255, 180, 0)    # 中間: シアン

        # ROI矩形
        x0, y0 = x - half, y - half
        x1, y1 = x + half, y + half
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, box_thickness)

        # 中心点
        cv2.circle(vis, (x, y), max(3, h // 512), color, -1)

        # 番号ラベル
        if show_index:
            cv2.putText(vis, str(i + 1), (x0 + label_offset, y0 + label_offset * 3),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (255, 255, 255), max(1, int(font_scale * 2)), cv2.LINE_AA)

    # 凡例
    legend_y = int(h * 0.02)
    legend_x = int(img.shape[1] * 0.01)
    for label, color in [("Start ROI", (0, 200, 0)),
                          ("End ROI",   (0, 0, 220)),
                          ("ROI",       (255, 180, 0))]:
        cv2.rectangle(vis, (legend_x, legend_y - 8),
                      (legend_x + 16, legend_y + 8), color, -1)
        cv2.putText(vis, label, (legend_x + 22, legend_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), max(1, int(font_scale * 2)), cv2.LINE_AA)
        legend_y += int(h * 0.03)

    return vis


def draw_action_vectors(
    img: np.ndarray,
    waypoints: list[tuple[int, int]],
) -> np.ndarray:
    """アクションベクトル（移動方向・距離）をヒートマップ風に可視化する。"""
    vis = img.copy()
    img_h = vis.shape[0]
    font_scale = max(0.35, img_h / 4096)
    arrow_thickness = max(2, img_h // 1024)
    max_dist = 512.0 * (img_h / 512)  # 画像サイズに合わせて距離の上限をスケール

    for i in range(len(waypoints) - 1):
        x0, y0 = waypoints[i]
        x1, y1 = waypoints[i + 1]
        dx, dy = x1 - x0, y1 - y0
        dist = math.sqrt(dx * dx + dy * dy)

        # 距離を色で表現 (短=青, 中=緑, 長=赤)
        t = min(dist / max_dist, 1.0)
        r = int(255 * t)
        b = int(255 * (1 - t))
        color = (b, 100, r)

        # 移動ベクトルを矢印で表示
        cv2.arrowedLine(vis, (x0, y0), (x1, y1), color, arrow_thickness, tipLength=0.1)
        cv2.putText(vis, f"{dist:.0f}px", ((x0 + x1) // 2 + 4, (y0 + y1) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (220, 220, 220), max(1, int(font_scale * 2)), cv2.LINE_AA)

    return vis


def visualize_episode(
    episode_path: str,
    image_path: str,
    output_dir: Path,
) -> None:
    """1エピソードを可視化して保存する。"""
    with open(episode_path) as f:
        episode = json.load(f)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"画像が読み込めません: {image_path}")

    waypoints = [(s["pixel_x"], s["pixel_y"]) for s in episode["steps"]]
    episode_id = episode["episode_id"]
    method = episode.get("annotation_method", "manual")
    patch_size = episode.get("patch_size", 224)
    n_steps = len(waypoints)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── ① ROI可視化 ───────────────────────────────────────────
    vis_waypoints = draw_annotation(img, waypoints, patch_size)

    title = f"Episode {episode_id:04d} | {method} | {n_steps} steps | {Path(image_path).name}"
    cv2.putText(vis_waypoints, title, (6, img.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    out1 = output_dir / f"episode_{episode_id:04d}_waypoints.png"
    cv2.imwrite(str(out1), vis_waypoints)
    print(f"[vis] 保存: {out1}")

    # ── ② アクションベクトル可視化 ───────────────────────────────
    vis_actions = draw_action_vectors(img, waypoints)
    cv2.putText(vis_actions, title + " [actions]", (6, img.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    out2 = output_dir / f"episode_{episode_id:04d}_actions.png"
    cv2.imwrite(str(out2), vis_actions)
    print(f"[vis] 保存: {out2}")

    # ── ③ 横並びサマリ ─────────────────────────────────────────
    combined = np.hstack([vis_waypoints, vis_actions])
    out3 = output_dir / f"episode_{episode_id:04d}_summary.png"
    cv2.imwrite(str(out3), combined)
    print(f"[vis] 保存: {out3}")

    # ── ④ アクション統計 ─────────────────────────────────────
    actions = [s["action_vector"] for s in episode["steps"]]
    dists = [math.sqrt(a[0]**2 + a[1]**2) for a in actions[:-1]]
    if dists:
        print(f"[vis] アクション統計: 平均={sum(dists)/len(dists):.1f}px, "
              f"最大={max(dists):.1f}px, 最小={min(dists):.1f}px")


def main() -> None:
    parser = argparse.ArgumentParser(description="アノテーション結果の可視化ツール")
    parser.add_argument("--episode", help="エピソードJSONのパス（単一）")
    parser.add_argument("--image", help="対応する画像パス（--episode 使用時）")
    parser.add_argument("--episode_dir", help="エピソードJSONのディレクトリ（一括処理）")
    parser.add_argument("--image_dir", help="画像ディレクトリ（--episode_dir 使用時）")
    parser.add_argument("--output_dir", default="data/vis",
                        help="可視化画像の出力先 (デフォルト: data/vis)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.episode and args.image:
        # 単一エピソード
        visualize_episode(args.episode, args.image, output_dir)

    elif args.episode_dir and args.image_dir:
        # ディレクトリ一括処理（source_image フィールドで正確にマッチ）
        ep_dir = Path(args.episode_dir)
        img_dir = Path(args.image_dir)
        episodes = sorted(ep_dir.glob("episode_*.json"))

        # 画像ファイルをファイル名→パスの辞書に
        image_map = {p.name: p for p in img_dir.glob("*.png")}

        print(f"[vis] {len(episodes)} エピソード × {len(image_map)} 画像")

        for ep_path in episodes:
            with open(ep_path) as f:
                ep = json.load(f)
            src = ep.get("source_image", "")
            img_path = image_map.get(src)
            if img_path is None:
                print(f"  [スキップ] {ep_path.name}: 画像 '{src}' が見つかりません")
                continue
            print(f"\n--- {ep_path.name} + {img_path.name} ---")
            try:
                visualize_episode(str(ep_path), str(img_path), output_dir)
            except Exception as e:
                print(f"  [エラー] {e}")
    else:
        parser.error("--episode --image か --episode_dir --image_dir を指定してください。")


if __name__ == "__main__":
    main()
