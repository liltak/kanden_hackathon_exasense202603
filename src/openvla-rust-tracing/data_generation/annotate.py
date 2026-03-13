"""
手動アノテーションツール (フリークリック・ピクセル移動量)

画像上の任意の点をクリックして訪問順序を指定し、
ピクセル単位の移動量をアクションとして記録する。

アクション形式: [Δx, Δy, 0, 0, 0, 0, 0]
  Δx = x方向ピクセル移動量 (右が正)
  Δy = y方向ピクセル移動量 (下が正)
  最終ステップ (終端) = [0, 0, 0, 0, 0, 0, 0]

観測画像: クリック点を中心とした patch_size × patch_size のクロップ

操作方法:
  左クリック  : 点を訪問順に追加
  右クリック  : 最も近い点を削除
  z           : 直前のクリックを取り消す
  s           : JSON・パッチ画像を保存
  q           : 終了（保存なし）

使用方法:
  python annotate.py --image path/to/rust_image.jpg --output_dir data/raw

複数エピソードを作成する場合:
  python annotate.py --image rust1.jpg --output_dir data/raw --episode_id 0
  python annotate.py --image rust2.jpg --output_dir data/raw --episode_id 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# ─── 定数 ─────────────────────────────────────────────────────────────────
ACTION_DIM = 7
INSTRUCTION = "Follow the rust trace. Navigate to continue tracking the corrosion path."
ROI_INSET_SIZE = 160   # 右下インセットの表示サイズ (px)


def draw_canvas(
    base_img: np.ndarray,
    visit_sequence: list[tuple[int, int]],
    mouse_pos: tuple[int, int],
    patch_size: int,
) -> np.ndarray:
    """訪問済み点・ROI枠・プレビューインセットを描画した表示用画像を返す。"""
    vis = base_img.copy()
    h, w = vis.shape[:2]
    mx, my = mouse_pos
    half = patch_size // 2

    # ── カーソル位置の ROI 枠 ──────────────────────────────────────────
    rx0, ry0 = mx - half, my - half
    rx1, ry1 = mx + half, my + half
    cv2.rectangle(vis, (rx0, ry0), (rx1, ry1), (0, 200, 255), 1)

    # ── 経路を矢印で描画 ───────────────────────────────────────────────
    for i in range(1, len(visit_sequence)):
        cv2.arrowedLine(vis, visit_sequence[i - 1], visit_sequence[i],
                        (0, 255, 255), 2, tipLength=0.2)

    # ── 各点を描画 ────────────────────────────────────────────────────
    for idx, (x, y) in enumerate(visit_sequence):
        cv2.circle(vis, (x, y), 8, (0, 180, 0), -1)
        cv2.circle(vis, (x, y), 8, (255, 255, 255), 1)
        cv2.putText(vis, str(idx + 1), (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ── 右下インセット: カーソル位置のクロッププレビュー ───────────────
    crop = crop_patch(base_img, mx, my, patch_size)
    inset = cv2.resize(crop, (ROI_INSET_SIZE, ROI_INSET_SIZE))
    ix0 = w - ROI_INSET_SIZE - 4
    iy0 = h - ROI_INSET_SIZE - 4
    vis[iy0:iy0 + ROI_INSET_SIZE, ix0:ix0 + ROI_INSET_SIZE] = inset
    cv2.rectangle(vis, (ix0 - 1, iy0 - 1),
                  (ix0 + ROI_INSET_SIZE, iy0 + ROI_INSET_SIZE), (0, 200, 255), 1)
    cv2.putText(vis, "ROI", (ix0 + 4, iy0 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

    return vis


def crop_patch(
    img: np.ndarray,
    x: int,
    y: int,
    patch_size: int,
) -> np.ndarray:
    """(x, y) を中心とした patch_size × patch_size のクロップを返す。端はゼロパディング。"""
    h, w = img.shape[:2]
    half = patch_size // 2

    canvas = np.zeros((h + patch_size, w + patch_size, 3), dtype=np.uint8)
    canvas[half:half + h, half:half + w] = img

    cx = x + half
    cy = y + half
    patch = canvas[cy - half:cy + half, cx - half:cx + half]
    return cv2.resize(patch, (patch_size, patch_size))


def save_episode(
    base_img: np.ndarray,
    visit_sequence: list[tuple[int, int]],
    output_dir: Path,
    episode_id: int,
    source_image_name: str,
    patch_size: int,
    step_offset: int,
) -> int:
    """エピソードをJSON・パッチ画像として保存し、次の step_offset を返す。"""
    steps_dir = output_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)

    steps = []

    for seq_idx, (x, y) in enumerate(visit_sequence):
        step_id = step_offset + seq_idx

        patch = crop_patch(base_img, x, y, patch_size)
        img_filename = f"step_{step_id:06d}.png"
        cv2.imwrite(str(steps_dir / img_filename), patch)

        if seq_idx < len(visit_sequence) - 1:
            nx, ny = visit_sequence[seq_idx + 1]
            action_vec = [float(nx - x), float(ny - y), 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            action_vec = [0.0] * ACTION_DIM

        steps.append({
            "step_id": step_id,
            "episode_id": episode_id,
            "image_path": f"steps/{img_filename}",
            "instruction": INSTRUCTION,
            "action_vector": action_vec,
            "pixel_x": x,
            "pixel_y": y,
            "is_first": seq_idx == 0,
            "is_last": seq_idx == len(visit_sequence) - 1,
        })

    episode = {
        "episode_id": episode_id,
        "source_image": source_image_name,
        "patch_size": patch_size,
        "steps": steps,
    }

    episode_dir = output_dir / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)
    episode_path = episode_dir / f"episode_{episode_id:04d}.json"
    with open(episode_path, "w") as f:
        json.dump(episode, f, indent=2, ensure_ascii=False)

    print(f"[annotate] 保存完了: {episode_path}  ({len(steps)} steps)")
    return step_offset + len(steps)


def run(args: argparse.Namespace) -> None:
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"画像が読み込めません: {args.image}")

    h, w = img.shape[:2]
    visit_sequence: list[tuple[int, int]] = []
    mouse_pos = [w // 2, h // 2]   # [x, y] ミュータブルリストで共有
    window_name = "Annotate (L=add, R=delete, z=undo, s=save, q=quit)"

    def on_mouse(event: int, x: int, y: int, flags: int, param) -> None:
        mouse_pos[0], mouse_pos[1] = x, y

        if event == cv2.EVENT_LBUTTONDOWN:
            if 0 <= x < w and 0 <= y < h:
                visit_sequence.append((x, y))
                if len(visit_sequence) >= 2:
                    px, py = visit_sequence[-2]
                    dx, dy = x - px, y - py
                    dist = int((dx ** 2 + dy ** 2) ** 0.5)
                    print(f"  [{len(visit_sequence)}] ({x}, {y})  Δx={dx:+d}, Δy={dy:+d}  距離={dist}px")
                else:
                    print(f"  [{len(visit_sequence)}] ({x}, {y})  (始点)")

        elif event == cv2.EVENT_RBUTTONDOWN:
            if not visit_sequence:
                return
            # 最も近い点を削除
            dists = [(i, (px - x) ** 2 + (py - y) ** 2)
                     for i, (px, py) in enumerate(visit_sequence)]
            nearest_idx, _ = min(dists, key=lambda t: t[1])
            removed = visit_sequence.pop(nearest_idx)
            print(f"  [右クリック削除] 点{nearest_idx + 1}: {removed}")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(w, 1400), min(h, 900))
    cv2.setMouseCallback(window_name, on_mouse)

    print(f"[annotate] 画像: {args.image}  ({w}×{h}px)  patch_size={args.patch_size}")
    print("  左クリック: 点追加  /  右クリック: 近い点を削除  /  z: 最後を取り消し  /  s: 保存  /  q: 終了")

    output_dir = Path(args.output_dir)
    step_offset = args.step_offset

    while True:
        canvas = draw_canvas(img, visit_sequence, tuple(mouse_pos), args.patch_size)
        cv2.setWindowTitle(window_name, f"{window_name} | 点数: {len(visit_sequence)}")
        cv2.imshow(window_name, canvas)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            print("[annotate] 保存せずに終了します。")
            break
        elif key == ord("z"):
            if visit_sequence:
                removed = visit_sequence.pop()
                print(f"  取り消し: {removed}")
        elif key == ord("s"):
            if len(visit_sequence) < 2:
                print("[annotate] 2点以上クリックしてから保存してください。")
                continue
            step_offset = save_episode(
                img, visit_sequence, output_dir,
                args.episode_id, Path(args.image).name,
                args.patch_size, step_offset,
            )
            break

    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="手動アノテーションツール（フリークリック）")
    parser.add_argument("--image", required=True, help="入力画像パス")
    parser.add_argument("--output_dir", default="data/raw",
                        help="出力ディレクトリ (デフォルト: data/raw)")
    parser.add_argument("--episode_id", type=int, default=0,
                        help="エピソードID")
    parser.add_argument("--patch_size", type=int, default=224,
                        help="観測クロップサイズ px (デフォルト: 224)")
    parser.add_argument("--step_offset", type=int, default=0,
                        help="step_id の開始番号（複数エピソード連番用）")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
