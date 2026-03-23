"""
自動アノテーションツール (クラック線の自動トラッキング)

クラック画像を解析し、暗い線を自動検出・骨格化してパスを追跡する。
annotate.py と同一の JSON・パッチ画像形式で出力する。

アルゴリズム:
  1. グレースケール化 → 閾値で二値化（暗いピクセル=クラック）
  2. Zhang-Suen 細線化で1px幅の骨格を取得
  3. 端点（隣接1px）から DFS でパスを追跡
  4. 一定間隔でサブサンプリングしてウェイポイントを生成
  5. annotate.py と同じ形式で保存

使用方法:
  # 生成済み画像を自動アノテーション
  python auto_annotate.py --image crack_generated/00_concrete_w2.5.png --output_dir data/raw

  # generate_crack.py の既知パスから生成（最高精度）
  python auto_annotate.py --image crack_generated/00_concrete_w2.5.png \\
                          --output_dir data/raw --use_generated_path --seed 0

  # 閾値・ウェイポイント数を調整
  python auto_annotate.py --image path/to/crack.png --threshold 100 --n_waypoints 30
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


# ─── 定数 ─────────────────────────────────────────────────────────────────
ACTION_DIM = 7
INSTRUCTION = "Follow the rust trace. Navigate to continue tracking the corrosion path."


# ─── Zhang-Suen 細線化 ────────────────────────────────────────────────────

def _zs_iteration(binary: np.ndarray, step: int) -> np.ndarray:
    """Zhang-Suen アルゴリズムの1イテレーション（step=0 or 1）。"""
    h, w = binary.shape
    marked = np.zeros_like(binary)

    # パディングして境界処理を簡略化
    pad = np.pad(binary, 1, constant_values=0)

    for y in range(1, h + 1):
        for x in range(1, w + 1):
            if pad[y, x] == 0:
                continue
            # 8近傍を時計回りに取得: p2..p9
            p2 = pad[y - 1, x]
            p3 = pad[y - 1, x + 1]
            p4 = pad[y,     x + 1]
            p5 = pad[y + 1, x + 1]
            p6 = pad[y + 1, x]
            p7 = pad[y + 1, x - 1]
            p8 = pad[y,     x - 1]
            p9 = pad[y - 1, x - 1]

            neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
            B = sum(neighbors)               # 前景近傍数
            if not (2 <= B <= 6):
                continue

            # 0→1 の遷移回数
            ring = neighbors + [neighbors[0]]
            A = sum(1 for i in range(8) if ring[i] == 0 and ring[i + 1] == 1)
            if A != 1:
                continue

            if step == 0:
                if p2 * p4 * p6 != 0:
                    continue
                if p4 * p6 * p8 != 0:
                    continue
            else:
                if p2 * p4 * p8 != 0:
                    continue
                if p2 * p6 * p8 != 0:
                    continue

            marked[y - 1, x - 1] = 1

    return binary & ~marked


def zhang_suen_thin(binary: np.ndarray, max_iter: int = 300) -> np.ndarray:
    """
    Zhang-Suen 細線化。
    binary: 前景=1, 背景=0 の uint8 配列
    """
    skel = binary.copy().astype(np.uint8)
    for _ in range(max_iter):
        prev = skel.copy()
        skel = _zs_iteration(skel, 0)
        skel = _zs_iteration(skel, 1)
        if np.array_equal(skel, prev):
            break
    return skel


# ─── スケルトン追跡 ────────────────────────────────────────────────────────

def _neighbors8(y: int, x: int, h: int, w: int):
    """8近傍座標を返す（範囲内のみ）。"""
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                yield ny, nx


def find_endpoints(skel: np.ndarray) -> list[tuple[int, int]]:
    """骨格の端点（隣接前景ピクセルが1個）を返す。"""
    h, w = skel.shape
    pts = []
    ys, xs = np.where(skel > 0)
    for y, x in zip(ys.tolist(), xs.tolist()):
        n = sum(1 for ny, nx in _neighbors8(y, x, h, w) if skel[ny, nx] > 0)
        if n == 1:
            pts.append((y, x))
    return pts


def trace_skeleton(skel: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    """
    start から貪欲にスケルトンをたどり、順序付きピクセルリストを返す。
    一本線前提（分岐なし）なので常に未訪問の隣接ピクセルへ進む。
    (y, x) 形式。
    """
    h, w = skel.shape
    visited = np.zeros_like(skel, dtype=bool)
    path: list[tuple[int, int]] = [start]
    visited[start[0], start[1]] = True

    cy, cx = start
    while True:
        next_pts = [
            (ny, nx)
            for ny, nx in _neighbors8(cy, cx, h, w)
            if skel[ny, nx] > 0 and not visited[ny, nx]
        ]
        if not next_pts:
            break
        # 複数候補がある場合は直前の移動方向に最も近いものを選ぶ
        if len(path) >= 2:
            dy_prev = cy - path[-2][0]
            dx_prev = cx - path[-2][1]
            def continuity(pt):
                dy = pt[0] - cy
                dx = pt[1] - cx
                return -(dy * dy_prev + dx * dx_prev)  # 内積が大きい＝同方向
            next_pts.sort(key=continuity)
        ny, nx = next_pts[0]
        visited[ny, nx] = True
        path.append((ny, nx))
        cy, cx = ny, nx

    return path


def subsample_path(
    path: list[tuple[int, int]],
    n_waypoints: int,
) -> list[tuple[int, int]]:
    """パスを等間隔に n_waypoints 点にサブサンプリングする。"""
    if len(path) <= n_waypoints:
        return path

    # 累積弧長を計算して均等サンプリング
    arc = [0.0]
    for i in range(1, len(path)):
        dy = path[i][0] - path[i - 1][0]
        dx = path[i][1] - path[i - 1][1]
        arc.append(arc[-1] + math.sqrt(dy * dy + dx * dx))

    total = arc[-1]
    targets = [total * i / (n_waypoints - 1) for i in range(n_waypoints)]

    result = []
    j = 0
    for t in targets:
        while j < len(arc) - 1 and arc[j + 1] < t:
            j += 1
        result.append(path[j])
    return result


# ─── クラック検出 ─────────────────────────────────────────────────────────

def detect_crack_path(
    image_path: str,
    threshold: int = 80,
    n_waypoints: int = 20,
    min_component_px: int = 50,
) -> list[tuple[int, int]]:
    """
    画像からクラックを自動検出し、ウェイポイント列を返す。
    座標は (x, y) 形式（annotate.py に合わせる）。

    Parameters
    ----------
    image_path : str
        入力画像パス（RGB/グレースケール PNG/JPG）
    threshold : int
        二値化閾値（0〜255）。値以下のピクセルをクラックとみなす
    n_waypoints : int
        出力するウェイポイント数
    min_component_px : int
        ノイズ除去のための最小連結成分サイズ（px）

    Returns
    -------
    list of (x, y) tuples
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"画像が読み込めません: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── 1. 閾値で二値化（暗いピクセル = クラック）─────────────────────
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # ── 2. ノイズ除去（小さい連結成分を除外）───────────────────────────
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean = np.zeros_like(binary)
    for label in range(1, n_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_component_px:
            clean[labels == label] = 1

    if clean.sum() == 0:
        raise ValueError(
            f"クラックが検出されませんでした。threshold={threshold} を調整してください。"
        )

    # ── 3. Zhang-Suen 細線化 ─────────────────────────────────────────
    print(f"[auto_annotate] 細線化中... (前景ピクセル数: {clean.sum()})")
    skel = zhang_suen_thin(clean)
    print(f"[auto_annotate] 骨格ピクセル数: {skel.sum()}")

    # ── 4. 端点を探してトレース ──────────────────────────────────────
    endpoints = find_endpoints(skel)
    print(f"[auto_annotate] 端点数: {len(endpoints)}")

    if len(endpoints) == 0:
        # 端点なし（閉じたループ） → 最初の前景ピクセルを起点に
        ys, xs = np.where(skel > 0)
        start = (int(ys[0]), int(xs[0]))
    else:
        # 最も離れた端点ペアの片方を起点に
        if len(endpoints) >= 2:
            max_dist = -1
            best = endpoints[0]
            for i in range(len(endpoints)):
                for j in range(i + 1, len(endpoints)):
                    dy = endpoints[i][0] - endpoints[j][0]
                    dx = endpoints[i][1] - endpoints[j][1]
                    d = dy * dy + dx * dx
                    if d > max_dist:
                        max_dist = d
                        best = endpoints[i]
            start = best
        else:
            start = endpoints[0]

    path_yx = trace_skeleton(skel, start)
    print(f"[auto_annotate] 追跡パス長: {len(path_yx)} px")

    # ── 5. サブサンプリング → (x, y) に変換 ──────────────────────────
    waypoints_yx = subsample_path(path_yx, n_waypoints)
    waypoints_xy = [(x, y) for (y, x) in waypoints_yx]

    return waypoints_xy


# ─── generate_crack.py の既知パスを使う場合 ──────────────────────────────

def get_generated_path(seed: int, n_waypoints: int = 20) -> list[tuple[int, int]]:
    """
    generate_crack.py の generate_crack_path() を呼び出し、
    クラック生成時の正解パスを直接取得する（最高精度）。
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from generate_crack import generate_crack_path  # noqa: PLC0415

    path_f = generate_crack_path(seed=seed, max_curve_deg=45)
    # float座標を int に変換
    path_i = [(int(round(x)), int(round(y))) for (x, y) in path_f]

    # 512×512 範囲内に収まる点のみ
    path_i = [(x, y) for (x, y) in path_i if 0 <= x < 512 and 0 <= y < 512]

    return subsample_path(path_i, n_waypoints)


# ─── annotate.py と同一形式で保存 ─────────────────────────────────────────

def crop_patch(img: np.ndarray, x: int, y: int, patch_size: int) -> np.ndarray:
    """(x, y) を中心とした patch_size × patch_size のクロップ（ゼロパディング）。"""
    h, w = img.shape[:2]
    half = patch_size // 2
    canvas = np.zeros((h + patch_size, w + patch_size, 3), dtype=np.uint8)
    canvas[half:half + h, half:half + w] = img
    cx, cy = x + half, y + half
    return canvas[cy - half:cy + half, cx - half:cx + half]


def save_episode(
    image_path: str,
    visit_sequence: list[tuple[int, int]],
    output_dir: Path,
    episode_id: int,
    patch_size: int,
    step_offset: int,
) -> int:
    """annotate.py と同一形式でエピソードを保存し、次の step_offset を返す。"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"画像が読み込めません: {image_path}")

    steps_dir = output_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    episode_dir = output_dir / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)

    steps = []
    for seq_idx, (x, y) in enumerate(visit_sequence):
        step_id = step_offset + seq_idx

        patch = crop_patch(img, x, y, patch_size)
        img_filename = f"step_{step_id:06d}.png"
        cv2.imwrite(str(steps_dir / img_filename), patch)

        if seq_idx < len(visit_sequence) - 1:
            nx, ny = visit_sequence[seq_idx + 1]
            action_vec = [float(nx - x), float(ny - y), 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            action_vec = [0.0] * ACTION_DIM

        steps.append({
            "step_id":      step_id,
            "episode_id":   episode_id,
            "image_path":   f"steps/{img_filename}",
            "instruction":  INSTRUCTION,
            "action_vector": action_vec,
            "pixel_x":      x,
            "pixel_y":      y,
            "is_first":     seq_idx == 0,
            "is_last":      seq_idx == len(visit_sequence) - 1,
            "annotation_method": "auto",   # 手動と区別するフィールド
        })

    episode = {
        "episode_id":   episode_id,
        "source_image": Path(image_path).name,
        "patch_size":   patch_size,
        "annotation_method": "auto",
        "steps":        steps,
    }

    episode_path = episode_dir / f"episode_{episode_id:04d}.json"
    with open(episode_path, "w") as f:
        json.dump(episode, f, indent=2, ensure_ascii=False)

    print(f"[auto_annotate] 保存完了: {episode_path}  ({len(steps)} steps)")
    return step_offset + len(steps)


# ─── CLI ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="自動クラックトラッキング・アノテーションツール"
    )
    parser.add_argument("--image", required=True,
                        help="入力画像パス")
    parser.add_argument("--output_dir", default="data/raw",
                        help="出力ディレクトリ (デフォルト: data/raw)")
    parser.add_argument("--episode_id", type=int, default=0,
                        help="エピソードID")
    parser.add_argument("--patch_size", type=int, default=224,
                        help="観測クロップサイズ px (デフォルト: 224)")
    parser.add_argument("--step_offset", type=int, default=0,
                        help="step_id の開始番号（複数エピソード連番用）")
    parser.add_argument("--threshold", type=int, default=80,
                        help="クラック検出の二値化閾値 0-255 (デフォルト: 80)")
    parser.add_argument("--n_waypoints", type=int, default=20,
                        help="生成するウェイポイント数 (デフォルト: 20)")
    parser.add_argument("--min_component_px", type=int, default=50,
                        help="ノイズ除去の最小連結成分サイズ (デフォルト: 50)")
    parser.add_argument("--use_generated_path", action="store_true",
                        help="generate_crack.py の既知パスを使う（生成画像専用・最高精度）")
    parser.add_argument("--seed", type=int, default=0,
                        help="--use_generated_path 使用時のシード値")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.use_generated_path:
        print(f"[auto_annotate] モード: 既知パス使用 (seed={args.seed})")
        waypoints = get_generated_path(seed=args.seed, n_waypoints=args.n_waypoints)
    else:
        print(f"[auto_annotate] モード: 画像解析 (threshold={args.threshold})")
        waypoints = detect_crack_path(
            args.image,
            threshold=args.threshold,
            n_waypoints=args.n_waypoints,
            min_component_px=args.min_component_px,
        )

    print(f"[auto_annotate] ウェイポイント数: {len(waypoints)}")
    for i, (x, y) in enumerate(waypoints):
        if i == 0:
            print(f"  [{i+1}] ({x}, {y})  (始点)")
        elif i < len(waypoints) - 1:
            px, py = waypoints[i - 1]
            dx, dy = x - px, y - py
            dist = int(math.sqrt(dx*dx + dy*dy))
            print(f"  [{i+1}] ({x}, {y})  Δx={dx:+d}, Δy={dy:+d}  距離={dist}px")
        else:
            px, py = waypoints[i - 1]
            dx, dy = x - px, y - py
            dist = int(math.sqrt(dx*dx + dy*dy))
            print(f"  [{i+1}] ({x}, {y})  Δx={dx:+d}, Δy={dy:+d}  距離={dist}px  (終点)")

    save_episode(
        image_path=args.image,
        visit_sequence=waypoints,
        output_dir=output_dir,
        episode_id=args.episode_id,
        patch_size=args.patch_size,
        step_offset=args.step_offset,
    )


if __name__ == "__main__":
    main()
