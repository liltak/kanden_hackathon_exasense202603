"""
サビ線合成 + DFS探索 → JSON + 画像ファイル出力

【実行環境】Mac ローカル / H100 どちらでも動作
  依存: opencv-python, numpy のみ (TensorFlow 不要)

1エピソード = 1連結成分 (サビをDFSで辿りきる)

各ステップ:
  image:   現在パッチ (224×224 RGB) → JPEG保存
  action:  [Δcol, Δrow, 0, 0, 0, 0, 0]  Δcol=列変化, Δrow=行変化 (値: -1/0/1)

出力:
  output_dir/episodes/episode_XXXX.json
  output_dir/steps/step_XXXXXX.jpg
  output_dir/dataset_info.json

TFRecord変換は convert_to_rlds.py (H100) で行う。

使用方法:
  python generate_dataset.py \
    --output_dir data/raw \
    --n_source_images 50
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


# ─── 定数 ─────────────────────────────────────────────────────────────────
PATCH_SIZE = 224
ACTION_DIM = 7
INSTRUCTION = "Follow the rust trace. Navigate to continue tracking the corrosion path."

# 8近傍のデルタ (row_delta, col_delta)
NEIGHBOR_DELTAS: list[tuple[int, int]] = [
    (-1,  0), ( 1,  0), ( 0, -1), ( 0,  1),
    (-1, -1), (-1,  1), ( 1, -1), ( 1,  1),
]


# ─── テクスチャ生成 ───────────────────────────────────────────────────────
def generate_concrete_texture(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """コンクリート調のグレーテクスチャを生成する。"""
    base = rng.integers(100, 160, (height, width), dtype=np.uint8)
    noise = rng.integers(0, 30, (height, width), dtype=np.uint8)
    img = cv2.add(base, noise)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for _ in range(rng.integers(3, 8)):
        pt1 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        pt2 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        cv2.line(img, pt1, pt2, (70, 70, 70), int(rng.integers(1, 3)))
    return img


def generate_steel_texture(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """鉄板調のテクスチャを生成する。"""
    base = rng.integers(60, 110, (height, width), dtype=np.uint8)
    for _ in range(rng.integers(5, 20)):
        row = int(rng.integers(0, height))
        val = int(rng.integers(80, 130))
        base[row, :] = np.clip(base[row, :].astype(int) + val - 90, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    img[:, :, 0] = np.clip(img[:, :, 0].astype(int) + 10, 0, 255).astype(np.uint8)
    return img


def generate_base_texture(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """コンクリートと鉄板をランダムに選択して返す。"""
    if rng.random() < 0.5:
        return generate_concrete_texture(height, width, rng)
    return generate_steel_texture(height, width, rng)


# ─── サビ線合成 ────────────────────────────────────────────────────────────
def _rust_color(rng: np.random.Generator) -> tuple[int, int, int]:
    """BGR サビ色 (茶色〜赤茶色) を返す。"""
    r = int(rng.integers(140, 200))
    g = int(rng.integers(60, 100))
    b = int(rng.integers(20, 60))
    return (b, g, r)


def _draw_rust_stroke(
    mask: np.ndarray,
    img: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    rng: np.random.Generator,
) -> None:
    """start→end 間にランダムなサビストロークを描画する。"""
    thickness = int(rng.integers(2, 7))
    color = _rust_color(rng)
    mid_x = (start[0] + end[0]) // 2 + int(rng.integers(-20, 20))
    mid_y = (start[1] + end[1]) // 2 + int(rng.integers(-20, 20))
    t_vals = np.linspace(0, 1, 20)
    curve = np.array([
        (1 - t) ** 2 * np.array(start) + 2 * (1 - t) * t * np.array([mid_x, mid_y]) + t ** 2 * np.array(end)
        for t in t_vals
    ], dtype=np.int32)
    cv2.polylines(img, [curve], False, color, thickness)
    cv2.polylines(mask, [curve], False, 255, thickness + 2)


def synthesize_rust_lines(
    img: np.ndarray,
    rng: np.random.Generator,
    n_components: int = 1,
    branch_prob: float = 0.5,
    n_strokes: int = 30,
    stroke_length: int = 200,
) -> np.ndarray:
    """画像上にサビ線を合成し、サビマスク (0/255) を返す。"""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for _ in range(n_components):
        sx, sy = int(rng.integers(20, w - 20)), int(rng.integers(20, h - 20))
        queue = [(sx, sy)]

        for _ in range(rng.integers(n_strokes // 2, n_strokes)):
            if not queue:
                break
            cx, cy = queue.pop(rng.integers(0, len(queue)))
            angle = rng.uniform(0, 2 * math.pi)
            length = int(rng.integers(stroke_length // 2, stroke_length))
            nx = int(np.clip(cx + length * math.cos(angle), 10, w - 10))
            ny = int(np.clip(cy + length * math.sin(angle), 10, h - 10))

            if rng.random() > 0.15:
                _draw_rust_stroke(mask, img, (cx, cy), (nx, ny), rng)

            queue.append((nx, ny))

            if rng.random() < branch_prob:
                angle2 = angle + rng.uniform(math.pi / 6, math.pi / 3)
                length2 = int(rng.integers(stroke_length // 4, stroke_length // 2))
                bx = int(np.clip(nx + length2 * math.cos(angle2), 10, w - 10))
                by = int(np.clip(ny + length2 * math.sin(angle2), 10, h - 10))
                if rng.random() > 0.15:
                    _draw_rust_stroke(mask, img, (nx, ny), (bx, by), rng)
                queue.append((bx, by))

    return mask


# ─── ドメインランダマイゼーション ───────────────────────────────────────
def apply_domain_randomization(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """ノイズ・ブラー・色ムラを適用する。"""
    aug = img.copy()
    noise = rng.normal(0, rng.uniform(3, 12), aug.shape).astype(np.int16)
    aug = np.clip(aug.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    ksize = int(rng.choice([3, 5, 7]))
    aug = cv2.GaussianBlur(aug, (ksize, ksize), 0)
    for c in range(3):
        scale = rng.uniform(0.85, 1.15)
        aug[:, :, c] = np.clip(aug[:, :, c].astype(float) * scale, 0, 255).astype(np.uint8)
    n_patches = int(rng.integers(0, 5))
    h, w = aug.shape[:2]
    for _ in range(n_patches):
        px, py = int(rng.integers(0, w)), int(rng.integers(0, h))
        pr = int(rng.integers(10, 50))
        brightness = int(rng.integers(-30, 30))
        cv2.circle(aug, (px, py), pr, (brightness, brightness, brightness), -1)
        aug = np.clip(aug.astype(np.int16), 0, 255).astype(np.uint8)
    return aug


# ─── グリッド分割 ──────────────────────────────────────────────────────────
def split_into_patches(
    img: np.ndarray,
    mask: np.ndarray,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """画像をパッチサイズで格子状に分割する。"""
    h, w = img.shape[:2]
    rows = max(1, h // patch_size)
    cols = max(1, w // patch_size)

    patches = np.zeros((rows, cols, patch_size, patch_size, 3), dtype=np.uint8)
    patch_masks = np.zeros((rows, cols, patch_size, patch_size), dtype=np.uint8)

    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * patch_size, (r + 1) * patch_size
            x0, x1 = c * patch_size, (c + 1) * patch_size
            patches[r, c] = cv2.resize(img[y0:y1, x0:x1], (patch_size, patch_size))
            patch_masks[r, c] = cv2.resize(mask[y0:y1, x0:x1], (patch_size, patch_size))

    return patches, patch_masks, rows, cols


# ─── 連結成分 ──────────────────────────────────────────────────────────────
def _get_connected_components(
    patch_has_rust: np.ndarray,
) -> tuple[np.ndarray, int]:
    """サビを持つパッチの連結成分ラベル (8連結) を返す。"""
    binary = (patch_has_rust > 0).astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(binary, connectivity=8)
    return labels, n_labels - 1  # 背景(0)を除く


# ─── DFS探索 → ステップリスト ─────────────────────────────────────────────
def dfs_component(
    start_r: int,
    start_c: int,
    component_id: int,
    labels: np.ndarray,
    patches: np.ndarray,
    rng: np.random.Generator,
) -> list[dict]:
    """
    1連結成分をDFS探索し、ステップリストを返す。

    各ステップ:
      image:  現在位置のパッチ画像 (RGB, 224×224)
      action: [Δcol, Δrow, 0, 0, 0, 0, 0]
    """
    rows, cols = labels.shape
    steps = []
    stack = [(start_r, start_c)]
    visited: set[tuple[int, int]] = {(start_r, start_c)}

    while stack:
        r, c = stack[-1]

        neighbors = [
            (r + dr, c + dc)
            for dr, dc in NEIGHBOR_DELTAS
            if (0 <= r + dr < rows
                and 0 <= c + dc < cols
                and labels[r + dr, c + dc] == component_id
                and (r + dr, c + dc) not in visited)
        ]

        if neighbors:
            idx = int(rng.integers(0, len(neighbors)))
            nr, nc = neighbors[idx]
            action = [float(nc - c), float(nr - r), 0.0, 0.0, 0.0, 0.0, 0.0]
            image_rgb = cv2.cvtColor(patches[r, c], cv2.COLOR_BGR2RGB)
            steps.append({"image": image_rgb, "action": action})
            visited.add((nr, nc))
            stack.append((nr, nc))
        else:
            stack.pop()
            if stack:
                pr, pc = stack[-1]
                action = [float(pc - c), float(pr - r), 0.0, 0.0, 0.0, 0.0, 0.0]
                image_rgb = cv2.cvtColor(patches[r, c], cv2.COLOR_BGR2RGB)
                steps.append({"image": image_rgb, "action": action})

    return steps


# ─── 画像1枚からエピソード群を生成 ───────────────────────────────────────
def generate_episodes_from_image(
    rng: np.random.Generator,
    image_size: tuple[int, int] = (1120, 1120),
    n_rust_components: int = 1,
    n_strokes: int = 30,
    stroke_length: int = 200,
) -> tuple[list[list[dict]], np.ndarray]:
    """
    特大画像1枚を生成し、各連結成分を1エピソードとして返す。
    Returns: (episodes, full_image_bgr)
      episodes: list of episodes (各エピソードはステップのリスト)
      full_image_bgr: 全体画像 (annotate.py 用)
    """
    h, w = image_size
    img = generate_base_texture(h, w, rng)
    img = apply_domain_randomization(img, rng)
    rust_mask = synthesize_rust_lines(
        img, rng,
        n_components=n_rust_components,
        n_strokes=n_strokes,
        stroke_length=stroke_length,
    )
    img = apply_domain_randomization(img, rng)

    patches, patch_masks, rows, cols = split_into_patches(img, rust_mask, PATCH_SIZE)
    patch_has_rust = (patch_masks.sum(axis=(2, 3)) > 0).astype(np.uint8)
    labels, n_labels = _get_connected_components(patch_has_rust)

    episodes = []
    for comp_id in range(1, n_labels + 1):
        comp_positions = np.argwhere(labels == comp_id)
        if len(comp_positions) == 0:
            continue
        start_r, start_c = int(comp_positions[0][0]), int(comp_positions[0][1])
        steps = dfs_component(start_r, start_c, comp_id, labels, patches, rng)
        if len(steps) >= 2:
            episodes.append(steps)

    return episodes, img


# ─── JSON + 画像ファイル保存 ──────────────────────────────────────────────
def write_raw_episodes(
    episodes: list[list[dict]],
    output_dir: Path,
    episode_id_offset: int = 0,
) -> tuple[int, int]:
    """
    エピソードをJSON + JPEG画像として保存する。(エピソード数, ステップ数) を返す。
    annotate.py と同じディレクトリ構造で出力する。
    """
    steps_dir = output_dir / "steps"
    episodes_dir = output_dir / "episodes"
    steps_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)

    step_id = 0
    # 既存のstepファイルと番号が衝突しないようにオフセットを計算
    existing = sorted(steps_dir.glob("step_*.jpg"))
    if existing:
        last = int(existing[-1].stem.split("_")[1])
        step_id = last + 1

    n_episodes = 0
    for ep_idx, steps in enumerate(episodes):
        ep_id = episode_id_offset + ep_idx
        episode_steps = []

        for seq_idx, step in enumerate(steps):
            img_filename = f"step_{step_id:06d}.jpg"
            img_bgr = cv2.cvtColor(step["image"], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(steps_dir / img_filename), img_bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

            episode_steps.append({
                "step_id": step_id,
                "episode_id": ep_id,
                "image_path": f"steps/{img_filename}",
                "instruction": INSTRUCTION,
                "action_vector": step["action"],
                "is_first": seq_idx == 0,
                "is_last": seq_idx == len(steps) - 1,
            })
            step_id += 1

        ep_path = episodes_dir / f"episode_{ep_id:04d}.json"
        with open(ep_path, "w") as f:
            json.dump({
                "episode_id": ep_id,
                "steps": episode_steps,
            }, f, indent=2)

        n_episodes += 1

    return n_episodes, step_id


# ─── メイン ──────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="サビ線合成 + DFS探索 → JSON + 画像ファイル出力 (TF不要)"
    )
    parser.add_argument("--output_dir", type=str, default="data/raw",
                        help="出力ディレクトリ (デフォルト: data/raw)")
    parser.add_argument("--n_source_images", type=int, default=50,
                        help="生成する特大画像の枚数 (デフォルト: 50)")
    parser.add_argument("--image_size", type=int, nargs=2, default=[1120, 1120],
                        help="特大画像サイズ H W (デフォルト: 1120 1120)")
    parser.add_argument("--n_rust_components", type=int, default=1,
                        help="画像あたりのサビ連結成分数 (デフォルト: 1)")
    parser.add_argument("--n_strokes", type=int, default=30,
                        help="サビ1成分あたりの最大ストローク数 (デフォルト: 30)")
    parser.add_argument("--stroke_length", type=int, default=200,
                        help="ストローク最大長さ px (デフォルト: 200)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    all_episodes: list[list[dict]] = []

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"[generate] {args.n_source_images} 枚の画像からエピソードを生成中...")
    for i in range(args.n_source_images):
        episodes, full_img = generate_episodes_from_image(
            rng,
            image_size=tuple(args.image_size),
            n_rust_components=args.n_rust_components,
            n_strokes=args.n_strokes,
            stroke_length=args.stroke_length,
        )
        # 全体画像を保存 (annotate.py 用)
        cv2.imwrite(str(images_dir / f"source_{i:04d}.jpg"), full_img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        all_episodes.extend(episodes)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.n_source_images} 完了 (累積エピソード数: {len(all_episodes)})")

    print(f"\n[generate] JSON + 画像を書き込み中 → {output_dir}")
    n_ep, n_steps = write_raw_episodes(all_episodes, output_dir)

    print(f"\n[完了] {n_ep} エピソード, {n_steps} ステップ → {output_dir}")

    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump({
            "n_episodes": n_ep,
            "n_steps": n_steps,
            "patch_size": PATCH_SIZE,
            "action_dim": ACTION_DIM,
            "instruction": INSTRUCTION,
            "action_format": "[delta_col, delta_row, 0, 0, 0, 0, 0]",
        }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
