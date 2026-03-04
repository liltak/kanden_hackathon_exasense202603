"""
タスク1: サビ線合成 + 探索ログ生成スクリプト

【実行環境】Mac ローカル / H100 どちらでも動作
  依存: opencv-python, numpy のみ (GPU 不要)
  pip install opencv-python numpy

リアルテクスチャ画像上にOpenCVでサビ線を合成し、
連結成分ベースのDFS探索によって各ステップを記録する。

出力形式:
  output_dir/
    steps/
      step_{n:06d}.png     # 224x224 パッチ画像
    metadata.json           # 全ステップのメタデータ
    episodes/
      episode_{n:04d}.json  # エピソード単位のメタデータ
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ─── アクション定義 ────────────────────────────────────────────────────────
# 9 方向の離散アクションを 3D 連続ベクトルにマッピング
ACTIONS: dict[str, tuple[float, float, float]] = {
    "up":         (0.0,   1.0,  0.0),
    "down":       (0.0,  -1.0,  0.0),
    "left":       (-1.0,  0.0,  0.0),
    "right":      (1.0,   0.0,  0.0),
    "upper_right":( 0.707, 0.707, 0.0),
    "upper_left": (-0.707, 0.707, 0.0),
    "lower_right":( 0.707,-0.707, 0.0),
    "lower_left": (-0.707,-0.707, 0.0),
    "backtrack":  (0.0,   0.0,  1.0),  # z=1: バックトラック専用
}

# グリッド移動のデルタ (row_delta, col_delta)
ACTION_DELTAS: dict[str, tuple[int, int]] = {
    "up":          (-1,  0),
    "down":        ( 1,  0),
    "left":        ( 0, -1),
    "right":       ( 0,  1),
    "upper_right": (-1,  1),
    "upper_left":  (-1, -1),
    "lower_right": ( 1,  1),
    "lower_left":  ( 1, -1),
}

PATCH_SIZE = 224  # OpenVLA の入力サイズ


# ─── データクラス ─────────────────────────────────────────────────────────
@dataclass
class Step:
    step_id: int
    episode_id: int
    image_path: str            # 相対パス
    instruction: str
    action_name: str
    action_vector: list[float]  # [x, y, z]
    grid_row: int
    grid_col: int
    is_rust: bool
    is_backtrack: bool
    component_id: int


@dataclass
class Episode:
    episode_id: int
    source_image: str
    grid_rows: int
    grid_cols: int
    steps: list[Step] = field(default_factory=list)


# ─── テクスチャ生成 ───────────────────────────────────────────────────────
def generate_concrete_texture(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """コンクリート調のグレーテクスチャを生成する。"""
    base = rng.integers(100, 160, (height, width), dtype=np.uint8)
    noise = rng.integers(0, 30, (height, width), dtype=np.uint8)
    img = cv2.add(base, noise)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # ランダムなひび割れ模様
    for _ in range(rng.integers(3, 8)):
        pt1 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        pt2 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        cv2.line(img, pt1, pt2, (70, 70, 70), int(rng.integers(1, 3)))
    return img


def generate_steel_texture(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """鉄板調のテクスチャを生成する。"""
    base = rng.integers(60, 110, (height, width), dtype=np.uint8)
    # 水平方向のスクラッチ
    for _ in range(rng.integers(5, 20)):
        row = int(rng.integers(0, height))
        val = int(rng.integers(80, 130))
        base[row, :] = np.clip(base[row, :].astype(int) + val - 90, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    # わずかな青みを追加して金属感を演出
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
    # ベジェ曲線風の揺らぎを追加
    mid_x = (start[0] + end[0]) // 2 + int(rng.integers(-20, 20))
    mid_y = (start[1] + end[1]) // 2 + int(rng.integers(-20, 20))
    pts = np.array([[start], [(mid_x, mid_y)], [end]], dtype=np.float32)
    # 二次ベジェ曲線をポリラインで近似
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
    n_components: int = 3,
    branch_prob: float = 0.3,
) -> np.ndarray:
    """
    画像上にサビ線を合成し、サビマスク (0/255) を返す。

    - Y字分岐: branch_prob で分岐点を生成
    - 行き止まり: 確率的に経路を短く打ち切る
    - 途切れ: 途中で乱数的にギャップを挿入
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for _ in range(n_components):
        # 開始点をランダムに決定
        sx, sy = int(rng.integers(20, w - 20)), int(rng.integers(20, h - 20))
        queue = [(sx, sy)]

        for _ in range(rng.integers(5, 15)):
            if not queue:
                break
            cx, cy = queue.pop(rng.integers(0, len(queue)))

            # 次の点へランダムに移動
            angle = rng.uniform(0, 2 * math.pi)
            length = int(rng.integers(30, 100))
            nx = int(np.clip(cx + length * math.cos(angle), 10, w - 10))
            ny = int(np.clip(cy + length * math.sin(angle), 10, h - 10))

            # 途切れギャップを確率的に挿入
            if rng.random() > 0.15:
                _draw_rust_stroke(mask, img, (cx, cy), (nx, ny), rng)

            queue.append((nx, ny))

            # Y字分岐
            if rng.random() < branch_prob:
                angle2 = angle + rng.uniform(math.pi / 6, math.pi / 3)
                length2 = int(rng.integers(20, 70))
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

    # ガウシアンノイズ
    noise = rng.normal(0, rng.uniform(3, 12), aug.shape).astype(np.int16)
    aug = np.clip(aug.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # ガウシアンブラー (奇数カーネルサイズ)
    ksize = int(rng.choice([3, 5, 7]))
    aug = cv2.GaussianBlur(aug, (ksize, ksize), 0)

    # 色ムラ (チャンネルごとにスケール)
    for c in range(3):
        scale = rng.uniform(0.85, 1.15)
        aug[:, :, c] = np.clip(aug[:, :, c].astype(float) * scale, 0, 255).astype(np.uint8)

    # ランダムな明暗パッチ (汚れ・光反射)
    n_patches = int(rng.integers(0, 5))
    h, w = aug.shape[:2]
    for _ in range(n_patches):
        px, py = int(rng.integers(0, w)), int(rng.integers(0, h))
        pr = int(rng.integers(10, 50))
        brightness = int(rng.integers(-30, 30))
        cv2.circle(aug, (px, py), pr, (brightness, brightness, brightness), -1)
        aug = np.clip(aug.astype(np.int16), 0, 255).astype(np.uint8)

    return aug


# ─── グリッド分割 + パッチ抽出 ────────────────────────────────────────────
def split_into_patches(
    img: np.ndarray,
    mask: np.ndarray,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    画像をパッチサイズで格子状に分割する。
    各パッチは (patch_size, patch_size) にリサイズして返す。

    Returns:
        patches:      (rows, cols, patch_size, patch_size, 3)
        patch_masks:  (rows, cols, patch_size, patch_size)  binary
        rows, cols: グリッドサイズ
    """
    h, w = img.shape[:2]
    rows = max(1, h // patch_size)
    cols = max(1, w // patch_size)

    patches = np.zeros((rows, cols, patch_size, patch_size, 3), dtype=np.uint8)
    patch_masks = np.zeros((rows, cols, patch_size, patch_size), dtype=np.uint8)

    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * patch_size, (r + 1) * patch_size
            x0, x1 = c * patch_size, (c + 1) * patch_size
            crop = img[y0:y1, x0:x1]
            mask_crop = mask[y0:y1, x0:x1]
            patches[r, c] = cv2.resize(crop, (patch_size, patch_size))
            patch_masks[r, c] = cv2.resize(mask_crop, (patch_size, patch_size))

    return patches, patch_masks, rows, cols


# ─── 連結成分ベースの二段構え探索 ────────────────────────────────────────
def _get_connected_components(
    patch_has_rust: np.ndarray,
) -> tuple[np.ndarray, int]:
    """
    サビを持つパッチの連結成分ラベルを返す。
    8連結でラベリングする。

    Returns:
        labels: (rows, cols) int, 0 = サビなし, 1〜N = 各成分
        n_labels: 成分数
    """
    binary = (patch_has_rust > 0).astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(binary, connectivity=8)
    return labels, n_labels - 1  # 0ラベル(背景)を除く


def _nearest_unvisited_component(
    current_r: int,
    current_c: int,
    labels: np.ndarray,
    visited_components: set[int],
) -> Optional[tuple[int, int, int]]:
    """
    現在位置から最も近い未探索連結成分のパッチ座標を返す。

    Returns:
        (row, col, component_id) or None
    """
    rows, cols = labels.shape
    best_dist = float("inf")
    best_pos = None
    best_comp = None

    for r in range(rows):
        for c in range(cols):
            comp = labels[r, c]
            if comp == 0:
                continue
            if comp in visited_components:
                continue
            dist = math.hypot(r - current_r, c - current_c)
            if dist < best_dist:
                best_dist = dist
                best_pos = (r, c)
                best_comp = comp

    if best_pos is None:
        return None
    return (best_pos[0], best_pos[1], best_comp)


def _delta_to_action(dr: int, dc: int) -> str:
    """(row_delta, col_delta) を最近傍アクション名に変換する。"""
    if dr == 0 and dc == 0:
        return "backtrack"
    best_action = "up"
    best_dist = float("inf")
    for action, (ar, ac) in ACTION_DELTAS.items():
        dist = math.hypot(dr - ar, dc - ac)
        if dist < best_dist:
            best_dist = dist
            best_action = action
    return best_action


def _dfs_component(
    start_r: int,
    start_c: int,
    component_id: int,
    labels: np.ndarray,
    visited_patches: set[tuple[int, int]],
    patches: np.ndarray,
    patch_masks: np.ndarray,
    episode: Episode,
    output_dir: Path,
    step_counter: list[int],  # mutable counter
    rng: np.random.Generator,
) -> None:
    """
    単一連結成分内を DFS で探索し、ステップを記録する。
    """
    rows, cols = labels.shape
    stack: list[tuple[int, int]] = [(start_r, start_c)]
    parent: dict[tuple[int, int], Optional[tuple[int, int]]] = {(start_r, start_c): None}
    visited_patches.add((start_r, start_c))

    while stack:
        r, c = stack[-1]

        # 未訪問の隣接パッチを探す
        neighbors = []
        for action_name, (dr, dc) in ACTION_DELTAS.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if labels[nr, nc] == component_id and (nr, nc) not in visited_patches:
                    neighbors.append((nr, nc, action_name))

        if neighbors:
            # ランダムに次のパッチを選択
            rng.shuffle(np.array(range(len(neighbors))))
            nr, nc, action_name = neighbors[rng.integers(0, len(neighbors))]

            # ステップ記録
            action_vec = list(ACTIONS[action_name])
            instruction = (
                f"Follow the rust trace. Move {action_name.replace('_', ' ')} "
                f"to continue tracking the corrosion path."
            )
            step = _record_step(
                r, c, nr, nc, action_name, action_vec,
                instruction, component_id, False,
                patches[nr, nc], episode, output_dir, step_counter
            )
            episode.steps.append(step)

            visited_patches.add((nr, nc))
            parent[(nr, nc)] = (r, c)
            stack.append((nr, nc))

        else:
            # バックトラック
            stack.pop()
            if stack:
                pr, pc = stack[-1]
                dr, dc = pr - r, pc - c
                action_vec = list(ACTIONS["backtrack"])
                instruction = (
                    "Dead end reached. Backtracking to explore alternative rust path."
                )
                step = _record_step(
                    r, c, pr, pc, "backtrack", action_vec,
                    instruction, component_id, True,
                    patches[pr, pc], episode, output_dir, step_counter
                )
                episode.steps.append(step)


def _record_step(
    from_r: int, from_c: int,
    to_r: int, to_c: int,
    action_name: str,
    action_vec: list[float],
    instruction: str,
    component_id: int,
    is_backtrack: bool,
    patch_img: np.ndarray,
    episode: Episode,
    output_dir: Path,
    step_counter: list[int],
) -> Step:
    """パッチ画像を保存してステップを返す。"""
    step_id = step_counter[0]
    step_counter[0] += 1

    img_filename = f"step_{step_id:06d}.png"
    img_path = output_dir / "steps" / img_filename
    cv2.imwrite(str(img_path), patch_img)

    return Step(
        step_id=step_id,
        episode_id=episode.episode_id,
        image_path=str(Path("steps") / img_filename),
        instruction=instruction,
        action_name=action_name,
        action_vector=action_vec,
        grid_row=int(to_r),
        grid_col=int(to_c),
        is_rust=(action_name != "backtrack"),
        is_backtrack=is_backtrack,
        component_id=int(component_id),
    )


# ─── エピソード生成 ───────────────────────────────────────────────────────
def generate_episode(
    episode_id: int,
    output_dir: Path,
    step_counter: list[int],
    rng: np.random.Generator,
    image_size: tuple[int, int] = (1120, 1120),
    n_rust_components: int = 3,
) -> Episode:
    """
    1エピソード = 1枚の特大画像を格子分割して探索する。
    """
    h, w = image_size
    img = generate_base_texture(h, w, rng)
    img = apply_domain_randomization(img, rng)
    rust_mask = synthesize_rust_lines(img, rng, n_components=n_rust_components)

    # ドメインランダマイゼーション後に再度ぼかして自然な見た目に
    img = apply_domain_randomization(img, rng)

    patches, patch_masks, rows, cols = split_into_patches(img, rust_mask, PATCH_SIZE)

    episode = Episode(
        episode_id=episode_id,
        source_image=f"episode_{episode_id:04d}_source.png",
        grid_rows=rows,
        grid_cols=cols,
    )

    # ソース画像を保存
    source_path = output_dir / "sources" / episode.source_image
    source_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(source_path), img)

    # パッチごとのサビ有無 (8x8グリッドなど)
    patch_has_rust = (patch_masks.sum(axis=(2, 3)) > 0).astype(np.uint8)
    labels, n_components = _get_connected_components(patch_has_rust)

    if n_components == 0:
        return episode  # サビなしエピソードはスキップ

    visited_patches: set[tuple[int, int]] = set()
    visited_components: set[int] = set()

    # 最初の開始パッチ: ラベル1の最初のパッチ
    start_r, start_c = np.argwhere(labels == 1)[0]

    current_r, current_c = int(start_r), int(start_c)

    while True:
        # 現在の連結成分を DFS 探索
        comp_id = labels[current_r, current_c]
        if comp_id != 0 and comp_id not in visited_components:
            visited_components.add(comp_id)
            _dfs_component(
                current_r, current_c,
                comp_id,
                labels, visited_patches, patches, patch_masks,
                episode, output_dir, step_counter, rng,
            )

        # 次の未探索成分へジャンプ
        next_info = _nearest_unvisited_component(
            current_r, current_c, labels, visited_components
        )
        if next_info is None:
            break

        nr, nc, next_comp = next_info
        # ジャンプステップを記録 (連続した成分間の遷移)
        action_name = _delta_to_action(nr - current_r, nc - current_c)
        action_vec = list(ACTIONS[action_name])
        instruction = (
            "No adjacent rust found. Jumping to nearest unvisited rust component "
            "to continue coverage."
        )
        step = _record_step(
            current_r, current_c, nr, nc,
            action_name, action_vec, instruction,
            next_comp, False,
            patches[nr, nc], episode, output_dir, step_counter
        )
        episode.steps.append(step)

        visited_patches.add((nr, nc))
        current_r, current_c = nr, nc

    return episode


# ─── メイン ──────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="サビ線合成 + 探索ログ生成")
    parser.add_argument("--output_dir", type=str, default="data/rust_dataset",
                        help="出力ディレクトリ")
    parser.add_argument("--n_episodes", type=int, default=50,
                        help="生成エピソード数")
    parser.add_argument("--image_size", type=int, nargs=2, default=[1120, 1120],
                        help="特大画像サイズ (H W)")
    parser.add_argument("--n_rust_components", type=int, default=3,
                        help="エピソードあたりのサビ連結成分数")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    (output_dir / "steps").mkdir(parents=True, exist_ok=True)
    (output_dir / "episodes").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    step_counter = [0]
    all_steps: list[dict] = []
    all_episodes: list[dict] = []

    print(f"[generate_dataset] {args.n_episodes} エピソードを生成します...")

    for ep_id in range(args.n_episodes):
        episode = generate_episode(
            episode_id=ep_id,
            output_dir=output_dir,
            step_counter=step_counter,
            rng=rng,
            image_size=tuple(args.image_size),
            n_rust_components=args.n_rust_components,
        )

        # エピソードメタデータを保存
        ep_dict = asdict(episode)
        ep_path = output_dir / "episodes" / f"episode_{ep_id:04d}.json"
        with open(ep_path, "w") as f:
            json.dump(ep_dict, f, indent=2)

        all_steps.extend([asdict(s) for s in episode.steps])
        ep_dict_summary = {k: v for k, v in ep_dict.items() if k != "steps"}
        ep_dict_summary["n_steps"] = len(episode.steps)
        all_episodes.append(ep_dict_summary)

        if (ep_id + 1) % 10 == 0:
            print(f"  エピソード {ep_id + 1}/{args.n_episodes} 完了 "
                  f"(累積ステップ数: {step_counter[0]})")

    # 全体メタデータを保存
    metadata = {
        "n_episodes": len(all_episodes),
        "n_steps": step_counter[0],
        "patch_size": PATCH_SIZE,
        "actions": {k: list(v) for k, v in ACTIONS.items()},
        "episodes": all_episodes,
        "steps": all_steps,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[generate_dataset] 完了!")
    print(f"  エピソード数: {len(all_episodes)}")
    print(f"  総ステップ数: {step_counter[0]}")
    print(f"  出力先: {output_dir}")


if __name__ == "__main__":
    main()
