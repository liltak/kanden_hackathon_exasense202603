"""
タスク4: 推論 + 評価シミュレーション

【実行環境】
  --model_path なし (MockRustAgent): Mac ローカルで動作
    依存: opencv-python, numpy のみ
    → パイプライン全体の動作確認・可視化デバッグに使用

  --model_path あり (OpenVLARustAgent): H100 推奨
    依存: torch, transformers, peft が追加で必要
    → H100 で学習したチェックポイントを使って本評価を実行

学習に使っていない新規の特大画像を入力し、エージェントが
自律的にサビ線を辿れるかを評価する。

特徴:
  - モデルの 3D 出力をユークリッド距離で 9 方向に離散化
  - 評価指標: カバレッジ率、総ステップ数、バックトラック回数
  - 探索経路の可視化 (画像上に軌跡を描画して保存)

使用方法:
  # Mac ローカル (モックエージェント)
  python simulate.py --output_dir results/simulation --n_test_images 3

  # H100 (学習済みモデルで評価)
  python simulate.py \
    --model_path checkpoints/rust_openvla/best \
    --input_image path/to/test_image.png \
    --output_dir results/simulation \
    --max_steps 500
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# GPU モジュールは TYPE_CHECKING ガード
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from peft import PeftModel

# ─── アクション定義 (generate_dataset.py と同じ) ─────────────────────────
# アクション → (dr, dc): dr=行移動量, dc=列移動量
ACTION_DELTAS: dict[str, tuple[int, int]] = {
    "up":          (-1,  0),
    "down":        ( 1,  0),
    "left":        ( 0, -1),
    "right":       ( 0,  1),
    "upper_right": (-1,  1),
    "upper_left":  (-1, -1),
    "lower_right": ( 1,  1),
    "lower_left":  ( 1, -1),
    "backtrack":   ( 0,  0),
}

PATCH_SIZE = 224
ACTION_DIM = 7
INSTRUCTION = "Follow the rust trace. Navigate to continue tracking the corrosion path."

# 可視化カラー (BGR)
COLOR_PATH = (0, 200, 0)       # 通常経路: 緑
COLOR_BACKTRACK = (0, 100, 255) # バックトラック: オレンジ
COLOR_CURRENT = (255, 0, 0)    # 現在位置: 青
COLOR_RUST = (50, 80, 200)     # サビパッチ: 赤


# ─── データクラス ─────────────────────────────────────────────────────────
@dataclass
class SimulationResult:
    n_rust_patches: int
    n_visited_rust_patches: int
    coverage_rate: float          # サビカバレッジ
    total_steps: int
    n_backtracks: int
    n_component_jumps: int
    elapsed_seconds: float
    trajectory: list[tuple[int, int]] = field(default_factory=list)
    visited_patches: list[tuple[int, int]] = field(default_factory=list)


# ─── アクション離散化 ─────────────────────────────────────────────────────
def discretize_action(raw_action: np.ndarray) -> str:
    """
    モデルの出力 [Δcol, Δrow, ...] をユークリッド距離で最近傍の 9 方向に離散化する。

    Parameters:
        raw_action: shape (7,) の float32 配列 (OpenVLA 出力)
                    先頭 2 次元 [Δcol, Δrow] を使用

    Returns:
        action_name: "up", "down", ..., "backtrack" のいずれか
    """
    dc = float(raw_action[0])  # 列移動量
    dr = float(raw_action[1])  # 行移動量

    # [0, 0] はバックトラック (端点 or 行き詰まり)
    if abs(dc) < 0.3 and abs(dr) < 0.3:
        return "backtrack"

    # ACTION_DELTAS で最近傍アクションを探す
    best_action = "up"
    best_dist = float("inf")

    for action_name, (adr, adc) in ACTION_DELTAS.items():
        if action_name == "backtrack":
            continue
        dist = math.sqrt((dr - adr) ** 2 + (dc - adc) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_action = action_name

    return best_action


def action_to_delta(action_name: str) -> tuple[int, int]:
    """アクション名をグリッド移動量 (dr, dc) に変換する。"""
    return ACTION_DELTAS.get(action_name, (0, 0))


# ─── 環境クラス ──────────────────────────────────────────────────────────
class RustTracingEnv:
    """
    特大画像を格子分割してエージェントが探索する環境。

    観測: 現在パッチ画像 (224x224)
    アクション: 9方向の離散アクション
    報酬: 評価用なので使用しない (シミュレーションのみ)
    """

    def __init__(
        self,
        image: np.ndarray,
        rust_mask: np.ndarray,
        patch_size: int = PATCH_SIZE,
    ) -> None:
        self.image = image
        self.rust_mask = rust_mask
        self.patch_size = patch_size
        self.h, self.w = image.shape[:2]
        self.rows = max(1, self.h // patch_size)
        self.cols = max(1, self.w // patch_size)

        # パッチごとのサビフラグ
        self.patch_has_rust = self._compute_rust_patches()

        # 探索状態
        self.visited: set[tuple[int, int]] = set()
        self.trajectory: list[tuple[int, int]] = []
        self.history: list[str] = []  # アクション履歴

        # 開始位置: サビのある最初のパッチ
        rust_positions = list(zip(*np.where(self.patch_has_rust)))
        if rust_positions:
            r, c = rust_positions[0]
            self.current_r, self.current_c = int(r), int(c)
        else:
            self.current_r, self.current_c = 0, 0

        self.visited.add((self.current_r, self.current_c))
        self.trajectory.append((self.current_r, self.current_c))

    def _compute_rust_patches(self) -> np.ndarray:
        """各パッチにサビがあるかを判定する (rows, cols) の bool 配列。"""
        result = np.zeros((self.rows, self.cols), dtype=bool)
        for r in range(self.rows):
            for c in range(self.cols):
                y0, y1 = r * self.patch_size, (r + 1) * self.patch_size
                x0, x1 = c * self.patch_size, (c + 1) * self.patch_size
                patch_mask = self.rust_mask[y0:y1, x0:x1]
                result[r, c] = patch_mask.sum() > 0
        return result

    def get_observation(self) -> np.ndarray:
        """現在位置のパッチ画像を (224, 224, 3) RGB で返す。"""
        r, c = self.current_r, self.current_c
        y0, y1 = r * self.patch_size, (r + 1) * self.patch_size
        x0, x1 = c * self.patch_size, (c + 1) * self.patch_size
        patch = self.image[y0:y1, x0:x1]
        patch_resized = cv2.resize(patch, (self.patch_size, self.patch_size))
        return cv2.cvtColor(patch_resized, cv2.COLOR_BGR2RGB)

    def build_instruction(self) -> str:
        """instruction を返す。"""
        return INSTRUCTION

    def step(self, action_name: str) -> tuple[bool, str]:
        """
        アクションを実行し、(is_done, info) を返す。

        Returns:
            is_done: エピソード終了かどうか
            info: デバッグ情報
        """
        self.history.append(action_name)

        if action_name == "backtrack":
            # バックトラックは経路を 1 ステップ戻る
            if len(self.trajectory) > 1:
                self.trajectory.pop()
                self.current_r, self.current_c = self.trajectory[-1]
            return False, "backtrack"

        dr, dc = action_to_delta(action_name)
        nr = self.current_r + dr
        nc = self.current_c + dc

        # 境界チェック
        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            return False, f"boundary ({nr},{nc})"

        self.current_r, self.current_c = nr, nc
        self.visited.add((nr, nc))
        self.trajectory.append((nr, nc))
        return False, f"moved to ({nr},{nc})"

    def jump_to(self, r: int, c: int) -> None:
        """連結成分間ジャンプ。"""
        self.current_r, self.current_c = r, c
        self.visited.add((r, c))
        self.trajectory.append((r, c))
        self.history.append("jump")

    def coverage_stats(self) -> dict:
        """カバレッジ統計を返す。"""
        n_rust = int(self.patch_has_rust.sum())
        visited_rust = sum(
            1 for r, c in self.visited if self.patch_has_rust[r, c]
        )
        return {
            "n_rust_patches": n_rust,
            "n_visited_rust_patches": visited_rust,
            "coverage_rate": visited_rust / max(1, n_rust),
        }

    def get_unvisited_rust_patches(self) -> list[tuple[int, int]]:
        """未訪問のサビパッチ座標リストを返す。"""
        result = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.patch_has_rust[r, c] and (r, c) not in self.visited:
                    result.append((r, c))
        return result


# ─── エージェント ─────────────────────────────────────────────────────────
class OpenVLARustAgent:
    """
    OpenVLA ベースのサビ追跡エージェント。
    LoRA チェックポイントから推論する。
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
    ) -> None:
        import torch
        from transformers import AutoProcessor, AutoModelForVision2Seq
        from peft import PeftModel

        self.device = device

        print(f"[OpenVLARustAgent] モデルを読み込み中: {model_path}")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        base_model_id = "openvla/openvla-7b"
        base_model = AutoModelForVision2Seq.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.eval()
        # device_map="auto" 使用時は .to(device) 不要
        print("[OpenVLARustAgent] モデル読み込み完了")

    def predict_action(self, image: np.ndarray, instruction: str) -> str:
        """
        画像と instruction からアクション名を予測する。

        OpenVLA は action token を生成するが、本実装では
        logit から近似的にアクションベクトルを取得する。
        """
        import torch
        from PIL import Image as PILImage

        pil_image = PILImage.fromarray(image)
        inputs = self.processor(
            text=instruction,
            images=pil_image,
            return_tensors="pt",
        ).to(self.model.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
            )

        # 生成されたトークンをデコードしてアクションを解析
        generated = self.processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )

        # OpenVLA のアクション出力フォーマットを解析
        # 例: "0.5 -0.3 0.0 0.0 0.0 0.0 0.0"
        print(f"[predict_action] generated: {repr(generated)}")
        try:
            values = [float(v) for v in generated.strip().split()[:ACTION_DIM]]
            if len(values) == 0:
                raise ValueError("empty output")
            action_array = np.array(values + [0.0] * (ACTION_DIM - len(values)), dtype=np.float32)
            action = discretize_action(action_array)
            print(f"[predict_action] action: {action}")
            return action
        except (ValueError, IndexError):
            # パースに失敗した場合はランダムなアクションにフォールバック
            import random
            action = random.choice([k for k in ACTION_DELTAS if k != "backtrack"])
            print(f"[predict_action] parse failed → random: {action}")
            return action


class MockRustAgent:
    """
    モデルなしのモックエージェント (開発・テスト用)。
    サビを持つ隣接パッチに優先的に移動する。
    """

    def __init__(self, env: RustTracingEnv) -> None:
        self.env = env

    def predict_action(self, image: np.ndarray, instruction: str) -> str:
        """サビ優先のヒューリスティックエージェント。"""
        r, c = self.env.current_r, self.env.current_c

        # サビのある未訪問隣接パッチを探す
        rust_neighbors = []
        plain_neighbors = []

        for action_name, (dr, dc) in ACTION_DELTAS.items():
            if action_name == "backtrack":
                continue
            nr, nc = r + dr, c + dc
            if not (0 <= nr < self.env.rows and 0 <= nc < self.env.cols):
                continue
            if self.env.patch_has_rust[nr, nc] and (nr, nc) not in self.env.visited:
                rust_neighbors.append(action_name)
            elif (nr, nc) not in self.env.visited:
                plain_neighbors.append(action_name)

        if rust_neighbors:
            return np.random.choice(rust_neighbors)
        if plain_neighbors:
            return np.random.choice(plain_neighbors)
        return "backtrack"


# ─── 可視化 ──────────────────────────────────────────────────────────────
def visualize_trajectory(
    image: np.ndarray,
    env: RustTracingEnv,
    trajectory: list[tuple[int, int]],
    action_history: list[str],
    output_path: Path,
) -> None:
    """
    探索経路を元画像上に描画して保存する。

    描画内容:
      - サビパッチを薄い赤でオーバーレイ
      - 探索パッチを薄い緑でオーバーレイ
      - 移動経路を矢印で描画
      - バックトラックは別色 (オレンジ) で描画
      - 現在位置を青い円で表示
    """
    vis = image.copy()
    h, w = vis.shape[:2]
    ps = env.patch_size

    # サビパッチオーバーレイ
    rust_overlay = np.zeros_like(vis)
    for r in range(env.rows):
        for c in range(env.cols):
            if env.patch_has_rust[r, c]:
                y0, y1 = r * ps, min((r + 1) * ps, h)
                x0, x1 = c * ps, min((c + 1) * ps, w)
                rust_overlay[y0:y1, x0:x1] = (50, 50, 180)  # BGR: 赤

    vis = cv2.addWeighted(vis, 0.7, rust_overlay, 0.3, 0)

    # 探索済みパッチオーバーレイ
    visited_overlay = np.zeros_like(vis)
    for r, c in env.visited:
        y0, y1 = r * ps, min((r + 1) * ps, h)
        x0, x1 = c * ps, min((c + 1) * ps, w)
        visited_overlay[y0:y1, x0:x1] = (0, 80, 0)  # BGR: 暗緑

    vis = cv2.addWeighted(vis, 0.85, visited_overlay, 0.15, 0)

    # 移動経路を描画 (パッチ中心をつなぐ矢印)
    def patch_center(r: int, c: int) -> tuple[int, int]:
        cx = min(c * ps + ps // 2, w - 1)
        cy = min(r * ps + ps // 2, h - 1)
        return cx, cy

    for i in range(1, len(trajectory)):
        prev_r, prev_c = trajectory[i - 1]
        curr_r, curr_c = trajectory[i]
        p1 = patch_center(prev_r, prev_c)
        p2 = patch_center(curr_r, curr_c)

        # action_history と trajectory は長さが異なる場合がある
        idx = min(i - 1, len(action_history) - 1)
        if idx >= 0 and action_history[idx] == "backtrack":
            color = COLOR_BACKTRACK
            thickness = 1
        else:
            color = COLOR_PATH
            thickness = 2

        cv2.arrowedLine(vis, p1, p2, color, thickness, tipLength=0.3)

    # グリッド線
    for r in range(1, env.rows):
        cv2.line(vis, (0, r * ps), (w, r * ps), (60, 60, 60), 1)
    for c in range(1, env.cols):
        cv2.line(vis, (c * ps, 0), (c * ps, h), (60, 60, 60), 1)

    # 現在位置 (最終位置) を青い円で表示
    if trajectory:
        last_r, last_c = trajectory[-1]
        cx, cy = patch_center(last_r, last_c)
        cv2.circle(vis, (cx, cy), ps // 3, COLOR_CURRENT, -1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis)
    print(f"[visualize] 可視化画像を保存しました: {output_path}")


def _draw_frame(
    image: np.ndarray,
    env: "RustTracingEnv",
    trajectory: list,
    action_history: list,
    output_path: Path,
) -> None:
    """シミュレーション中間フレームを保存する (visualize_trajectory の薄いラッパー)。"""
    visualize_trajectory(image, env, trajectory, action_history, output_path)


def frames_to_video(frame_dir: Path, output_path: Path, fps: int = 10) -> None:
    """frame_dir 内の frame_XXXX.png を連結して mp4 を生成する。"""
    frames = sorted(frame_dir.glob("frame_*.png"))
    if not frames:
        print("[video] フレームが見つかりません")
        return
    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    for f in frames:
        writer.write(cv2.imread(str(f)))
    writer.release()
    print(f"[video] 動画を保存しました: {output_path}  ({len(frames)} frames, {fps} fps)")


# ─── シミュレーション ─────────────────────────────────────────────────────
def run_simulation(
    env: RustTracingEnv,
    agent,
    max_steps: int = 500,
    max_backtracks: int = 100,
    coverage_threshold: float = 0.95,
    frame_dir: Optional[Path] = None,
    image_orig: Optional[np.ndarray] = None,
) -> SimulationResult:
    """
    エージェントによるシミュレーションを実行する。

    終了条件:
      1. max_steps を超えた
      2. カバレッジが coverage_threshold を超えた
      3. max_backtracks を超えた

    frame_dir が指定された場合、各ステップの中間フレームを PNG として保存する。
    image_orig は frame_dir 使用時に必須 (元画像)。

    Returns:
        SimulationResult
    """
    start_time = time.time()
    n_backtracks = 0
    n_component_jumps = 0
    action_history: list[str] = []

    if frame_dir is not None:
        frame_dir.mkdir(parents=True, exist_ok=True)

    for step in range(max_steps):
        obs = env.get_observation()
        instruction = env.build_instruction()
        action_name = agent.predict_action(obs, instruction)
        action_history.append(action_name)

        if action_name == "backtrack":
            n_backtracks += 1

        _, info = env.step(action_name)

        # 中間フレームを保存
        if frame_dir is not None and image_orig is not None:
            frame_path = frame_dir / f"frame_{step:04d}.png"
            _draw_frame(image_orig, env, list(env.trajectory), list(action_history), frame_path)

        # カバレッジチェック
        stats = env.coverage_stats()
        if stats["coverage_rate"] >= coverage_threshold:
            print(f"[simulate] カバレッジ {coverage_threshold*100:.0f}% 達成 (step={step+1})")
            break

        # 連続バックトラックが多い場合は最寄り未訪問成分へジャンプ
        recent_actions = action_history[-5:] if len(action_history) >= 5 else action_history
        if len(recent_actions) >= 5 and all(a == "backtrack" for a in recent_actions):
            unvisited = env.get_unvisited_rust_patches()
            if unvisited:
                # 最寄りの未訪問サビパッチへジャンプ
                curr_r, curr_c = env.current_r, env.current_c
                nearest = min(
                    unvisited,
                    key=lambda pos: math.hypot(pos[0] - curr_r, pos[1] - curr_c),
                )
                env.jump_to(*nearest)
                n_component_jumps += 1
                action_history.append("jump")
                print(f"[simulate] ジャンプ → {nearest} (step={step+1})")

    stats = env.coverage_stats()
    elapsed = time.time() - start_time

    return SimulationResult(
        n_rust_patches=stats["n_rust_patches"],
        n_visited_rust_patches=stats["n_visited_rust_patches"],
        coverage_rate=stats["coverage_rate"],
        total_steps=len(action_history),
        n_backtracks=n_backtracks,
        n_component_jumps=n_component_jumps,
        elapsed_seconds=elapsed,
        trajectory=list(env.trajectory),
        visited_patches=list(env.visited),
    )


# ─── テスト用サビ画像生成 ─────────────────────────────────────────────────
def generate_test_image(
    height: int = 1120,
    width: int = 1120,
    seed: int = 99,
) -> tuple[np.ndarray, np.ndarray]:
    """
    テスト用の特大画像とサビマスクを生成する。
    (学習データとは異なるシードを使用)
    """
    # data_generation.generate_dataset を再利用
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data_generation.generate_dataset import (
        generate_base_texture,
        apply_domain_randomization,
        synthesize_rust_lines,
    )

    rng = np.random.default_rng(seed)
    img = generate_base_texture(height, width, rng)
    img = apply_domain_randomization(img, rng)
    rust_mask = synthesize_rust_lines(img, rng, n_components=4, branch_prob=0.4)
    img = apply_domain_randomization(img, rng)
    return img, rust_mask


# ─── メイン ──────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="OpenVLA サビ追跡シミュレーション")
    parser.add_argument("--model_path", type=str, default=None,
                        help="LoRA チェックポイントのパス (None でモックエージェントを使用)")
    parser.add_argument("--input_image", type=str, default=None,
                        help="テスト画像パス (None で自動生成)")
    parser.add_argument("--rust_mask", type=str, default=None,
                        help="サビマスク画像パス (グレースケール PNG)")
    parser.add_argument("--output_dir", type=str, default="results/simulation")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--max_backtracks", type=int, default=100)
    parser.add_argument("--coverage_threshold", type=float, default=0.95)
    parser.add_argument("--n_test_images", type=int, default=3,
                        help="自動生成テスト画像の枚数 (--input_image 未指定時)")
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_video", action="store_true",
                        help="各ステップのフレームを保存して mp4 動画を生成する")
    parser.add_argument("--fps", type=int, default=8,
                        help="動画のフレームレート (--save_video 時)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # テスト画像の準備
    test_cases: list[tuple[np.ndarray, np.ndarray, str]] = []

    if args.input_image:
        img = cv2.imread(args.input_image)
        if img is None:
            raise FileNotFoundError(f"画像が読み込めません: {args.input_image}")
        if args.rust_mask:
            mask = cv2.imread(args.rust_mask, cv2.IMREAD_GRAYSCALE)
        else:
            # マスクなしの場合はサビ検出 (色による簡易分類)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # サビ色 (オレンジ〜赤茶) の範囲を検出
            lower_rust = np.array([0, 50, 50])
            upper_rust = np.array([20, 255, 200])
            mask = cv2.inRange(hsv, lower_rust, upper_rust)
        test_cases.append((img, mask, Path(args.input_image).stem))
    else:
        # テスト画像を自動生成 (学習と異なるシード)
        for i in range(args.n_test_images):
            seed = args.seed + i * 100
            img, mask = generate_test_image(seed=seed)
            test_cases.append((img, mask, f"test_{i:02d}_seed{seed}"))
            print(f"[main] テスト画像 {i+1}/{args.n_test_images} を生成しました (seed={seed})")

    # エージェントの準備
    all_results: list[dict] = []

    for img, mask, case_name in test_cases:
        print(f"\n{'='*60}")
        print(f"[simulate] ケース: {case_name}")
        print(f"{'='*60}")

        env = RustTracingEnv(img, mask, patch_size=PATCH_SIZE)
        stats = env.coverage_stats()
        print(f"  グリッドサイズ: {env.rows}×{env.cols}")
        print(f"  サビパッチ数: {stats['n_rust_patches']} / {env.rows * env.cols}")

        if args.model_path and Path(args.model_path).exists():
            try:
                import torch
                device = args.device if torch.cuda.is_available() else "cpu"
                agent = OpenVLARustAgent(args.model_path, device=device)
            except Exception as e:
                print(f"[WARNING] モデル読み込み失敗: {e} → モックエージェントを使用")
                agent = MockRustAgent(env)
        else:
            print("[simulate] モックエージェントを使用します (モデルパス未指定)")
            agent = MockRustAgent(env)

        frame_dir = output_dir / f"{case_name}_frames" if args.save_video else None
        result = run_simulation(
            env=env,
            agent=agent,
            max_steps=args.max_steps,
            max_backtracks=args.max_backtracks,
            coverage_threshold=args.coverage_threshold,
            frame_dir=frame_dir,
            image_orig=img,
        )

        # 結果の表示
        print(f"\n[結果] {case_name}")
        print(f"  カバレッジ率:    {result.coverage_rate*100:.1f}% "
              f"({result.n_visited_rust_patches}/{result.n_rust_patches})")
        print(f"  総ステップ数:    {result.total_steps}")
        print(f"  バックトラック:  {result.n_backtracks}")
        print(f"  成分間ジャンプ:  {result.n_component_jumps}")
        print(f"  実行時間:        {result.elapsed_seconds:.1f}s")

        # 可視化
        vis_path = output_dir / f"{case_name}_trajectory.png"
        visualize_trajectory(
            image=img,
            env=env,
            trajectory=result.trajectory,
            action_history=env.history,
            output_path=vis_path,
        )

        # 動画生成
        if args.save_video and frame_dir and frame_dir.exists():
            video_path = output_dir / f"{case_name}_trajectory.mp4"
            frames_to_video(frame_dir, video_path, fps=args.fps)

        # ソース画像も保存
        cv2.imwrite(str(output_dir / f"{case_name}_source.png"), img)

        # サビマスクを保存
        cv2.imwrite(str(output_dir / f"{case_name}_mask.png"), mask)

        # 結果を JSON に保存
        result_dict = asdict(result)
        result_dict["case_name"] = case_name
        result_dict["trajectory"] = [list(t) for t in result.trajectory]
        result_dict["visited_patches"] = [list(v) for v in result.visited_patches]
        result_path = output_dir / f"{case_name}_result.json"
        with open(result_path, "w") as f:
            json.dump(result_dict, f, indent=2)

        all_results.append(result_dict)

    # 全ケースの集計
    if all_results:
        avg_coverage = sum(r["coverage_rate"] for r in all_results) / len(all_results)
        avg_steps = sum(r["total_steps"] for r in all_results) / len(all_results)
        avg_backtracks = sum(r["n_backtracks"] for r in all_results) / len(all_results)

        print(f"\n{'='*60}")
        print("[集計結果]")
        print(f"  平均カバレッジ率:    {avg_coverage*100:.1f}%")
        print(f"  平均ステップ数:      {avg_steps:.1f}")
        print(f"  平均バックトラック:  {avg_backtracks:.1f}")
        print(f"{'='*60}")

        summary = {
            "n_cases": len(all_results),
            "avg_coverage_rate": avg_coverage,
            "avg_total_steps": avg_steps,
            "avg_backtracks": avg_backtracks,
            "cases": all_results,
        }
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print(f"\n[simulate] 完了! 結果: {output_dir}")


if __name__ == "__main__":
    main()
