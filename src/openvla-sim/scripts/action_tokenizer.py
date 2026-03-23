"""
ActionTokenizer ― OpenVLA 互換の 256bin 離散アクショントークン化

OpenVLA の仕様:
  - 各アクション次元を独立に 256 段階に量子化
  - データセットの min/max から bin 境界を計算
  - 整数 0〜255 をゼロ埋め 3 桁文字列 ("000"〜"255") で表現
  - 統計は action_stats.npz に保存 / 読み込み

使い方:
  # 学習時: データセットから統計を計算して保存
  tokenizer = ActionTokenizer.from_dataset(samples)
  tokenizer.save("checkpoints/best/action_stats.npz")
  action_str = tokenizer.encode([0.1, -0.2, 0.5, 0.0])  # → "145 109 182 127"

  # 推論時: 保存済み統計を読み込んで逆変換
  tokenizer = ActionTokenizer.load("checkpoints/best/action_stats.npz")
  action = tokenizer.decode("145 109 182 127")  # → [0.1, -0.2, 0.5, 0.0]
"""

from __future__ import annotations

import numpy as np


class ActionTokenizer:
    """連続値アクション ↔ 256bin 離散トークン の変換"""

    N_BINS = 256

    def __init__(self, action_min: np.ndarray, action_max: np.ndarray) -> None:
        self.action_min = np.array(action_min, dtype=np.float32)
        self.action_max = np.array(action_max, dtype=np.float32)
        # 値域が狭い次元はゼロ除算を回避
        self.action_range = np.where(
            (self.action_max - self.action_min) < 1e-6,
            1.0,
            self.action_max - self.action_min,
        )

    # ── エンコード ────────────────────────────────────────────────────────
    def encode(self, action: list | np.ndarray) -> str:
        """連続値 → 256bin トークン文字列 (例: "145 109 182 127")"""
        action = np.array(action, dtype=np.float32)
        normalized = (action - self.action_min) / self.action_range
        normalized = np.clip(normalized, 0.0, 1.0)
        bins = (normalized * (self.N_BINS - 1)).round().astype(int)
        return " ".join(f"{b:03d}" for b in bins)

    # ── デコード ────────────────────────────────────────────────────────
    def decode(self, token_str: str) -> np.ndarray:
        """256bin トークン文字列 → 連続値アクション"""
        dim = len(self.action_min)
        parts = token_str.strip().split()
        # 末尾から dim 個の 3 桁整数を探す
        bins = []
        for t in reversed(parts):
            try:
                v = int(t)
                if 0 <= v <= 255:
                    bins.insert(0, v)
                    if len(bins) == dim:
                        break
            except ValueError:
                if bins:
                    break
        # 見つからなかった次元はゼロ (中央 bin)
        while len(bins) < dim:
            bins.append(self.N_BINS // 2)

        bins = np.array(bins[:dim], dtype=np.float32)
        normalized = bins / (self.N_BINS - 1)
        return normalized * self.action_range + self.action_min

    # ── 保存 / 読み込み ──────────────────────────────────────────────────
    def save(self, path: str) -> None:
        np.savez(path, action_min=self.action_min, action_max=self.action_max)

    @classmethod
    def load(cls, path: str) -> "ActionTokenizer":
        data = np.load(path)
        return cls(data["action_min"], data["action_max"])

    # ── データセットから統計を計算 ────────────────────────────────────────
    @classmethod
    def from_dataset(cls, actions: list[list] | np.ndarray) -> "ActionTokenizer":
        """アクション配列 (N, D) から min/max を計算して ActionTokenizer を生成"""
        arr = np.array(actions, dtype=np.float32)
        return cls(arr.min(axis=0), arr.max(axis=0))

    def __repr__(self) -> str:
        return (
            f"ActionTokenizer(dim={len(self.action_min)}, "
            f"min={self.action_min.tolist()}, max={self.action_max.tolist()})"
        )
