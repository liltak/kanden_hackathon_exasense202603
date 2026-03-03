# OpenVLA Rust Line Tracing Agent

特大インフラ画像（コンクリート壁・鉄板等）を格子状に分割し、
OpenVLA (7B) を用いてサビの経路を自律的に辿るエージェント。

## アーキテクチャ

```
特大画像 (1120×1120 等)
    ↓ 格子分割 (224×224 パッチ)
各パッチ + instruction
    ↓ OpenVLA 7B + LoRA
3D アクションベクトル [x, y, z]
    ↓ ユークリッド距離 最近傍離散化
9方向アクション (up/down/left/right/diagonal × 4 + backtrack)
    ↓ 連結成分ベース二段構え探索
  ① 成分内: DFS (バックトラック対応)
  ② 成分間: 最寄りの未探索成分へジャンプ
```

## アクション設計

| アクション   | 3D ベクトル (x, y, z)       | グリッド移動 (dr, dc) |
|------------|--------------------------|---------------------|
| up         | (0, 1, 0)               | (-1, 0)             |
| down       | (0, -1, 0)              | (+1, 0)             |
| left       | (-1, 0, 0)              | (0, -1)             |
| right      | (1, 0, 0)               | (0, +1)             |
| upper_right| (0.707, 0.707, 0)       | (-1, +1)            |
| upper_left | (-0.707, 0.707, 0)      | (-1, -1)            |
| lower_right| (0.707, -0.707, 0)      | (+1, +1)            |
| lower_left | (-0.707, -0.707, 0)     | (+1, -1)            |
| **backtrack** | **(0, 0, 1)**        | (0, 0) ← z=1 専用   |

z 軸をバックトラック専用フラグとして使用し、通常移動との誤判定を防ぐ。

## ディレクトリ構成

```
src/openvla-rust-tracing/
├── data_generation/
│   ├── generate_dataset.py   # タスク1: サビ線合成 + 探索ログ生成
│   └── convert_to_rlds.py    # タスク2: RLDS/TFRecord 変換
├── training/
│   └── train.py              # タスク3: OpenVLA LoRA ファインチューニング
├── evaluation/
│   └── simulate.py           # タスク4: 推論 + 評価シミュレーション
└── README.md
```

## 実行手順

### Step 1: データ生成

```bash
# 50エピソードのデータセットを生成 (約5分)
python src/openvla-rust-tracing/data_generation/generate_dataset.py \
  --output_dir data/rust_dataset \
  --n_episodes 50 \
  --image_size 1120 1120 \
  --n_rust_components 3 \
  --seed 42
```

出力:
```
data/rust_dataset/
  steps/         # step_000000.png ... (224×224 パッチ画像)
  sources/       # episode_0000_source.png ... (元画像)
  episodes/      # episode_0000.json ...
  metadata.json  # 全ステップのメタデータ
```

### Step 2: TFRecord 変換

```bash
pip install tensorflow
python src/openvla-rust-tracing/data_generation/convert_to_rlds.py \
  --input_dir data/rust_dataset \
  --output_dir data/rust_rlds \
  --split_ratio 0.9
```

### Step 3: ファインチューニング (H100 推奨)

```bash
pip install transformers peft accelerate wandb trl
torchrun --nproc_per_node=1 \
  src/openvla-rust-tracing/training/train.py \
  --data_dir data/rust_rlds \
  --output_dir checkpoints/rust_openvla \
  --model_name_or_path openvla/openvla-7b \
  --lora_rank 16 \
  --bf16 \
  --history_len 3 \
  --use_minimap \
  --wandb_project rust_openvla
```

### Step 4: 評価シミュレーション

```bash
# モデルあり (学習済みチェックポイントを使用)
python src/openvla-rust-tracing/evaluation/simulate.py \
  --model_path checkpoints/rust_openvla/best \
  --output_dir results/simulation \
  --n_test_images 5 \
  --max_steps 500

# モデルなし (モックエージェントでデバッグ)
python src/openvla-rust-tracing/evaluation/simulate.py \
  --output_dir results/simulation \
  --n_test_images 3
```

## 評価指標

| 指標 | 説明 |
|------|------|
| **カバレッジ率** | 全サビパッチのうち訪問できた割合 |
| **総ステップ数** | エピソード完了までのアクション数 |
| **バックトラック回数** | 行き止まりから戻った回数 |
| **成分間ジャンプ数** | 別の連結成分へ移動した回数 |

## 技術的な注意点

- OpenVLA のアクションヘッドは**改造しない**。
  アクションはすべて連続 3D ベクトルとして扱う。
- 「戻る」は z=1 で表現。通常移動は z=0。
  推論時に z > 0.5 のしきい値でバックトラックを判定。
- 純粋な DFS はサビの途切れに対応できないため、
  連結成分ベースの二段構え探索を採用。
- GPU モジュール (torch, transformers) は `TYPE_CHECKING`
  ガード + 遅延インポートで macOS でも importable。
