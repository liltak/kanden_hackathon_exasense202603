# OpenVLA Rust Line Tracing Agent

特大インフラ画像（コンクリート壁・鉄板等）を格子状に分割し、
OpenVLA (7B) を用いてサビの経路を自律的に辿るエージェント。

---

## 開発環境と実行可能範囲

各スクリプトがどの環境で動くかを以下に示す。

| スクリプト | Mac ローカル | H100 (Linux) | 必要な追加インストール |
|-----------|:-----------:|:------------:|----------------------|
| `generate_dataset.py` | ✅ | ✅ | `opencv-python numpy` |
| `convert_to_rlds.py`  | ✅ | ✅ | Mac: `tensorflow-macos tensorflow-metal` / Linux: `tensorflow` |
| `train.py`            | ❌ | ✅ | `torch transformers peft accelerate wandb trl tensorflow` |
| `simulate.py` (モックエージェント) | ✅ | ✅ | `opencv-python numpy` |
| `simulate.py` (実モデル)           | ❌ | ✅ | 上記 + `torch transformers peft` |

**Mac で動かない理由（`train.py` / 実モデル推論）**
OpenVLA 7B は bf16 で約 14GB VRAM が必要。Mac の MPS では速度・メモリ両面で非現実的。

---

## 推奨開発フロー

```
[Mac ローカル]                          [H100]
─────────────────────────────────       ──────────────────────────────
Step 1: データ生成                       Step 3: LoRA ファインチューニング
  generate_dataset.py           →  データを転送  →  train.py
                                                         ↓ チェックポイント
Step 2: TFRecord 変換             ←  チェックポイントを転送
  convert_to_rlds.py

Step 4a: パイプライン動作確認            Step 4b: 本評価
  simulate.py (モック)                     simulate.py --model_path ...
```

---

## セットアップ

### Mac ローカル

```bash
# プロジェクトルートで実行
pip install opencv-python numpy

# TFRecord 変換も Mac で行う場合 (Apple Silicon)
pip install tensorflow-macos tensorflow-metal
```

### H100 (Linux)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft accelerate wandb trl
pip install tensorflow opencv-python numpy
```

---

## 実行手順

### Step 1: データ生成（Mac / H100）

```bash
python src/openvla-rust-tracing/data_generation/generate_dataset.py \
  --output_dir data/rust_dataset \
  --n_episodes 50 \
  --image_size 1120 1120 \
  --n_rust_components 3 \
  --seed 42
```

出力ディレクトリ:
```
data/rust_dataset/
  steps/         # step_000000.png ... (224×224 パッチ画像)
  sources/       # episode_0000_source.png ... (元の特大画像)
  episodes/      # episode_0000.json ...
  metadata.json  # 全ステップのメタデータ (アクション・instruction 含む)
```

### Step 2: TFRecord 変換（Mac / H100）

```bash
# Mac の場合
pip install tensorflow-macos tensorflow-metal

# Linux / H100 の場合
pip install tensorflow

python src/openvla-rust-tracing/data_generation/convert_to_rlds.py \
  --input_dir data/rust_dataset \
  --output_dir data/rust_rlds \
  --split_ratio 0.9

# 変換済みファイルの検証のみ実行
python src/openvla-rust-tracing/data_generation/convert_to_rlds.py \
  --test --output_dir data/rust_rlds
```

### Step 3: LoRA ファインチューニング（H100 専用）

```bash
# H100 上で実行
torchrun --nproc_per_node=1 \
  src/openvla-rust-tracing/training/train.py \
  --data_dir data/rust_rlds \
  --output_dir checkpoints/rust_openvla \
  --model_name_or_path openvla/openvla-7b \
  --lora_rank 16 \
  --bf16 \
  --history_len 3 \
  --use_minimap \
  --num_epochs 5 \
  --wandb_project rust_openvla
```

主なオプション:

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--lora_rank` | 16 | LoRA のランク数 |
| `--history_len` | 3 | 参照する過去フレーム数 (3〜5) |
| `--use_minimap` | off | 探索ミニマップをパッチにオーバーレイ |
| `--bf16` | on | bfloat16 で訓練 (H100 推奨) |
| `--num_epochs` | 5 | エポック数 |

### Step 4a: パイプライン動作確認（Mac / モックエージェント）

モデルなしでパイプライン全体・可視化を確認できる。

```bash
python src/openvla-rust-tracing/evaluation/simulate.py \
  --output_dir results/simulation \
  --n_test_images 3 \
  --max_steps 200
```

### Step 4b: 本評価（H100 / 学習済みモデル）

```bash
# H100 上で実行
python src/openvla-rust-tracing/evaluation/simulate.py \
  --model_path checkpoints/rust_openvla/best \
  --output_dir results/simulation \
  --n_test_images 5 \
  --max_steps 500 \
  --coverage_threshold 0.95

# 特定の画像を指定する場合
python src/openvla-rust-tracing/evaluation/simulate.py \
  --model_path checkpoints/rust_openvla/best \
  --input_image path/to/test_image.png \
  --output_dir results/simulation
```

---

## 評価指標

| 指標 | 説明 |
|------|------|
| **カバレッジ率** | 全サビパッチのうち訪問できた割合 |
| **総ステップ数** | エピソード完了までのアクション数 |
| **バックトラック回数** | 行き止まりから引き返した回数 |
| **成分間ジャンプ数** | 途切れたサビ成分間を移動した回数 |

---

## アーキテクチャ

```
特大画像 (1120×1120 等)
    ↓ 格子分割 (224×224 パッチ)
各パッチ + instruction (直近 3〜5 フレームの履歴付き)
    ↓ OpenVLA 7B + LoRA (H100)
3D アクションベクトル [x, y, z]
    ↓ ユークリッド距離 最近傍離散化 (z > 0.5 → backtrack)
9方向アクション
    ↓ 連結成分ベース二段構え探索
  ① 成分内: DFS (バックトラック対応)
  ② 成分間: 最寄りの未探索成分へジャンプ
```

## アクション設計

| アクション | 3D ベクトル (x, y, z) | グリッド移動 (dr, dc) |
|-----------|----------------------|---------------------|
| up | (0, 1, 0) | (-1, 0) |
| down | (0, -1, 0) | (+1, 0) |
| left | (-1, 0, 0) | (0, -1) |
| right | (1, 0, 0) | (0, +1) |
| upper_right | (0.707, 0.707, 0) | (-1, +1) |
| upper_left | (-0.707, 0.707, 0) | (-1, -1) |
| lower_right | (0.707, -0.707, 0) | (+1, +1) |
| lower_left | (-0.707, -0.707, 0) | (+1, -1) |
| **backtrack** | **(0, 0, 1)** | (0, 0) ← z=1 専用 |

z 軸をバックトラック専用フラグとして使用し、通常移動との誤判定を防ぐ。

---

## 技術的な注意点

- OpenVLA のアクションヘッドは**改造しない**。アクションはすべて連続 3D ベクトルとして扱う。
- 「戻る」は z=1 で表現。通常移動は z=0。推論時に z > 0.5 のしきい値でバックトラックを判定。
- 純粋な DFS はサビの途切れに対応できないため、連結成分ベースの二段構え探索を採用。
- `torch` / `transformers` / `peft` は `try/except` または `TYPE_CHECKING` で保護しており、Mac でインポートしてもクラッシュしない。
