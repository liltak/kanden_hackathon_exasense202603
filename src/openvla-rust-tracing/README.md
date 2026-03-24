# OpenVLA Crack Line Tracing Agent

大規模インフラ画像（コンクリート壁・石材・アスファルト等）においてクラック（ひび割れ）の経路を自律的に追従する VLA エージェント。
OpenVLA (7B) を LoRA ファインチューニングし、224×224 パッチを逐次観察しながら次の移動方向を予測する。

---

## 概要

| 項目 | 内容 |
|------|------|
| **タスク** | クラック経路の自律追従（始点から終点まで） |
| **ベースモデル** | OpenVLA 7B (Vision-Language-Action model) |
| **ファインチューニング** | LoRA (rank=32) |
| **入力** | 224×224 パッチ画像 + 命令文 |
| **出力** | [Δx, Δy] 移動ベクトル（256 bin 離散トークン） |
| **訓練データ** | 合成生成クラック画像（コンクリート・石材・木目・アスファルト） |

---

## 開発環境と実行可能範囲

| スクリプト | Mac ローカル | H100 (Linux) | 必要な追加インストール |
|-----------|:-----------:|:------------:|----------------------|
| `generate_crack.py` | ✅ | ✅ | `pillow numpy` |
| `annotate.py`       | ✅ | ✅ | `pillow numpy opencv-python` |
| `convert_to_rlds.py`| ✅ | ✅ | Mac: `tensorflow-macos tensorflow-metal` / Linux: `tensorflow` |
| `train.py`          | ❌ | ✅ | `torch transformers peft accelerate tensorboard` |
| `infer.py`          | ❌ | ✅ | 上記 + `torch transformers peft` |
| `rollout.py`        | ❌ | ✅ | 上記 + `opencv-python` |

**Mac で学習・推論が動かない理由**
OpenVLA 7B は bf16 で約 14GB VRAM が必要。Mac の MPS では速度・メモリ両面で非現実的。

---

## 推奨開発フロー

```
[Mac ローカル]                              [H100]
─────────────────────────────────           ──────────────────────────────
Step 1: クラック画像生成                     Step 3: LoRA ファインチューニング
  generate_crack.py              →  転送  →  train.py
                                                    ↓ チェックポイント
Step 2: アノテーション確認 (任意)            Step 4: 推論 / ロールアウト
  visualize_annotation.py        ←  転送  ←  infer.py / rollout.py
```

---

## セットアップ

### Mac ローカル

```bash
pip install pillow numpy opencv-python

# TFRecord 変換も Mac で行う場合 (Apple Silicon)
pip install tensorflow-macos tensorflow-metal
```

### H100 (Linux)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft accelerate tensorboard
pip install tensorflow pillow numpy opencv-python
```

---

## 実行手順

### Step 1: クラック画像生成（Mac / H100）

```bash
# デフォルト: 100エピソード
python src/openvla-rust-tracing/data_generation/generate_crack.py

# エピソード数・シードを指定
python src/openvla-rust-tracing/data_generation/generate_crack.py \
  --n 300 \
  --seed 42 \
  --out data/crack_generated

# 実背景画像を使用する場合（images/ フォルダ内の jpg/png を利用）
python src/openvla-rust-tracing/data_generation/generate_crack.py \
  --n 100 \
  --bg_dir src/openvla-rust-tracing/images \
  --out data/crack_real_bg
```

出力ディレクトリ構造:
```
data/crack_generated/
  ├── 0000_concrete_none_w2.5.png     # 合成クラック画像 (4096×4096)
  ├── 0001_stone_wave_w3.0.png
  ├── ...
  └── annotations/
        ├── steps/
        │   ├── step_000000.png       # 224×224 パッチ画像
        │   └── ...
        └── episode_0000.json         # 各ステップの座標・アクション・命令文
```

生成される画像のバリエーション:
| テクスチャ | 変形モード | 線幅 |
|-----------|----------|------|
| `concrete`, `stone`, `wood`, `asphalt` | `none`, `wave`, `zigzag`, `bend` | 2.5〜4.0px (スケール込み) |

### Step 2: アノテーション確認（Mac / H100、任意）

```bash
# 指定エピソードのアノテーションをオーバーレイ表示
python src/openvla-rust-tracing/data_generation/visualize_annotation.py \
  --data_dir data/crack_generated/annotations \
  --episode_id 0
```

### Step 3: LoRA ファインチューニング（H100 専用）

```bash
torchrun --nproc_per_node=1 \
  src/openvla-rust-tracing/training/train.py \
  --data data/crack_generated/annotations \
  --out checkpoints/crack_openvla \
  --model openvla/openvla-7b \
  --epochs 5 \
  --lora_rank 32 \
  --batch_size 16 \
  --lr 5e-4 \
  --bf16
```

主なオプション:

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--lora_rank` | 32 | LoRA のランク数 |
| `--epochs` | 5 | エポック数 |
| `--batch_size` | 16 | バッチサイズ |
| `--grad_accum` | 1 | 勾配累積ステップ数 |
| `--lr` | 5e-4 | 学習率 |
| `--bf16` | off | bfloat16 で訓練（H100 推奨） |

データ分割は train 80% / val 10% / test 10%（エピソード単位・seed固定）。

### Step 4: 推論・ロールアウト（H100）

```bash
# 単一パッチ推論
python src/openvla-rust-tracing/training/infer.py \
  --model_path checkpoints/crack_openvla/best \
  --image path/to/patch.png

# エピソード全体のロールアウト評価
python src/openvla-rust-tracing/training/rollout.py \
  --model_path checkpoints/crack_openvla/best \
  --data data/crack_generated/annotations \
  --output_dir results/rollout
```

---

## アーキテクチャ

```
大規模インフラ画像 (4096×4096)
    │
    ▼ クラック検出 (生成時の正確なパス or 外部セグメンテーター)
クラック経路座標列 [(x0,y0), (x1,y1), ...]
    │
    ▼ 弧長に基づく等間隔サブサンプリング (50% オーバーラップ)
ウェイポイント列
    │
    ▼ 224×224 パッチクロップ (ゼロパディング込み)
パッチ画像 + 命令文 "Follow the crack. Navigate to continue tracking the crack path."
    │
    ▼ OpenVLA 7B + LoRA (H100, bf16)
連続 2D ベクトル [Δx, Δy] → ActionTokenizer で 256bin 離散化
    │
    ▼ 次のパッチ中心座標 = 現在座標 + (Δx, Δy)
次ステップへ
```

### ActionTokenizer

アクション [Δx, Δy] を 256 個の均一ビンに離散化して文字列トークンとして出力する。
統計（平均・標準偏差）は訓練エピソードから自動計算し、`action_stats.npz` として保存。

---

## データ生成の詳細

### クラック形状のバリエーション

| パラメータ | 説明 |
|-----------|------|
| `deform=none` | 緩やかなカーブのみ（最大 45°方向変化） |
| `deform=wave` | サイン波状の蛇行 |
| `deform=zigzag` | ランダムなジグザグ |
| `deform=bend` | 途中で一回大きく折れ曲がる |

### 描画アルゴリズム

リアルなひび割れ表現のため 3 層重ね描画を採用：
1. **影レイヤー**: 広め・薄い灰色 → ガウシアンブラーでふんわり
2. **芯レイヤー**: 中幅・暗い → 中ブラーでエッジを柔らかく
3. **中心線**: 細い・最も暗い → 軽くブラーして馴染ませる

その後、ガウスノイズを加えてリアリティを向上。

---

## 評価指標

| 指標 | 説明 |
|------|------|
| **経路追従精度 (Path Following Accuracy)** | 予測ウェイポイントと正解ウェイポイントの平均 L2 誤差（px） |
| **方向誤差 (Angular Error)** | 予測移動方向と正解方向の角度差（度） |
| **完走率 (Episode Completion Rate)** | エピソード終点まで到達できた割合 |
| **ドリフト距離** | クラック本線からの平均逸脱距離（px） |
| **Validation Loss** | ActionTokenizer 離散化後のクロスエントロピー損失 |

---

## 技術的注意点

- OpenVLA のアクションヘッドは**改造しない**。2D アクションはすべて自然言語トークン列として出力。
- アクション離散化は **ActionTokenizer** が担当。統計は train セットのみから計算し、val/test にはリークしない。
- データ分割は**エピソード単位**（ステップ単位ではない）で行い、時系列リークを防止。
- `torch` / `transformers` / `peft` は `try/except` + `TYPE_CHECKING` で保護し、Mac でもクラッシュしない。
- 実背景画像（`--bg_dir`）を使用する場合は著作権フリー素材を利用。

---

## ファイル構成

```
openvla-rust-tracing/
├── README.md
├── PAPER.md                         # 論文・関連研究ドラフト
├── data_generation/
│   ├── generate_crack.py            # 合成クラック画像 + アノテーション生成
│   ├── annotate.py                  # 手動アノテーション補助
│   ├── auto_annotate.py             # 自動アノテーション
│   ├── visualize_annotation.py      # アノテーション可視化
│   └── convert_to_rlds.py           # RLDS (TFRecord) 形式へ変換
├── training/
│   ├── train.py                     # LoRA ファインチューニング
│   ├── infer.py                     # 単一パッチ推論
│   ├── rollout.py                   # エピソード全体のロールアウト
│   ├── action_tokenizer.py          # 256bin 離散化
│   └── pick_start.py                # 追従開始点の選択
├── images/                          # 実クラック画像サンプル
├── background/                      # 背景テクスチャサンプル
├── results/rollout/                 # ロールアウト結果
├── generate_data_slurm.sh           # Slurm: データ生成ジョブ
├── train_slurm.sh                   # Slurm: 学習ジョブ
├── infer_slurm.sh                   # Slurm: 推論ジョブ
└── rollout_slurm.sh                 # Slurm: ロールアウトジョブ
```
