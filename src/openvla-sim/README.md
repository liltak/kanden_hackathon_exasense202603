# openvla-sim

Genesis シミュレーターを使ったドローン VLA（Vision-Language-Action）の学習・推論パイプライン。
**OpenVLA 7B LoRA ファインチューニング**で自然言語命令によるドローンナビゲーションを学習する。

## 概要

室内環境に配置された3Dオブジェクト（ソファ・アームチェア・引き出し）を Genesis でシミュレーションし、
FPVカメラ画像と自然言語命令（例：「ソファに近づけ」）からドローン制御アクションを出力するモデルを学習する。

```
collect.py → train.py → infer.py
 データ収集    LoRAファインチューニング  自律飛行推論
```

アクション形式（7次元・OpenVLA 互換）：

| インデックス | 内容 | 単位 |
|---|---|---|
| 0 `vx_body` | 機首方向の速度（前進） | m/s |
| 1 `vy_body` | 機体左方向の速度 | m/s |
| 2 `vz_body` | 上方向の速度 | m/s |
| 3 `yaw_rate` | ヨー角速度（回転） | rad/s |
| 4〜6 | ゼロ埋め（OpenVLA 互換用） | — |

---

## セットアップ

### 実行環境

| 項目 | バージョン |
|---|---|
| OS | Ubuntu 22.04 |
| Python | 3.10+ |
| CUDA | 12.1 |
| GPU | NVIDIA H100 (VRAM 80GB) |
| PyTorch | 2.x (CUDA 12.1 対応) |

### インストール

```bash
cd openvla-sim

# 1. venv を作成してアクティベート
python -m venv .venv
source .venv/bin/activate

# 2. PyTorch（CUDA 12.1 対応版）をインストール
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. 依存ライブラリをインストール
pip install "transformers==4.44.0" peft accelerate tensorboard Pillow scipy "timm>=0.9.10,<1.0.0"
pip install -e third_party/Genesis

# 4. 動作確認
python -c "import torch; print(torch.cuda.is_available())"  # True になること
python -c "import genesis"                                    # エラーなければOK
```

---

## 使用方法

### ディレクトリ構成

```
scripts/
  collect.py           # データ収集（Genesis シミュレーター）
  train.py             # OpenVLA 7B LoRA ファインチューニング（H100必須）
  infer.py             # 推論・自律飛行確認（H100必須）
  action_tokenizer.py  # アクションのトークナイザ
  convert_to_rlds.py   # RLDS形式への変換
objects/               # 3Dオブジェクト (.glb)
third_party/Genesis/   # Genesis シミュレーター
train_slurm.sh         # Slurm ジョブスクリプト（学習）
infer_slurm.sh         # Slurm ジョブスクリプト（推論）
```

### 1. データ収集

```bash
python scripts/collect.py --episodes 5000 --out dataset/
```

### 2. LoRA ファインチューニング（H100）

```bash
torchrun --nproc_per_node=1 scripts/train.py \
  --data       dataset/ \
  --out        checkpoints/drone_openvla \
  --model      openvla/openvla-7b \
  --epochs     15 \
  --lora_rank  32 \
  --batch_size 16 \
  --lr         5e-4 \
  --bf16
```

Slurm を使う場合：

```bash
sbatch train_slurm.sh
tail -f logs/slurm-<JOB_ID>.out
```

### 3. 推論・自律飛行確認

```bash
python scripts/infer.py \
  --ckpt_dir checkpoints/drone_openvla/best \
  --instruction "ソファに近づけ"
```

---

## 学習 loss の確認方法

### ターミナル出力（追加設定なし）

`train.py` は 10 ステップごとに loss を表示する：

```
Epoch 1/15 | Step 10 | Loss: 2.3451 | LR: 5.00e-04
...
[Epoch 1] Val Loss: 1.7234
✓ Best model saved (val_loss=1.7234)
```

```bash
# バックグラウンド実行しながら確認
nohup torchrun --nproc_per_node=1 scripts/train.py ... > train.log 2>&1 &
tail -f train.log
```

### tensorboard

```bash
tensorboard --logdir checkpoints/drone_openvla --port 6006
# SSH ポートフォワード: ssh -L 6006:localhost:6006 h100
```

---

## H100 から手元 PC に映像を映す方法

| 状況 | 方法 |
|---|---|
| とりあえず動作確認したい | X11 フォワーディング（`ssh -X h100`） |
| 学習後の結果を動画で残したい | ヘッドレスで MP4 保存 → `scp` で転送 |
| リアルタイムで確認したい | VNC（`ssh -L 5901:localhost:5901 h100`） |

---

## 使用データ

| データセット | 用途 | ライセンス | URL |
|---|---|---|---|
| Genesis シミュレーター自動生成データ（合成） | LoRA 学習データ（FPV画像 + アクション） | — | — |
| modern_arm_chair_01_4k.glb（Poly Haven） | 3Dシーン構築 | CC0 1.0 | https://polyhaven.com/a/modern_arm_chair_01 |
| sofa_02_4k.glb（Poly Haven） | 3Dシーン構築 | CC0 1.0 | https://polyhaven.com/a/sofa_02 |
| vintage_wooden_drawer_01_4k.glb（Poly Haven） | 3Dシーン構築 | CC0 1.0 | https://polyhaven.com/a/vintage_wooden_drawer_01 |

---

## 使用モデル

| モデル名 | 用途 | ライセンス | 利用規約 URL |
|---|---|---|---|
| openvla/openvla-7b | LoRA ファインチューニングのベースモデル | MIT License | https://huggingface.co/openvla/openvla-7b |
| Genesis（物理シミュレーター） | 学習データ生成・推論シミュレーション環境 | Apache 2.0 | https://github.com/Genesis-Embodied-AI/Genesis |

---

## ライセンス

MIT License
