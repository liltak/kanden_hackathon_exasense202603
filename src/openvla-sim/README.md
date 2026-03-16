# openvla-sim

Genesis シミュレーターを使ったドローン VLA（Vision-Language-Action）の学習・推論パイプライン。
**OpenVLA 7B LoRA ファインチューニング**でドローンナビゲーションを学習する。

## セットアップ（H100 / Ubuntu）

### 1. venv を作成してアクティベート

```bash
cd ~/projects/kanden_hackathon_exasense/src/openvla-sim

python3 -m venv .venv
source .venv/bin/activate
```

### 2. PyTorch（CUDA対応版）をインストール

```bash
# CUDA 12.8 の場合（H100 推奨）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# CUDA バージョンを確認したい場合
nvcc --version
```

### 3. ライブラリをインストール

```bash
# Genesis 依存 + OpenVLA 学習用
pip install numpy scipy Pillow
pip install transformers peft accelerate wandb
pip install -e third_party/Genesis
```

### 4. 動作確認

```bash
python -c "import torch; print(torch.cuda.is_available())"  # True になること
python -c "import genesis"                                    # エラーなければOK
python -c "import transformers; print(transformers.__version__)"
```

---

## 構成

```
scripts/
  collect.py   # データ収集（Mac / H100）
  train.py     # OpenVLA 7B LoRA ファインチューニング（H100必須）
  infer.py     # 推論・自律飛行確認（H100必須）
objects/       # 3Dオブジェクト (.glb)
third_party/   # Genesis シミュレーター
```

## アクション形式

7次元・OpenVLA 互換（先頭4次元がドローン制御）：

| インデックス | 内容 | 単位 |
|---|---|---|
| 0 `vx_body` | 機首方向の速度（前進） | m/s |
| 1 `vy_body` | 機体左方向の速度 | m/s |
| 2 `vz_body` | 上方向の速度 | m/s |
| 3 `yaw_rate` | ヨー角速度（回転） | rad/s |
| 4〜6 | ゼロ埋め（OpenVLA 互換用） | - |

## 実行手順

```bash
# 1. データ収集（Mac でも実行可能）
python scripts/collect.py --episodes 500 --out scripts/dataset/

# 2. OpenVLA 7B LoRA ファインチューニング（H100）
torchrun --nproc_per_node=1 scripts/train.py \
  --data scripts/dataset/ \
  --out scripts/checkpoints/drone_openvla \
  --epochs 5 \
  --lora_rank 8 \
  --bf16

# 3. 推論・自律飛行確認（H100）
python scripts/infer.py \
  --ckpt_dir scripts/checkpoints/drone_openvla/best \
  --instruction "ソファに近づけ"
```

---

## H100 から手元PCに映像を映す方法

Genesis のビューアは OpenGL を使うため、H100 サーバーから手元 PC に映像を転送する必要がある。
用途に応じて以下の3つの方法から選ぶ。

---

### 方法1：X11 フォワーディング（最もシンプル）

SSH 接続時に `-X` または `-Y` フラグを付けるだけで、GUIウィンドウがそのまま手元PCに表示される。

```bash
# 手元PCのターミナルから接続
ssh -X h100
# または（信頼済みサーバーなら -Y の方が高速）
ssh -Y h100

# 接続後、通常通り実行
python scripts/infer.py \
  --ckpt_dir scripts/checkpoints/drone_openvla/best \
  --instruction "ソファに近づけ"
```

**メリット**: 設定不要、コード変更なし
**デメリット**: ネットワーク遅延で重い場合がある。Macの場合は [XQuartz](https://www.xquartz.org/) のインストールが必要。

```bash
# Mac の場合、事前に XQuartz をインストール
brew install --cask xquartz
# インストール後、一度ログアウト・ログインしてから ssh -X で接続
```

---

### 方法2：動画ファイルとして保存 → scp で転送（推奨）

ヘッドレスでレンダリングし、MP4として保存してから手元PCに転送する。

```python
# scene の show_viewer=False に変更
scene = gs.Scene(
    ...
    show_viewer=False,   # ← headless
    ...
)

# FPVカメラの記録を開始
fpv_cam.start_recording()

# メインループ（既存コードのまま）
while True:
    ...
    scene.step()

# 終了時に保存
fpv_cam.stop_recording(save_to="output.mp4", fps=30)
```

```bash
# 手元PCのターミナルから転送
scp h100:~/projects/kanden_hackathon_exasense/src/openvla-sim/scripts/output.mp4 ~/Downloads/
```

**メリット**: 品質が高い、ネットワーク帯域を消費しない
**デメリット**: リアルタイムに見えない（事後確認）

---

### 方法3：VNC（リアルタイム・高品質）

```bash
# H100 側：VNC サーバーを起動（初回のみ設定）
vncserver :1 -geometry 1920x1080 -depth 24

# 手元PC側：SSH ポートフォワードでトンネリング
ssh -L 5901:localhost:5901 h100

# Mac: Finder → 移動 → サーバへ接続 → vnc://localhost:5901
```

---

### 方法の選び方

| 状況 | 推奨方法 |
|---|---|
| とりあえず動作確認したい | **方法1**（X11フォワーディング） |
| 学習後の結果を動画で残したい | **方法2**（動画保存） |
| リアルタイムで操作しながら確認したい | **方法3**（VNC） |

---

## 学習 loss の確認方法

### 方法1：ターミナル出力（追加設定なし）

train.py は **10ステップごと**に loss を表示する：

```
Epoch 1/5 | Step 10 | Loss: 2.3451 | LR: 2.00e-04
Epoch 1/5 | Step 20 | Loss: 1.8923 | LR: 1.98e-04
...
[Epoch 1] Val Loss: 1.7234
✓ Best model saved (val_loss=1.7234)
```

```bash
# バックグラウンド実行しながらログを確認
nohup torchrun --nproc_per_node=1 scripts/train.py \
  --data scripts/dataset/ \
  --out scripts/checkpoints/drone_openvla \
  --epochs 5 --lora_rank 8 --bf16 > train.log 2>&1 &

tail -f train.log
```

---

### 方法2：wandb（クラウドで確認・最も手軽）

train.py は wandb を標準サポート。

```bash
pip install wandb
wandb login  # 初回のみ：APIキーを入力（https://wandb.ai/authorize で取得）
```

実行時に自動でログが送られる：

```bash
torchrun --nproc_per_node=1 scripts/train.py \
  --data scripts/dataset/ \
  --out scripts/checkpoints/drone_openvla \
  --epochs 5 --lora_rank 8 --bf16 \
  --wandb_project drone_openvla
```

SSHトンネル不要でそのままブラウザの [wandb.ai](https://wandb.ai) で確認できる。

---

### 方法の選び方

| 状況 | 推奨方法 |
|---|---|
| 設定なしでとりあえず確認 | **方法1**（ターミナル / tail -f） |
| どこからでも確認したい | **方法2**（wandb） |
