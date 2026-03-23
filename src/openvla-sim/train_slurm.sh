#!/bin/bash
#SBATCH -J drone-openvla
#SBATCH -p debug
#SBATCH --gres=gpu:2
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err

echo "JOB_ID=$SLURM_JOB_ID"
hostname
nvidia-smi

# ─── 作業ディレクトリ ──────────────────────────────────────────────────────────
cd /home/team-002/openvla-sim

# ─── ログ・出力ディレクトリ作成 ───────────────────────────────────────────────
mkdir -p logs
mkdir -p checkpoints_v3/drone_openvla

# ─── 仮想環境のセットアップ ────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  python -m venv .venv
fi
source .venv/bin/activate

# ─── 依存パッケージのインストール ─────────────────────────────────────────────
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
pip install "transformers==4.44.0" peft accelerate tensorboard Pillow scipy "timm>=0.9.10,<1.0.0" --quiet
pip install -e third_party/Genesis --quiet

# ─── GPU 設定（2台使用）────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0,1

# ─── 学習パラメータ ────────────────────────────────────────────────────────────
DATASET="dataset_v2"
CHECKPOINT="checkpoints_v3/drone_openvla"
MODEL="openvla/openvla-7b"
EPOCHS=15
LORA_RANK=32
BATCH_SIZE=16
GRAD_ACCUM=1
LR=5e-4

# ─── 学習実行 ─────────────────────────────────────────────────────────────────
torchrun --nproc_per_node=1 \
  openvla-sim/scripts/train.py \
  --data       "$DATASET" \
  --out        "$CHECKPOINT" \
  --model      "$MODEL" \
  --epochs     $EPOCHS \
  --lora_rank  $LORA_RANK \
  --batch_size $BATCH_SIZE \
  --grad_accum $GRAD_ACCUM \
  --lr         $LR \
  --bf16
