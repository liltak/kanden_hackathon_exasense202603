#!/bin/bash
#SBATCH -J crack-openvla-train
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -o logs/slurm-train-%j.out

echo "JOB_ID=$SLURM_JOB_ID"
hostname
nvidia-smi

# ─── 作業ディレクトリ ─────────────────────────────────────────────────────────
# SLURM_SUBMIT_DIR = sbatch を実行したディレクトリ（スプールディレクトリではない）
cd "$SLURM_SUBMIT_DIR"
echo "WORKDIR=$SLURM_SUBMIT_DIR"

# ─── ログ・チェックポイントディレクトリ作成 ──────────────────────────────────
mkdir -p logs
mkdir -p checkpoints/crack_openvla

# ─── 仮想環境のセットアップ ───────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
echo "Python: $(which python3)"
echo "pip:    $(which pip)"

# ─── 依存パッケージのインストール ─────────────────────────────────────────────
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
pip install "transformers==4.45.2" peft accelerate tensorboard \
            "timm>=0.9.10,<1.0.0" Pillow numpy --quiet

# ─── GPU 設定 ─────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0

# ─── 学習実行 ─────────────────────────────────────────────────────────────────
.venv/bin/torchrun --nproc_per_node=1 \
  training/train.py \
  --data   data/auto_raw \
  --out    checkpoints/crack_openvla \
  --model  openvla/openvla-7b \
  --epochs     5 \
  --lora_rank 32 \
  --batch_size 16 \
  --grad_accum  1 \
  --lr     5e-4 \
  --bf16

echo "学習完了: $(date)"
