#!/bin/bash
#SBATCH -J openvla-rust
#SBATCH -p debug
#SBATCH --gres=gpu:2
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -o logs/slurm-%j.out


echo "JOB_ID=$SLURM_JOB_ID"
hostname
nvidia-smi

# ─── 作業ディレクトリ ────────────────────────────────────────────────────────
cd /home/team-002/openvla

# ─── ログディレクトリ作成 ─────────────────────────────────────────────────────
mkdir -p logs
mkdir -p checkpoints/rust_openvla

# ─── 仮想環境のセットアップ ───────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  python -m venv .venv
fi
source .venv/bin/activate

# ─── 依存パッケージのインストール ─────────────────────────────────────────────
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
pip install "transformers==4.45.2" peft accelerate trl tensorflow opencv-python "timm>=0.9.10,<1.0.0" Pillow --quiet

# ─── GPU 1 のみ使用（GPU 0 は他プロセスで占有されているため）───────────────────
export CUDA_VISIBLE_DEVICES=1

# ─── 学習実行 ─────────────────────────────────────────────────────────────────
torchrun --nproc_per_node=1 \
  openvla-rust-tracing/training/train.py \
  --data_dir data/rust_rlds \
  --output_dir checkpoints/rust_openvla \
  --model_name_or_path openvla/openvla-7b \
  --lora_rank 8 \
  --bf16 \
  --history_len 1 \
  --num_epochs 5 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --wandb_project rust_openvla
