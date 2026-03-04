#!/bin/bash
#SBATCH -J openvla-rust-infer
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o logs/slurm-infer-%j.out


echo "JOB_ID=$SLURM_JOB_ID"
hostname
nvidia-smi

# ─── 作業ディレクトリ ────────────────────────────────────────────────────────
cd /home/team-002/openvla

# ─── ログ・出力ディレクトリ作成 ──────────────────────────────────────────────
mkdir -p logs
mkdir -p results/simulation

# ─── 仮想環境のセットアップ ───────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  python -m venv .venv
fi
source .venv/bin/activate

# ─── 依存パッケージのインストール ─────────────────────────────────────────────
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
pip install "transformers==4.45.2" peft accelerate "timm>=0.9.10,<1.0.0" opencv-python Pillow --quiet

# ─── 推論・評価実行 ───────────────────────────────────────────────────────────
python openvla-rust-tracing/evaluation/simulate.py \
  --model_path checkpoints/rust_openvla/best \
  --output_dir results/simulation \
  --n_test_images 3 \
  --max_steps 200 \
  --save_video \
  --fps 8 \
  --device cuda
