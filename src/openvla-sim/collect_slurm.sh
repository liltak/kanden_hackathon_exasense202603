#!/bin/bash
#SBATCH -J drone-collect
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH -o logs/collect-%j.out
#SBATCH -e logs/collect-%j.err

echo "JOB_ID=$SLURM_JOB_ID"
hostname
nvidia-smi

# ─── 作業ディレクトリ ──────────────────────────────────────────────────────────
cd "$HOME/openvla-sim"

# ─── ログ・出力ディレクトリ作成 ───────────────────────────────────────────────
mkdir -p logs
mkdir -p dataset

# ─── 仮想環境のセットアップ ────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  python -m venv .venv
fi
source .venv/bin/activate

# ─── 依存パッケージのインストール ─────────────────────────────────────────────
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --quiet
pip install "transformers==4.44.0" peft accelerate tensorboard Pillow scipy "timm>=0.9.10,<1.0.0" --quiet
pip install -e openvla-sim/third_party/Genesis --quiet

# ─── GPU 設定 ─────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0

# ─── データ収集実行 ───────────────────────────────────────────────────────────
echo "------- Collect Start -------"
python openvla-sim/scripts/collect_v2.py \
  --episodes 300 \
  --max_steps 1000 \
  --img_size 224 \
  --hover_steps 5 \
  --out dataset_v2
EXIT_CODE=$?

echo "------- Collect End (exit=$EXIT_CODE) -------"
exit $EXIT_CODE
