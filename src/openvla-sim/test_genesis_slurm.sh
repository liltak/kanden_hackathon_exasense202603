#!/bin/bash
#SBATCH -J genesis-h100-test
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -o logs/genesis-test-%j.out
#SBATCH -e logs/genesis-test-%j.err

echo "JOB_ID=$SLURM_JOB_ID"
hostname
nvidia-smi

# ─── 作業ディレクトリ ──────────────────────────────────────────────────────────
cd /home/team-002/openvla-sim

# ─── ログディレクトリ作成 ──────────────────────────────────────────────────────
mkdir -p logs

# ─── 仮想環境のセットアップ ────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  python -m venv .venv
fi
source .venv/bin/activate

# ─── 依存パッケージのインストール ─────────────────────────────────────────────
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --quiet
pip install -e third_party/Genesis --quiet

# ─── GPU 設定（1台使用）────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0

# ─── Genesis H100 テスト実行 ───────────────────────────────────────────────────
echo "------- Genesis H100 Smoke Test Start -------"
python openvla-sim/test_genesis_h100.py
EXIT_CODE=$?

echo "------- Genesis H100 Smoke Test End (exit=$EXIT_CODE) -------"
exit $EXIT_CODE
