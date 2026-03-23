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
cd "$SLURM_SUBMIT_DIR"
echo "WORKDIR=$SLURM_SUBMIT_DIR"

# ─── ログ・出力ディレクトリ作成 ──────────────────────────────────────────────
mkdir -p logs
mkdir -p results/infer

# ─── 仮想環境のセットアップ ───────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
echo "Python: $(which python3)"

# ─── 依存パッケージのインストール ─────────────────────────────────────────────
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
pip install "transformers==4.45.2" peft accelerate "timm>=0.9.10,<1.0.0" opencv-python Pillow --quiet

# ─── 推論・評価実行 ───────────────────────────────────────────────────────────
CKPT=checkpoints/crack_openvla/best

# エピソードJSON一覧を取得して最初の5件を評価
EPISODES=$(ls data/auto_raw/episodes/episode_*.json 2>/dev/null | head -5)

if [ -z "$EPISODES" ]; then
  echo "[ERROR] data/auto_raw/episodes/ にエピソードが見つかりません"
  exit 1
fi

for EP in $EPISODES; do
  EP_NAME=$(basename "$EP" .json)
  echo ""
  echo "--- 評価: $EP ---"
  python3 -u training/infer.py \
    --ckpt_dir   "$CKPT" \
    --episode    "$EP" \
    --image_dir  data/auto_raw/patches \
    --output_dir results/infer/$EP_NAME
done

echo ""
echo "推論完了: $(date)"
