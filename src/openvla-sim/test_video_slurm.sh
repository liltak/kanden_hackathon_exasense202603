#!/bin/bash
#SBATCH -J genesis-video-test
#SBATCH -p debug
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -o logs/test-video-%j.out
#SBATCH -e logs/test-video-%j.err

echo "JOB_ID=$SLURM_JOB_ID"
hostname
nvidia-smi

# ─── 作業ディレクトリ ──────────────────────────────────────────────────────────
cd /home/team-002/openvla-sim2

# ─── ログディレクトリ作成 ──────────────────────────────────────────────────────
mkdir -p logs

# ─── 仮想環境のセットアップ ────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  python -m venv .venv
fi
source .venv/bin/activate

# ─── 依存パッケージのインストール ─────────────────────────────────────────────
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e openvla-sim/third_party/Genesis --quiet

# ─── ANSIカラー無効化（ログ崩れ防止）─────────────────────────────────────────
export TERM=dumb
export NO_COLOR=1

# ─── GPU 設定 ─────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0

# ─── ヘッドレスOpenGLレンダリング設定（ディスプレイなし環境用）────────────────
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=0

# ─── Genesis 動画テスト実行 ────────────────────────────────────────────────────
echo "------- Genesis Video Test Start -------"
python openvla-sim/test_video.py \
  --output logs/test-video-${SLURM_JOB_ID}.mp4 \
  --steps 150
EXIT_CODE=$?

echo "------- Genesis Video Test End (exit=$EXIT_CODE) -------"
exit $EXIT_CODE
