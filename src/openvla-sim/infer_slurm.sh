#!/bin/bash
#SBATCH -J openvla-infer
#SBATCH -p debug
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -o logs/infer-%j.out
#SBATCH -e logs/infer-%j.err

echo "JOB_ID=$SLURM_JOB_ID"
hostname
nvidia-smi

# ─── 作業ディレクトリ ──────────────────────────────────────────────────────────
cd "$HOME/openvla-sim"

# ─── ログディレクトリ作成 ──────────────────────────────────────────────────────
mkdir -p logs

# ─── 仮想環境のセットアップ ────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  python -m venv .venv
fi
source .venv/bin/activate

# ─── 依存パッケージのインストール ─────────────────────────────────────────────
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --quiet
pip install "transformers==4.44.0" peft accelerate tensorboard Pillow scipy "timm>=0.9.10,<1.0.0" --quiet
pip install -e openvla-sim/third_party/Genesis --quiet
# ─── ANSIカラー無効化（ログ崩れ防止）─────────────────────────────────────────
export TERM=dumb
export NO_COLOR=1

# ─── GPU 設定（1台使用）────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0,1

# ─── ヘッドレスOpenGLレンダリング設定（ディスプレイなし環境用）────────────────
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=0

# ─── 引数設定 ─────────────────────────────────────────────────────────────────
CKPT_DIR="${CKPT_DIR:-checkpoints_v3/drone_openvla/best}"
TARGET="ソファ"                                # 単一オブジェクトモード: ソファ / アームチェア / 木製引き出し / "" で複数モード
INSTRUCTION="${INSTRUCTION:-Fly toward the sofa and do a full loop around it.}"  # TARGET 未指定時に使用
OUTPUT="${OUTPUT:-logs/infer-${SLURM_JOB_ID}.mp4}"
MAX_STEPS="${MAX_STEPS:-1000}"

# ─── OpenVLA 推論実行 ──────────────────────────────────────────────────────────
echo "------- OpenVLA Infer Start -------"
echo "  CKPT_DIR   : $CKPT_DIR"
echo "  TARGET     : ${TARGET:-（複数オブジェクトモード）}"
echo "  INSTRUCTION: $INSTRUCTION"
echo "  OUTPUT     : $OUTPUT"
echo "  MAX_STEPS  : $MAX_STEPS"

# --target が指定された場合は単一オブジェクトモード
if [ -n "$TARGET" ]; then
  python openvla-sim/scripts/infer.py \
    --ckpt_dir "$CKPT_DIR" \
    --target "$TARGET" \
    --instruction "$INSTRUCTION" \
    --output "$OUTPUT" \
    --max_steps "$MAX_STEPS"
else
  python openvla-sim/scripts/infer.py \
    --ckpt_dir "$CKPT_DIR" \
    --instruction "$INSTRUCTION" \
    --output "$OUTPUT" \
    --max_steps "$MAX_STEPS"
fi
EXIT_CODE=$?

echo "------- OpenVLA Infer End (exit=$EXIT_CODE) -------"
exit $EXIT_CODE
