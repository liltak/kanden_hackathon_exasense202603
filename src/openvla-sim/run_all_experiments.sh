#!/bin/bash
#SBATCH -J openvla-all-exp
#SBATCH -p debug
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -o logs/all-exp-%j.out
#SBATCH -e logs/all-exp-%j.err

echo "JOB_ID=$SLURM_JOB_ID"
hostname
nvidia-smi

# ─── 作業ディレクトリ ──────────────────────────────────────────────────────────
cd /home/team-002/openvla-sim2

# ─── ログ・出力ディレクトリ作成 ────────────────────────────────────────────────
mkdir -p logs
OUTPUT_DIR="output_epoch_0006/experiments"
mkdir -p "$OUTPUT_DIR"

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

# ─── GPU 設定 ────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0,1

# ─── ヘッドレスOpenGLレンダリング設定 ────────────────────────────────────────
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=0

# ─── 共通パラメータ ────────────────────────────────────────────────────────────
CKPT_DIR="${CKPT_DIR:-checkpoints_v4/drone_openvla/epoch_0006}"
MAX_STEPS="${MAX_STEPS:-500}"

# ─── ファイル名サニタイズ関数 ──────────────────────────────────────────────────
# スペース→アンダースコア、ファイル名に使えない記号を除去
sanitize() {
  echo "$1" | tr ' ' '_' | tr -d '.,()/'
}

# ─── 推論実行関数 ──────────────────────────────────────────────────────────────
# 引数: $1=exp番号_オブジェクト名ラベル  $2=instruction  $3=target(空文字で複数モード)
run_infer() {
  local label="$1"
  local instruction="$2"
  local target="$3"

  local safe_prompt
  safe_prompt=$(sanitize "$instruction")
  local output="${OUTPUT_DIR}/${label}_${safe_prompt}.mp4"

  echo ""
  echo "======================================================="
  echo "  LABEL      : $label"
  echo "  TARGET     : ${target:-（複数オブジェクトモード）}"
  echo "  INSTRUCTION: $instruction"
  echo "  OUTPUT     : $output"
  echo "======================================================="

  if [ -n "$target" ]; then
    python openvla-sim/scripts/infer.py \
      --ckpt_dir "$CKPT_DIR" \
      --target   "$target" \
      --instruction "$instruction" \
      --output   "$output" \
      --max_steps "$MAX_STEPS"
  else
    python openvla-sim/scripts/infer.py \
      --ckpt_dir "$CKPT_DIR" \
      --instruction "$instruction" \
      --output   "$output" \
      --max_steps "$MAX_STEPS"
  fi

  local exit_code=$?
  echo "--- 終了 (exit=$exit_code): $output ---"
  return $exit_code
}

# =============================================================================
# exp1: オブジェクト単体 × 正しい命令
#   ファイル名例: exp1_ソファ_Approach_the_sofa_fly_around_it_and_take_photos.mp4
# =============================================================================
echo ""
echo "############################################################"
echo "# exp1: 単体オブジェクト × 正しいオブジェクト名の命令"
echo "############################################################"

run_infer "exp1_ソファ" \
  "Approach the sofa, fly around it, and take photos." \
  "ソファ"

run_infer "exp1_アームチェア" \
  "Approach the arm chair, fly around it, and take photos." \
  "アームチェア"

run_infer "exp1_木製引き出し" \
  "Approach the wooden drawer, fly around it, and take photos." \
  "木製引き出し"

# =============================================================================
# exp2: オブジェクト単体 × 別オブジェクト名の命令（VLAが言語に惑わされるか検証）
#   配置オブジェクトと命令の対応（ローテーション方式）:
#     ソファ配置       → アームチェアを要求
#     アームチェア配置 → 木製引き出しを要求
#     木製引き出し配置 → ソファを要求
#   ファイル名例: exp2_ソファ_Approach_the_arm_chair_fly_around_it_and_take_photos.mp4
# =============================================================================
echo ""
echo "############################################################"
echo "# exp2: 単体オブジェクト × 別オブジェクト名の命令"
echo "############################################################"

run_infer "exp2_ソファ" \
  "Approach the arm chair, fly around it, and take photos." \
  "ソファ"

run_infer "exp2_アームチェア" \
  "Approach the wooden drawer, fly around it, and take photos." \
  "アームチェア"

run_infer "exp2_木製引き出し" \
  "Approach the sofa, fly around it, and take photos." \
  "木製引き出し"

# =============================================================================
# exp3: 3オブジェクト全配置 × 各オブジェクトへの正しい命令
#   --target を省略することで infer.py が3つ全てを配置する
#   ファイル名例: exp3_ソファ_Approach_the_sofa_fly_around_it_and_take_photos.mp4
# =============================================================================
echo ""
echo "############################################################"
echo "# exp3: 全オブジェクト配置 × 各オブジェクト名の命令"
echo "############################################################"

run_infer "exp3_ソファ" \
  "Approach the sofa, fly around it, and take photos." \
  ""

run_infer "exp3_アームチェア" \
  "Approach the arm chair, fly around it, and take photos." \
  ""

run_infer "exp3_木製引き出し" \
  "Approach the wooden drawer, fly around it, and take photos." \
  ""

# =============================================================================
# 完了サマリー
# =============================================================================
echo ""
echo "############################################################"
echo "# 全実験完了"
echo "# 出力先: $OUTPUT_DIR"
echo "############################################################"
ls -lh "$OUTPUT_DIR"
