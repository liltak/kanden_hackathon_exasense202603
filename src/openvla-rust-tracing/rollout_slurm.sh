#!/bin/bash
#SBATCH -J crack-rollout
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o logs/slurm-rollout-%j.out

echo "JOB_ID=$SLURM_JOB_ID"
hostname
nvidia-smi

# ─── 作業ディレクトリ ─────────────────────────────────────────────────────────
cd "$SLURM_SUBMIT_DIR"
echo "WORKDIR=$SLURM_SUBMIT_DIR"

# ─── 設定 ────────────────────────────────────────────────────────────────────
CKPT=checkpoints/crack_openvla/best
STARTS=results/rollout/starts.json
MAX_STEPS=30

# ─── ログディレクトリ作成 ─────────────────────────────────────────────────────
mkdir -p logs

# ─── 仮想環境のセットアップ ───────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
echo "Python: $(which python3)"

# ─── 依存パッケージのインストール ─────────────────────────────────────────────
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
pip install "transformers==4.45.2" peft accelerate "timm>=0.9.10,<1.0.0" \
            opencv-python Pillow numpy --quiet

# ─── チェックポイント確認 ─────────────────────────────────────────────────────
if [ ! -d "$CKPT" ]; then
  echo "[ERROR] チェックポイントが見つかりません: $CKPT"
  echo "  先に sbatch train_slurm.sh を実行してください"
  exit 1
fi

# ─── starts.json から各画像のロールアウトを実行 ───────────────────────────────
if [ ! -f "$STARTS" ]; then
  echo "[ERROR] 起点ファイルが見つかりません: $STARTS"
  echo "  Mac で python3 training/pick_start.py を実行してください"
  exit 1
fi

echo ""
echo "================================================================"
echo " ロールアウト実行 (starts.json: $STARTS)"
echo "================================================================"

python3 -u - <<PYEOF
import json, subprocess, sys
from pathlib import Path

starts = json.loads(Path("$STARTS").read_text())
print(f"{len(starts)} 枚の画像を処理します\n")

for i, s in enumerate(starts):
    img   = s["image"]
    x, y  = s["x"], s["y"]
    stem  = Path(img).stem
    out   = f"results/rollout/{stem}"

    print(f"[{i+1}/{len(starts)}] {Path(img).name}  起点=({x},{y})")
    cmd = [
        sys.executable, "training/rollout.py",
        "--ckpt_dir",   "$CKPT",
        "--image",      img,
        "--x",          str(x),
        "--y",          str(y),
        "--max_steps",  "$MAX_STEPS",
        "--output_dir", out,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  [ERROR] rollout 失敗: {img}")
    else:
        print(f"  → 結果: {out}/trajectory.png\n")

print("全ロールアウト完了")
PYEOF

echo ""
echo "完了: $(date)"
