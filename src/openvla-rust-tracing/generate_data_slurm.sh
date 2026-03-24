#!/bin/bash
#SBATCH -J crack-data-gen
#SBATCH -p debug
#SBATCH --gres=gpu:0
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -o logs/slurm-datagen-%j.out

echo "JOB_ID=$SLURM_JOB_ID"
hostname

# ─── 作業ディレクトリ ─────────────────────────────────────────────────────────
cd "$SLURM_SUBMIT_DIR"
echo "WORKDIR=$SLURM_SUBMIT_DIR"

# ─── エピソード数（変更する場合はここを編集）─────────────────────────────────
N_EPISODES=300

# ─── ログ・出力ディレクトリ作成 ──────────────────────────────────────────────
mkdir -p logs
mkdir -p crack_generated
mkdir -p data/auto_raw/episodes
mkdir -p data/auto_raw/patches

# ─── 仮想環境のセットアップ ───────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
echo "Python: $(which python3)"

pip install Pillow numpy opencv-python --quiet

# ─── STEP 1: クラック画像生成 + アノテーション ────────────────────────────────
N_IMAGES=$(ls crack_generated/annotations/episode_*.json 2>/dev/null | wc -l)
if [ "$N_IMAGES" -ge "$N_EPISODES" ]; then
  echo "[スキップ] crack_generated/annotations に ${N_IMAGES} エピソード既存 → STEP 1 をスキップ"
else
  echo ""
  echo "================================================================"
  echo " STEP 1: クラック画像生成 + アノテーション (N=${N_EPISODES})"
  echo "================================================================"
  python3 -u data_generation/generate_crack.py --n "$N_EPISODES" --bg_dir background
  echo "[done] 画像: $(ls crack_generated/*.png 2>/dev/null | wc -l) 枚"
fi

# ─── STEP 2.5: annotations → data/auto_raw 変換 ──────────────────────────
N_READY=$(ls data/auto_raw/episodes/*.json 2>/dev/null | wc -l)
if [ "$N_READY" -ge "$N_EPISODES" ]; then
  echo "[スキップ] data/auto_raw に ${N_READY} エピソード既存 → STEP 2.5 をスキップ"
else
  echo ""
  echo "================================================================"
  echo " STEP 2.5: crack_generated/annotations → data/auto_raw 変換"
  echo "================================================================"
  python3 -u - <<'PYEOF'
import json, shutil
from pathlib import Path

src = Path("crack_generated/annotations")
dst = Path("data/auto_raw")
(dst / "episodes").mkdir(parents=True, exist_ok=True)
(dst / "patches").mkdir(parents=True, exist_ok=True)

copied = 0
for src_img in (src / "steps").glob("*.png"):
    shutil.copy2(src_img, dst / "patches" / src_img.name)
    copied += 1

for src_ep in sorted(src.glob("episode_*.json")):
    ep = json.loads(src_ep.read_text(encoding="utf-8"))
    for step in ep.get("steps", []):
        if "image_path" in step:
            step["patch_path"] = "patches/" + Path(step.pop("image_path")).name
    (dst / "episodes" / src_ep.name).write_text(
        json.dumps(ep, indent=2, ensure_ascii=False), encoding="utf-8"
    )

print(f"[done] エピソード: {len(list((dst/'episodes').glob('*.json')))} 件")
print(f"[done] パッチ画像: {copied} 枚 → {dst}/patches/")
PYEOF
fi

# ─── STEP 3: データセット検証 ─────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " STEP 3: データセット検証 (train.py との互換性チェック)"
echo "================================================================"

python3 -u data_generation/convert_to_rlds.py \
  --data data/auto_raw \
  --show_stats

echo ""
echo "完了: $(date)"
echo "次のステップ: sbatch train_slurm.sh"
