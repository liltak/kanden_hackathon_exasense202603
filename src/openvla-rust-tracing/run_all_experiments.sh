#!/bin/bash
#SBATCH -J openvla-rust-all-exp
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH -o logs/all-exp-%j.out
#SBATCH -e logs/all-exp-%j.err

echo "JOB_ID=$SLURM_JOB_ID"
hostname
nvidia-smi

# ─── 作業ディレクトリ ────────────────────────────────────────────────────────
cd "$SLURM_SUBMIT_DIR"
echo "WORKDIR=$SLURM_SUBMIT_DIR"

# ─── ログ・出力ディレクトリ作成 ──────────────────────────────────────────────
mkdir -p logs
mkdir -p results/exp1
mkdir -p results/exp2_epoch_0001

# ─── 仮想環境のセットアップ ───────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
echo "Python: $(which python3)"

# ─── 依存パッケージのインストール ─────────────────────────────────────────────
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
pip install "transformers==4.45.2" peft accelerate "timm>=0.9.10,<1.0.0" opencv-python Pillow --quiet

# ─── 共通パラメータ ────────────────────────────────────────────────────────────
CKPT="${CKPT:-checkpoints/crack_openvla/epoch_0001}"
SPLIT_JSON="${SPLIT_JSON:-checkpoints/crack_openvla/data_split.json}"
EPISODE_DIR="${EPISODE_DIR:-data/auto_raw/episodes}"
IMAGE_DIR="${IMAGE_DIR:-data/auto_raw/patches}"
CRACK_IMAGE_DIR="${CRACK_IMAGE_DIR:-crack_generated}"
STARTS="${STARTS:-results/rollout/starts.json}"
MAX_STEPS="${MAX_STEPS:-100}"

# # =============================================================================
# # exp1: テストデータでのテスト
# #   全エピソードJSON を対象に推論・精度検証を実行する
# #   出力先: results/exp1/<episode_id>/
# #     - step_NNN.png : GT（緑矢印）+ Pred（青矢印）付きパッチ画像
# #     - trajectory.png : 全体軌跡の比較画像
# # =============================================================================
# echo ""
# echo "############################################################"
# echo "# exp1: テストデータでのロールアウト"
# echo "#   チェックポイント : $CKPT"
# echo "#   データ分割JSON  : $SPLIT_JSON"
# echo "#   エピソードDir   : $EPISODE_DIR"
# echo "#   クラック画像Dir  : $CRACK_IMAGE_DIR"
# echo "############################################################"

# if [ ! -f "$SPLIT_JSON" ]; then
#   echo "[ERROR] data_split.json が見つかりません: $SPLIT_JSON"
#   echo "  先に sbatch train_slurm.sh を実行してください"
#   exit 1
# fi

# python3 -u - <<PYEOF
# import json, subprocess, sys, os
# from pathlib import Path

# ckpt           = os.environ.get("CKPT",            "checkpoints/crack_openvla/best")
# split_json     = os.environ.get("SPLIT_JSON",      "checkpoints/crack_openvla/data_split.json")
# episode_dir    = Path(os.environ.get("EPISODE_DIR", "data/auto_raw/episodes"))
# crack_img_dir  = Path(os.environ.get("CRACK_IMAGE_DIR", "crack_generated"))
# max_steps      = os.environ.get("MAX_STEPS",       "100")

# split   = json.loads(Path(split_json).read_text())
# all_eps = sorted(episode_dir.glob("episode_*.json"))
# test_eps = [all_eps[i] for i in sorted(split["test"]) if i < len(all_eps)]

# if not test_eps:
#     print("[ERROR] テストエピソードが見つかりません")
#     sys.exit(1)

# total  = len(test_eps)
# errors = 0
# print(f"テストエピソード数: {total} 件\n")

# for i, ep_path in enumerate(test_eps):
#     ep = json.loads(ep_path.read_text())
#     ep_name = ep_path.stem

#     # 起点座標を最初のステップから取得
#     first = next((s for s in ep.get("steps", []) if s.get("is_first")), None) \
#             or (ep["steps"][0] if ep.get("steps") else None)
#     if first is None:
#         print(f"  [{i+1}/{total}] スキップ: steps が空 ({ep_name})")
#         errors += 1
#         continue

#     x, y = int(first["pixel_x"]), int(first["pixel_y"])

#     # 元画像パスを解決
#     src_img = crack_img_dir / ep.get("source_image", "")
#     if not src_img.exists():
#         print(f"  [{i+1}/{total}] スキップ: 元画像が見つかりません ({src_img})")
#         errors += 1
#         continue

#     out = f"results/exp1/{ep_name}"
#     print(f"[{i+1}/{total}] {ep_name}  起点=({x},{y})  画像={src_img.name}")

#     cmd = [
#         sys.executable, "training/infer.py",
#         "--ckpt_dir",   ckpt,
#         "--image",      str(src_img),
#         "--x",          str(x),
#         "--y",          str(y),
#         "--max_steps",  max_steps,
#         "--output_dir", out,
#     ]
#     result = subprocess.run(cmd)
#     if result.returncode != 0:
#         print(f"  [ERROR] 失敗 (exit={result.returncode})")
#         errors += 1
#     else:
#         print(f"  [OK] → {out}/trajectory.png\n")

# print(f"\n成功: {total - errors} / {total}")
# if errors:
#     print(f"失敗: {errors} 件")
# PYEOF

# ERRORS=$?

# # =============================================================================
# # 完了サマリー
# # =============================================================================
# echo ""
# echo "############################################################"
# echo "# exp1 完了"
# echo "#   出力先: results/exp1/"
# echo "############################################################"
# ls -lh results/exp1/

# =============================================================================
# exp2: 自然画像でのロールアウトテスト
#   starts.json に記載された自然画像と起点座標を使い、モデルが
#   クラック/錆トレースを自律追従できるか検証する
#   出力先: results/exp2/<image_stem>/
#     - patch_NNN.png   : 各ステップの切り出しパッチ
#     - trajectory.json : 軌跡データ（座標・delta値）
#     - trajectory.png  : 元画像上に軌跡を描画した可視化画像
# =============================================================================
echo ""
echo "############################################################"
echo "# exp2: 自然画像でのロールアウト"
echo "#   チェックポイント: $CKPT"
echo "#   starts.json    : $STARTS"
echo "#   最大ステップ数  : $MAX_STEPS"
echo "############################################################"

if [ ! -f "$STARTS" ]; then
  echo "[ERROR] 起点ファイルが見つかりません: $STARTS"
  echo "  Mac で python3 data_generation/pick_start.py を実行してください"
  exit 1
fi

python3 -u - <<'PYEOF'
import json, subprocess, sys, os
from pathlib import Path

ckpt      = os.environ.get("CKPT",      "checkpoints/crack_openvla/epoch_0001")
starts_f  = os.environ.get("STARTS",    "results/rollout/starts.json")
max_steps = os.environ.get("MAX_STEPS", "100")

starts = json.loads(Path(starts_f).read_text())
print(f"{len(starts)} 枚の自然画像を処理します\n")

total  = len(starts)
errors = 0

for i, s in enumerate(starts):
    img   = s["image"]
    x, y  = s["x"], s["y"]
    stem  = Path(img).stem
    out   = f"results/exp2/{stem}"

    print(f"[{i+1}/{total}] {Path(img).name}  起点=({x},{y})")
    cmd = [
        sys.executable, "training/infer.py",
        "--ckpt_dir",   ckpt,
        "--image",      img,
        "--x",          str(x),
        "--y",          str(y),
        "--max_steps",  max_steps,
        "--output_dir", out,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  [ERROR] infer 失敗 (exit={result.returncode}): {img}")
        errors += 1
    else:
        print(f"  [OK] → {out}/trajectory.png\n")

print(f"\n成功: {total - errors} / {total}")
if errors:
    print(f"失敗: {errors} 件")
PYEOF

echo ""
echo "############################################################"
echo "# exp2 完了"
echo "#   出力先: results/exp2/"
echo "############################################################"
ls -lh results/exp2/
