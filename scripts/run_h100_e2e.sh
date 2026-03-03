#!/usr/bin/env bash
# H100 E2E Pipeline Test — download data + run full Phase 1→2→3→4 pipeline
# Usage: bash scripts/run_h100_e2e.sh [--max-images 20] [--skip-vlm]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "=========================================="
echo "  ExaSense H100 E2E Pipeline Test"
echo "  Phase 1→2→3→4 (VGGT→Mesh→Solar→VLM)"
echo "=========================================="

# Step 1: Verify GPU
echo ""
echo "[1/4] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# Step 2: Download test data if needed
GARDEN_DIR="$PROJECT_DIR/data/raw/mipnerf360/garden"
if [ ! -d "$GARDEN_DIR/images" ]; then
    echo "[2/4] Downloading Mip-NeRF 360 Garden dataset..."
    bash "$SCRIPT_DIR/download_data.sh" garden
else
    N_IMAGES=$(find "$GARDEN_DIR/images" -type f \( -name "*.JPG" -o -name "*.jpg" -o -name "*.png" \) | wc -l)
    echo "[2/4] Dataset ready: $N_IMAGES images in $GARDEN_DIR/images"
fi

# Step 3: Verify key dependencies
echo ""
echo "[3/4] Checking dependencies..."
uv run python -c "
import torch
print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
try:
    import transformers; print(f'  transformers: {transformers.__version__}')
except: print('  transformers: NOT INSTALLED')
try:
    import open3d; print(f'  open3d: {open3d.__version__}')
except: print('  open3d: NOT INSTALLED')
"
echo ""

# Step 4: Run the E2E pipeline (Phase 1→2→3→4)
echo ""
echo "[4/4] Running E2E pipeline..."
uv run python "$SCRIPT_DIR/run_h100_e2e.py" "$@"

echo ""
echo "=========================================="
echo "  Done! Results in data/e2e_results/h100/"
echo "=========================================="
