#!/usr/bin/env bash
# H100 E2E Pipeline Test — download data + run full pipeline
# Usage: bash scripts/run_h100_e2e.sh [--max-images 20]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "=========================================="
echo "  ExaSense H100 E2E Pipeline Test"
echo "=========================================="

# Step 1: Verify GPU
echo ""
echo "[1/3] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# Step 2: Download test data if needed
GARDEN_DIR="$PROJECT_DIR/data/raw/mipnerf360/garden"
if [ ! -d "$GARDEN_DIR/images" ]; then
    echo "[2/3] Downloading Mip-NeRF 360 Garden dataset..."
    bash "$SCRIPT_DIR/download_data.sh" garden
else
    N_IMAGES=$(find "$GARDEN_DIR/images" -type f \( -name "*.JPG" -o -name "*.jpg" -o -name "*.png" \) | wc -l)
    echo "[2/3] Dataset ready: $N_IMAGES images in $GARDEN_DIR/images"
fi

# Step 3: Run the E2E pipeline
echo ""
echo "[3/3] Running E2E pipeline..."
uv run python "$SCRIPT_DIR/run_h100_e2e.py" "$@"

echo ""
echo "=========================================="
echo "  Done! Results in data/e2e_results/h100/"
echo "=========================================="
