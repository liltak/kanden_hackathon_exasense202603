#!/usr/bin/env bash
# H100 server setup for ExaSense - full pipeline with GPU support
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "  ExaSense H100 Environment Setup"
echo "=========================================="

# Check GPU
echo ""
echo "[1/6] Checking GPU..."
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Is CUDA installed?"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# Install uv if not present
echo "[2/6] Installing uv..."
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv $(uv --version)"

# Install base project
echo ""
echo "[3/6] Installing project dependencies..."
cd "$PROJECT_DIR"
uv sync --extra dev

# Install PyTorch with CUDA
echo ""
echo "[4/6] Installing PyTorch (CUDA 12.4)..."
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install GPU-specific packages
echo ""
echo "[5/6] Installing GPU packages..."
uv pip install \
    open3d \
    transformers>=4.45.0 \
    accelerate>=0.34.0 \
    bitsandbytes \
    vllm \
    embreex

# Install Unsloth for fine-tuning
echo ""
echo "[5b/6] Installing Unsloth..."
uv pip install unsloth

# Install COLMAP (if not already available)
echo ""
echo "[6/6] Checking COLMAP..."
if command -v colmap &>/dev/null; then
    echo "COLMAP already installed: $(colmap --version 2>&1 | head -1)"
else
    echo "COLMAP not found. Install with:"
    echo "  conda install -c conda-forge colmap"
    echo "  # OR build from source (see docker/Dockerfile)"
fi

# Verify
echo ""
echo "=========================================="
echo "  Verification"
echo "=========================================="
uv run python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

import pvlib; print(f'pvlib: {pvlib.__version__}')
import trimesh; print(f'trimesh: {trimesh.__version__}')
import plotly; print(f'plotly: {plotly.__version__}')
import gradio; print(f'gradio: {gradio.__version__}')

try:
    import transformers; print(f'transformers: {transformers.__version__}')
except: print('transformers: NOT INSTALLED')

try:
    import open3d; print(f'open3d: {open3d.__version__}')
except: print('open3d: NOT INSTALLED')

try:
    from trimesh.ray import ray_pyembree; print(f'Embree: available (accelerated ray backend)')
except: print('Embree: NOT AVAILABLE (using trimesh native ray backend)')

print()
print('All checks complete!')
"

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Quick start commands:"
echo "  uv run python -m src.simulation.runner     # Run demo simulation"
echo "  uv run python -m src.ui.app                # Launch WebUI"
echo "  uv run python -m src.vlm.model_loader      # Test VLM loading"
echo "  uv run pytest tests/ -v                    # Run tests"
echo ""
echo "Production deployment (k3s):"
echo "  sudo infra/onprem/setup-k3s.sh             # Install k3s + GPU Operator + Tailscale"
echo "  sudo infra/onprem/setup-storage.sh          # Setup MinIO + NFS storage"
echo "  # Join existing cluster:"
echo "  sudo infra/onprem/setup-k3s.sh --join https://<server>:6443 <token>"
echo ""
