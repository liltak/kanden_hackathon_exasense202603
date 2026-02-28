#!/usr/bin/env bash
# Environment setup for ExaSense
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== ExaSense Environment Setup ==="
echo "Project: $PROJECT_DIR"

# Detect GPU
if command -v nvidia-smi &>/dev/null; then
    echo "[GPU] NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    INSTALL_GPU=true
else
    echo "[GPU] No NVIDIA GPU detected. Installing CPU-only dependencies."
    INSTALL_GPU=false
fi

# Install with uv
cd "$PROJECT_DIR"

if command -v uv &>/dev/null; then
    echo "[install] Using uv..."
    if [ "$INSTALL_GPU" = true ]; then
        uv sync --extra gpu --extra dev
    else
        uv sync --extra dev
    fi
else
    echo "[install] uv not found. Using pip..."
    pip install -e ".[dev]"
fi

# Verify installation
echo ""
echo "=== Verification ==="
python -c "
import pvlib; print(f'pvlib {pvlib.__version__}')
import trimesh; print(f'trimesh {trimesh.__version__}')
import plotly; print(f'plotly {plotly.__version__}')
import gradio; print(f'gradio {gradio.__version__}')
import numpy; print(f'numpy {numpy.__version__}')
print('All core dependencies OK!')
"

if [ "$INSTALL_GPU" = true ]; then
    python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
fi

echo ""
echo "=== Setup complete ==="
echo "Run demo: python -m src.simulation.runner"
echo "Run UI:   python -m src.ui.app"
echo "Run tests: pytest tests/"
