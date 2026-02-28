# ExaSense - Factory Energy Optimization Solution

## Project Overview
Factory solar panel optimization: 3D reconstruction + solar simulation + VLM analysis + WebUI.

## Architecture (5-phase pipeline)
```
Phase 1-2 (GPU)     Phase 3 (CPU)          Phase 4 (GPU)    Phase 5
3D Recon + Mesh  →  Solar Simulation  →  VLM Analysis  →  Gradio WebUI
(VGGT/COLMAP)       (pvlib/trimesh)       (Qwen3.5-VL)     (FastAPI)
```

## Key Commands
```bash
uv run pytest tests/ -v                 # Run tests (18 tests)
uv run python -m src.simulation.runner  # Run demo simulation
uv run python -m src.ui.app             # Launch WebUI (port 7860)
scripts/setup_h100.sh                   # Full H100 GPU setup
```

## Module Structure
- `src/simulation/` — Solar simulation (Phase 3) - pvlib, trimesh, plotly
- `src/reconstruction/` — 3D recon (Phase 1-2) - VGGT, COLMAP, OpenSplat, mesh processing
- `src/vlm/` — VLM pipeline (Phase 4) - Qwen3.5-VL, Unsloth fine-tuning
- `src/ui/app.py` — Gradio 5-tab dashboard (Phase 5)
- `src/api/server.py` — FastAPI backend (Phase 5)

## GPU vs CPU Modules
GPU modules use TYPE_CHECKING + lazy imports for torch/transformers.
All modules are importable on macOS (CPU-only).

## Conventions
- Respond in Japanese
- Commit messages in English, conventional commit format
- Python 3.12, uv package manager
- Rich for console output, dataclasses for structured results
