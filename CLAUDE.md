# ExaSense - Factory Energy Optimization Solution

## Project Overview
Factory solar panel optimization: 3D reconstruction + solar simulation + VLM analysis + WebUI.

## Architecture (5-phase pipeline)
```
Phase 1-2 (GPU)     Phase 3 (CPU)          Phase 4 (GPU)    Phase 5
3D Recon + Mesh  →  Solar Simulation  →  VLM Analysis  →  Next.js + FastAPI
(VGGT/COLMAP)       (pvlib/trimesh)       (Qwen3.5-VL)     (REST/WS)
```

## Key Commands
```bash
uv run pytest tests/ -v                 # Run tests (18 tests)
uv run python -m src.simulation.runner  # Run demo simulation
uv run python -m src.api.server         # Launch FastAPI backend (port 8000)
cd frontend && npm run dev              # Launch Next.js frontend (port 3000)
scripts/setup_h100.sh                   # Full H100 GPU setup
```

## Module Structure
- `src/simulation/` — Solar simulation (Phase 3) - pvlib, trimesh, plotly
- `src/reconstruction/` — 3D recon (Phase 1-2) - VGGT, COLMAP, OpenSplat, mesh processing
- `src/vlm/` — VLM pipeline (Phase 4) - Qwen3.5-VL, Unsloth fine-tuning
- `src/api/server.py` — FastAPI backend (Phase 5) - REST API + WebSocket
- `frontend/` — Next.js 16 frontend (Phase 5) - React 19, Three.js, shadcn/ui

## GPU vs CPU Modules
GPU modules use TYPE_CHECKING + lazy imports for torch/transformers.
All modules are importable on macOS (CPU-only).

## Frontend (Phase 5)
- **Gradio は使用禁止** — `src/ui/app.py` は廃止済み。UIの話題では必ず Next.js を使うこと
- Next.js 16 (App Router) + React 19 + TypeScript
- Three.js (React Three Fiber / drei) で3Dビューア
- shadcn/ui + Radix UI + Tailwind CSS 4
- TanStack React Query で状態管理
- 日本語UI (Noto Sans JP)
- バックエンド通信: FastAPI REST (`localhost:8000`) + WebSocket

## Conventions
- Respond in Japanese
- Commit messages in English, conventional commit format
- Python 3.12, uv package manager
- Rich for console output, dataclasses for structured results
