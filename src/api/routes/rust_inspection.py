"""Rust inspection API routes — OpenVLA rust tracing results viewer.

Mock mode returns synthetic trajectory + metrics.
When connected to H100, will proxy real simulation results.
"""

from __future__ import annotations

import hashlib
import random
from urllib.parse import quote

from fastapi import APIRouter, UploadFile, File

from ..schemas import (
    RustInspectionRunRequest,
    RustInspectionRunResponse,
    RustInspectionStatusResponse,
    RustInspectionResultSchema,
    RustInspectionMetricsSchema,
)

router = APIRouter(prefix="/api/rust-inspection", tags=["rust-inspection"])


def _build_grid_svg(
    *,
    rows: int = 5,
    cols: int = 5,
    visited: list[tuple[int, int]],
    rust_cells: list[tuple[int, int]],
    trajectory: list[tuple[int, int]],
    width: int = 640,
    height: int = 640,
) -> str:
    """Build an SVG visualization of the rust tracing grid."""
    cell_w = width // cols
    cell_h = height // rows

    cells_svg = []
    for r in range(rows):
        for c in range(cols):
            x, y = c * cell_w, r * cell_h
            if (r, c) in rust_cells and (r, c) in visited:
                fill = "#22c55e"
                opacity = "0.7"
            elif (r, c) in rust_cells:
                fill = "#ef4444"
                opacity = "0.6"
            elif (r, c) in visited:
                fill = "#3b82f6"
                opacity = "0.3"
            else:
                fill = "#1e293b"
                opacity = "0.15"
            cells_svg.append(
                f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" '
                f'fill="{fill}" fill-opacity="{opacity}" stroke="#475569" stroke-width="1"/>'
            )

    path_points = []
    for r, c in trajectory:
        cx = c * cell_w + cell_w // 2
        cy = r * cell_h + cell_h // 2
        path_points.append(f"{cx},{cy}")

    path_svg = ""
    if len(path_points) >= 2:
        path_svg = (
            f'<polyline points="{" ".join(path_points)}" '
            f'fill="none" stroke="#f59e0b" stroke-width="3" stroke-opacity="0.9"/>'
        )
    if trajectory:
        sr, sc = trajectory[0]
        start_svg = (
            f'<circle cx="{sc * cell_w + cell_w // 2}" cy="{sr * cell_h + cell_h // 2}" '
            f'r="8" fill="#22d3ee" stroke="white" stroke-width="2"/>'
        )
        er, ec = trajectory[-1]
        end_svg = (
            f'<circle cx="{ec * cell_w + cell_w // 2}" cy="{er * cell_h + cell_h // 2}" '
            f'r="8" fill="#f43f5e" stroke="white" stroke-width="2"/>'
        )
    else:
        start_svg = end_svg = ""

    legend_y = height - 20
    legend = f"""
    <rect x="10" y="{legend_y - 16}" width="12" height="12" fill="#22c55e" rx="2"/>
    <text x="28" y="{legend_y - 6}" fill="#e2e8f0" font-size="11" font-family="Arial">検出済み</text>
    <rect x="100" y="{legend_y - 16}" width="12" height="12" fill="#ef4444" rx="2"/>
    <text x="118" y="{legend_y - 6}" fill="#e2e8f0" font-size="11" font-family="Arial">未検出サビ</text>
    <rect x="200" y="{legend_y - 16}" width="12" height="12" fill="#3b82f6" rx="2"/>
    <text x="218" y="{legend_y - 6}" fill="#e2e8f0" font-size="11" font-family="Arial">訪問済み</text>
    <circle cx="316" cy="{legend_y - 10}" r="5" fill="#22d3ee"/>
    <text x="326" y="{legend_y - 6}" fill="#e2e8f0" font-size="11" font-family="Arial">開始</text>
    <circle cx="376" cy="{legend_y - 10}" r="5" fill="#f43f5e"/>
    <text x="386" y="{legend_y - 6}" fill="#e2e8f0" font-size="11" font-family="Arial">終了</text>
    """

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="{width}" height="{height}" fill="#0f172a" rx="12"/>
  {"".join(cells_svg)}
  {path_svg}
  {start_svg}
  {end_svg}
  {legend}
</svg>"""
    return f"data:image/svg+xml;charset=UTF-8,{quote(svg)}"


def _generate_mock_result(seed: str, grid_rows: int, grid_cols: int) -> RustInspectionResultSchema:
    """Generate a deterministic mock inspection result."""
    rng = random.Random(seed)

    total_cells = grid_rows * grid_cols
    n_rust = rng.randint(max(3, total_cells // 5), max(4, total_cells // 3))
    all_cells = [(r, c) for r in range(grid_rows) for c in range(grid_cols)]
    rng.shuffle(all_cells)
    rust_cells = set(all_cells[:n_rust])

    trajectory: list[tuple[int, int]] = []
    visited_rust: set[tuple[int, int]] = set()
    visited: set[tuple[int, int]] = set()

    start = next(iter(rust_cells))
    pos = start
    trajectory.append(pos)
    visited.add(pos)
    if pos in rust_cells:
        visited_rust.add(pos)

    max_steps = total_cells * 3
    backtracks = 0
    for _ in range(max_steps):
        if len(visited_rust) >= len(rust_cells):
            break
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < grid_rows and 0 <= nc < grid_cols:
                neighbors.append((nr, nc))
        unvisited_rust = [n for n in neighbors if n in rust_cells and n not in visited_rust]
        if unvisited_rust:
            pos = rng.choice(unvisited_rust)
        else:
            unvisited = [n for n in neighbors if n not in visited]
            if unvisited:
                pos = rng.choice(unvisited)
            else:
                pos = rng.choice(neighbors)
                backtracks += 1
        trajectory.append(pos)
        visited.add(pos)
        if pos in rust_cells:
            visited_rust.add(pos)

    coverage = len(visited_rust) / len(rust_cells) if rust_cells else 0.0
    rust_list = sorted(rust_cells)
    visited_list = sorted(visited)

    trajectory_svg = _build_grid_svg(
        rows=grid_rows,
        cols=grid_cols,
        visited=visited_list,
        rust_cells=rust_list,
        trajectory=trajectory,
    )

    metrics = RustInspectionMetricsSchema(
        coverage_rate=round(coverage, 4),
        total_steps=len(trajectory),
        backtrack_count=backtracks,
        component_jumps=rng.randint(0, 3),
        rust_patch_count=len(rust_cells),
        visited_rust_count=len(visited_rust),
        grid_rows=grid_rows,
        grid_cols=grid_cols,
    )

    return RustInspectionResultSchema(
        trajectory_image_data_url=trajectory_svg,
        source_image_data_url=None,
        metrics=metrics,
    )


def _detect_gpu() -> tuple[bool, str, float | None]:
    try:
        import torch
        if torch.cuda.is_available():
            return (
                True,
                torch.cuda.get_device_name(0),
                round(torch.cuda.memory_allocated(0) / 1e9, 2),
            )
    except Exception:
        pass
    return False, "cpu/mock", None


@router.get("/status", response_model=RustInspectionStatusResponse)
async def rust_inspection_status() -> RustInspectionStatusResponse:
    gpu_available, device, vram_used_gb = _detect_gpu()
    return RustInspectionStatusResponse(
        service_status="ready",
        mock_mode=True,
        model_name="openvla/openvla-7b + LoRA (mock)",
        device=device,
        gpu_available=gpu_available,
        model_loaded=False,
        vram_used_gb=vram_used_gb,
    )


@router.post("/run", response_model=RustInspectionRunResponse)
async def run_rust_inspection(req: RustInspectionRunRequest) -> RustInspectionRunResponse:
    seed = hashlib.md5(f"{req.seed}|{req.grid_rows}|{req.grid_cols}".encode()).hexdigest()
    result = _generate_mock_result(seed, req.grid_rows, req.grid_cols)
    request_id = f"rust-{seed[:10]}"
    return RustInspectionRunResponse(
        request_id=request_id,
        status="complete",
        mock_mode=True,
        result=result,
    )


@router.post("/upload-and-run", response_model=RustInspectionRunResponse)
async def upload_and_run(
    file: UploadFile = File(...),
    grid_rows: int = 5,
    grid_cols: int = 5,
) -> RustInspectionRunResponse:
    """Upload an infrastructure image and run mock inspection."""
    content = await file.read()
    seed = hashlib.md5(content[:4096]).hexdigest()
    result = _generate_mock_result(seed, grid_rows, grid_cols)
    request_id = f"rust-{seed[:10]}"
    return RustInspectionRunResponse(
        request_id=request_id,
        status="complete",
        mock_mode=True,
        result=result,
    )
