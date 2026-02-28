"""FastAPI backend for ExaSense.

Provides REST API endpoints for each phase of the pipeline.
"""

from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI(
    title="ExaSense API",
    description="Factory Energy Optimization Solution API",
    version="0.1.0",
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "simulation_results"


@app.get("/health")
async def health():
    return {"status": "ok", "service": "exasense"}


@app.post("/simulate")
async def run_simulation(
    latitude: float = 34.69,
    longitude: float = 135.50,
    year: int = 2025,
):
    """Run solar simulation with specified parameters."""
    from ..simulation.runner import create_demo_mesh, load_config, run_simulation

    config_path = PROJECT_ROOT / "configs" / "solar_params.yaml"
    config = load_config(config_path)
    config["location"]["latitude"] = latitude
    config["location"]["longitude"] = longitude
    config["simulation"]["year"] = year

    mesh = create_demo_mesh()
    output_dir = RESULTS_DIR / "latest"

    irradiance_results, roi_report = run_simulation(mesh, config, output_dir)

    return {
        "status": "complete",
        "n_faces": len(irradiance_results),
        "n_suitable_faces": len(roi_report.proposals),
        "total_capacity_kw": roi_report.total_capacity_kw,
        "annual_generation_kwh": roi_report.total_annual_generation_kwh,
        "annual_savings_jpy": roi_report.total_annual_savings_jpy,
        "payback_years": roi_report.overall_payback_years,
        "npv_25y_jpy": roi_report.overall_npv_25y_jpy,
    }


@app.get("/results/irradiance")
async def get_irradiance():
    """Get latest irradiance results as JSON."""
    path = RESULTS_DIR / "latest" / "irradiance_results.json"
    if not path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "No simulation results found. Run /simulate first."},
        )
    return FileResponse(path, media_type="application/json")


@app.get("/results/heatmap")
async def get_heatmap():
    """Get latest heatmap visualization as HTML."""
    path = RESULTS_DIR / "latest" / "irradiance_heatmap.html"
    if not path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "No heatmap found. Run /simulate first."},
        )
    return FileResponse(path, media_type="text/html")


@app.post("/chat")
async def chat(message: str):
    """Chat with VLM about simulation results (mock for now)."""
    # TODO: Replace with actual VLM inference in Phase 4
    return {
        "response": (
            "## 太陽光パネル設置提案\n\n"
            "シミュレーション結果に基づき、南向き屋根面への設置を推奨します。\n\n"
            "*（Phase 4 でVLMに置き換わります）*"
        )
    }
