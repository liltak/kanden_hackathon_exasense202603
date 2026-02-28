"""Simulation API routes."""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models import Simulation, Task
from ..schemas import (
    FaceIrradianceSchema,
    PanelProposalSchema,
    ROIReportSchema,
    SimulationRequest,
    SimulationResult,
)
from ..ws import manager as ws_manager

router = APIRouter(prefix="/api/simulation", tags=["simulation"])

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "simulation_results"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# In-memory task store as fast cache; DB is source of truth
_tasks: dict[str, SimulationResult] = {}


def _irradiance_to_schema(r) -> FaceIrradianceSchema:
    return FaceIrradianceSchema(
        face_id=r.face_id,
        annual_irradiance_kwh_m2=r.annual_irradiance_kwh_m2,
        annual_direct_kwh_m2=r.annual_direct_kwh_m2,
        annual_diffuse_kwh_m2=r.annual_diffuse_kwh_m2,
        area_m2=r.area_m2,
        normal=r.normal,
        sun_hours=r.sun_hours,
    )


def _irradiance_to_dict(r) -> dict:
    return {
        "face_id": r.face_id,
        "annual_irradiance_kwh_m2": r.annual_irradiance_kwh_m2,
        "annual_direct_kwh_m2": r.annual_direct_kwh_m2,
        "annual_diffuse_kwh_m2": r.annual_diffuse_kwh_m2,
        "area_m2": r.area_m2,
        "normal": list(r.normal),
        "sun_hours": r.sun_hours,
    }


def _proposal_to_schema(p) -> PanelProposalSchema:
    return PanelProposalSchema(
        face_id=p.face_id,
        area_m2=p.area_m2,
        annual_generation_kwh=p.annual_generation_kwh,
        installed_capacity_kw=p.installed_capacity_kw,
        installation_cost_jpy=p.installation_cost_jpy,
        annual_savings_jpy=p.annual_savings_jpy,
        payback_years=p.payback_years,
        npv_25y_jpy=p.npv_25y_jpy,
        irr_percent=p.irr_percent,
        priority_rank=p.priority_rank,
    )


def _proposal_to_dict(p) -> dict:
    return {
        "face_id": p.face_id,
        "area_m2": p.area_m2,
        "annual_generation_kwh": p.annual_generation_kwh,
        "installed_capacity_kw": p.installed_capacity_kw,
        "installation_cost_jpy": p.installation_cost_jpy,
        "annual_savings_jpy": p.annual_savings_jpy,
        "payback_years": p.payback_years,
        "npv_25y_jpy": p.npv_25y_jpy,
        "irr_percent": p.irr_percent,
        "priority_rank": p.priority_rank,
    }


def _roi_to_schema(roi) -> ROIReportSchema:
    return ROIReportSchema(
        proposals=[_proposal_to_schema(p) for p in roi.proposals],
        total_area_m2=roi.total_area_m2,
        total_capacity_kw=roi.total_capacity_kw,
        total_annual_generation_kwh=roi.total_annual_generation_kwh,
        total_installation_cost_jpy=roi.total_installation_cost_jpy,
        total_annual_savings_jpy=roi.total_annual_savings_jpy,
        overall_payback_years=roi.overall_payback_years,
        overall_npv_25y_jpy=roi.overall_npv_25y_jpy,
    )


def _roi_to_dict(roi) -> dict:
    return {
        "proposals": [_proposal_to_dict(p) for p in roi.proposals],
        "total_area_m2": roi.total_area_m2,
        "total_capacity_kw": roi.total_capacity_kw,
        "total_annual_generation_kwh": roi.total_annual_generation_kwh,
        "total_installation_cost_jpy": roi.total_installation_cost_jpy,
        "total_annual_savings_jpy": roi.total_annual_savings_jpy,
        "overall_payback_years": roi.overall_payback_years,
        "overall_npv_25y_jpy": roi.overall_npv_25y_jpy,
    }


async def _run_simulation_async(task_id: str, req: SimulationRequest) -> None:
    """Run simulation in background thread and update task state + WS + DB."""
    import numpy as np

    from ...simulation.demo_factory import create_factory_complex, create_simple_factory
    from ...simulation.irradiance import compute_face_irradiance
    from ...simulation.ray_caster import compute_shadow_matrix
    from ...simulation.roi_calculator import generate_roi_report
    from ...simulation.runner import load_config
    from ...simulation.solar_position import (
        compute_clear_sky_irradiance,
        compute_solar_positions,
    )

    try:
        config_path = CONFIGS_DIR / "solar_params.yaml"
        config = load_config(config_path)
        config["location"]["latitude"] = req.latitude
        config["location"]["longitude"] = req.longitude
        config["simulation"]["year"] = req.year
        config["simulation"]["time_resolution_minutes"] = req.time_resolution_minutes
        config["panel"]["efficiency"] = req.panel_efficiency / 100
        config["electricity"]["price_per_kwh"] = req.electricity_price_jpy

        if req.mesh_source == "simple":
            mesh = create_simple_factory()
        else:
            mesh = create_factory_complex()

        loc = config["location"]
        sim = config["simulation"]
        t0 = time.time()

        # Step 1: Solar positions
        _tasks[task_id].status = "running"
        _tasks[task_id].step = "solar_positions"
        _tasks[task_id].progress = 0.0
        _tasks[task_id].message = "Computing solar positions..."
        await ws_manager.send_progress(task_id, "solar_positions", 0.0, "Computing solar positions...")

        solar = await asyncio.to_thread(
            compute_solar_positions,
            latitude=loc["latitude"],
            longitude=loc["longitude"],
            year=sim["year"],
            freq_minutes=sim["time_resolution_minutes"],
            timezone=loc["timezone"],
            altitude=loc.get("altitude", 0),
        )

        _tasks[task_id].progress = 0.25
        await ws_manager.send_progress(task_id, "clear_sky", 0.25, "Computing clear sky irradiance...")

        # Step 2: Clear sky
        cs = await asyncio.to_thread(
            compute_clear_sky_irradiance,
            latitude=loc["latitude"],
            longitude=loc["longitude"],
            year=sim["year"],
            freq_minutes=sim["time_resolution_minutes"],
            timezone=loc["timezone"],
            altitude=loc.get("altitude", 0),
            model=sim.get("dni_model", "ineichen"),
        )

        _tasks[task_id].progress = 0.5
        await ws_manager.send_progress(task_id, "ray_casting", 0.5, "Computing shadows...")

        # Step 3: Ray casting
        sun_dirs = solar.sun_direction_vectors()
        shadow_matrix = await asyncio.to_thread(
            compute_shadow_matrix,
            mesh=mesh,
            sun_directions=sun_dirs,
            sun_visible=solar.sun_visible,
            min_face_area=config.get("mesh", {}).get("min_face_area_m2", 0),
        )

        _tasks[task_id].progress = 0.75
        await ws_manager.send_progress(task_id, "irradiance", 0.75, "Computing irradiance...")

        # Step 4: Irradiance
        time_step_hours = sim["time_resolution_minutes"] / 60.0
        irr = await asyncio.to_thread(
            compute_face_irradiance,
            face_normals=mesh.face_normals,
            face_areas=mesh.area_faces,
            shadow_matrix=shadow_matrix,
            sun_directions=sun_dirs,
            dni=cs["dni"].values,
            dhi=cs["dhi"].values if sim.get("include_diffuse", True) else np.zeros(len(cs)),
            time_step_hours=time_step_hours,
        )

        # ROI
        panel_cfg = config.get("panel", {})
        elec_cfg = config.get("electricity", {})
        roi = generate_roi_report(
            irr,
            panel_efficiency=panel_cfg.get("efficiency", 0.20),
            cost_per_kw_jpy=panel_cfg.get("cost_per_kw", 250_000),
            electricity_price_jpy=elec_cfg.get("price_per_kwh", 30),
            annual_price_increase=elec_cfg.get("annual_price_increase", 0.02),
            degradation_rate=panel_cfg.get("degradation_rate", 0.005),
            lifespan_years=panel_cfg.get("lifespan_years", 25),
        )

        # Monthly GHI
        monthly = cs.resample("ME").sum() * (sim["time_resolution_minutes"] / 60) / 1000
        monthly_ghi = monthly["ghi"].values[:12].tolist()

        elapsed = time.time() - t0

        # Store in legacy _sim_store for backward compat (mesh objects, etc.)
        from .. import _sim_store

        _sim_store[task_id] = {
            "mesh": mesh,
            "irradiance": irr,
            "roi": roi,
            "config": config,
            "monthly_ghi": monthly_ghi,
        }

        # Persist to DB
        irradiance_schemas = [_irradiance_to_schema(r) for r in irr]
        roi_schema = _roi_to_schema(roi)

        try:
            from ..database import async_session

            async with async_session() as session:
                # Update task
                task_row = await session.get(Task, task_id)
                if task_row:
                    task_row.status = "complete"
                    task_row.progress = 1.0
                    task_row.step = "done"
                    task_row.message = "Simulation complete"
                    task_row.elapsed_seconds = elapsed
                    task_row.result = {
                        "n_faces": len(irr),
                        "n_proposals": len(roi.proposals),
                    }

                # Save simulation details
                sim_row = Simulation(
                    task_id=task_id,
                    latitude=req.latitude,
                    longitude=req.longitude,
                    year=req.year,
                    config=config,
                    irradiance_data=[_irradiance_to_dict(r) for r in irr],
                    roi_data=_roi_to_dict(roi),
                    monthly_ghi=monthly_ghi,
                )
                session.add(sim_row)
                await session.commit()
        except Exception:
            pass  # DB persistence is best-effort; in-memory still works

        _tasks[task_id].status = "complete"
        _tasks[task_id].progress = 1.0
        _tasks[task_id].step = "done"
        _tasks[task_id].message = "Simulation complete"
        _tasks[task_id].irradiance = irradiance_schemas
        _tasks[task_id].roi_report = roi_schema
        _tasks[task_id].elapsed_seconds = elapsed
        _tasks[task_id].monthly_ghi = monthly_ghi

        await ws_manager.send_progress(task_id, "done", 1.0, f"Complete in {elapsed:.1f}s")

    except Exception as e:
        _tasks[task_id].status = "failed"
        _tasks[task_id].message = str(e)
        await ws_manager.send_progress(task_id, "error", 0, str(e))


@router.post("/run", response_model=SimulationResult)
async def start_simulation(req: SimulationRequest, db: AsyncSession = Depends(get_db)):
    """Start a new simulation. Returns task_id for polling/WS."""
    task_id = str(uuid.uuid4())[:8]

    # Create DB record
    task_row = Task(
        id=task_id,
        task_type="simulation",
        status="pending",
        message="Queued",
        params=req.model_dump(),
    )
    db.add(task_row)
    await db.flush()

    _tasks[task_id] = SimulationResult(
        task_id=task_id,
        status="pending",
        progress=0.0,
        message="Queued",
    )
    asyncio.create_task(_run_simulation_async(task_id, req))
    return _tasks[task_id]


@router.get("/{task_id}", response_model=SimulationResult)
async def get_simulation_status(task_id: str, db: AsyncSession = Depends(get_db)):
    """Poll simulation status."""
    # Check in-memory cache first
    if task_id in _tasks:
        return _tasks[task_id]

    # Fall back to DB
    task_row = await db.get(Task, task_id)
    if not task_row:
        raise HTTPException(status_code=404, detail="Task not found")

    result = SimulationResult(
        task_id=task_row.id,
        status=task_row.status,
        progress=task_row.progress,
        step=task_row.step,
        message=task_row.message,
        elapsed_seconds=task_row.elapsed_seconds,
    )

    # Load simulation details if complete
    if task_row.status == "complete":
        stmt = select(Simulation).where(Simulation.task_id == task_id)
        sim_row = (await db.execute(stmt)).scalar_one_or_none()
        if sim_row:
            result.irradiance = [FaceIrradianceSchema(**d) for d in (sim_row.irradiance_data or [])]
            result.monthly_ghi = sim_row.monthly_ghi
            if sim_row.roi_data:
                result.roi_report = ROIReportSchema(**sim_row.roi_data)

    return result


@router.get("/{task_id}/irradiance", response_model=list[FaceIrradianceSchema])
async def get_irradiance(task_id: str, db: AsyncSession = Depends(get_db)):
    """Get per-face irradiance results."""
    task = _tasks.get(task_id)
    if task and task.status == "complete" and task.irradiance:
        return task.irradiance

    stmt = select(Simulation).where(Simulation.task_id == task_id)
    sim_row = (await db.execute(stmt)).scalar_one_or_none()
    if sim_row and sim_row.irradiance_data:
        return [FaceIrradianceSchema(**d) for d in sim_row.irradiance_data]

    raise HTTPException(status_code=404, detail="Results not available")


@router.get("/{task_id}/roi", response_model=ROIReportSchema)
async def get_roi(task_id: str, db: AsyncSession = Depends(get_db)):
    """Get ROI report."""
    task = _tasks.get(task_id)
    if task and task.status == "complete" and task.roi_report:
        return task.roi_report

    stmt = select(Simulation).where(Simulation.task_id == task_id)
    sim_row = (await db.execute(stmt)).scalar_one_or_none()
    if sim_row and sim_row.roi_data:
        return ROIReportSchema(**sim_row.roi_data)

    raise HTTPException(status_code=404, detail="Results not available")


@router.get("/{task_id}/monthly")
async def get_monthly_ghi(task_id: str, db: AsyncSession = Depends(get_db)):
    """Get monthly GHI data."""
    months = ["1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月"]

    task = _tasks.get(task_id)
    if task and task.status == "complete" and task.monthly_ghi:
        return {"months": months, "ghi_kwh_m2": task.monthly_ghi}

    stmt = select(Simulation).where(Simulation.task_id == task_id)
    sim_row = (await db.execute(stmt)).scalar_one_or_none()
    if sim_row and sim_row.monthly_ghi:
        return {"months": months, "ghi_kwh_m2": sim_row.monthly_ghi}

    raise HTTPException(status_code=404, detail="Results not available")
