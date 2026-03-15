"""Pydantic v2 schemas for ExaSense API.

Mirrors existing dataclasses (FaceIrradiance, PanelProposal, ROIReport)
and adds request/response models for all API endpoints.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ── Request Models ────────────────────────────────────────────────────────────


class SimulationRequest(BaseModel):
    latitude: float = Field(34.69, ge=-90, le=90, description="Latitude in degrees")
    longitude: float = Field(135.50, ge=-180, le=180, description="Longitude in degrees")
    year: int = Field(2025, ge=2000, le=2100)
    time_resolution_minutes: int = Field(60, description="Time step in minutes (30 or 60)")
    panel_efficiency: float = Field(20.0, ge=5, le=40, description="Panel efficiency in %")
    electricity_price_jpy: float = Field(30.0, ge=1, description="Electricity price JPY/kWh")
    mesh_source: str = Field(
        "complex",
        description="Mesh source: 'uploaded', 'simple', or 'complex'",
    )


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = None


class ReportGenerateRequest(BaseModel):
    task_id: str | None = Field(None, description="Simulation task ID (uses latest if omitted)")


class WaypointGenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=1000)
    negative_prompt: str | None = Field(None, max_length=1000)
    seed_source: Literal["demo", "upload", "viewer"] = "demo"
    seed_image_data_url: str | None = None
    template_key: str | None = Field(None, max_length=100)
    view_name: Literal["bird", "south", "west"] = "bird"
    steps: int = Field(12, ge=1, le=50)
    guidance_scale: float = Field(6.5, ge=0, le=20)
    strength: float = Field(0.65, ge=0, le=1)


# ── Response Models ───────────────────────────────────────────────────────────


class FaceIrradianceSchema(BaseModel):
    face_id: int
    annual_irradiance_kwh_m2: float
    annual_direct_kwh_m2: float
    annual_diffuse_kwh_m2: float
    area_m2: float
    normal: tuple[float, float, float]
    sun_hours: float


class PanelProposalSchema(BaseModel):
    face_id: int
    area_m2: float
    annual_generation_kwh: float
    installed_capacity_kw: float
    installation_cost_jpy: float
    annual_savings_jpy: float
    payback_years: float
    npv_25y_jpy: float
    irr_percent: float
    priority_rank: int


class ROIReportSchema(BaseModel):
    proposals: list[PanelProposalSchema]
    total_area_m2: float
    total_capacity_kw: float
    total_annual_generation_kwh: float
    total_installation_cost_jpy: float
    total_annual_savings_jpy: float
    overall_payback_years: float
    overall_npv_25y_jpy: float


class SimulationResult(BaseModel):
    task_id: str
    status: str = Field(description="pending | running | complete | failed")
    progress: float = Field(0.0, ge=0, le=1)
    step: str | None = None
    message: str | None = None
    irradiance: list[FaceIrradianceSchema] | None = None
    roi_report: ROIReportSchema | None = None
    elapsed_seconds: float | None = None
    monthly_ghi: list[float] | None = None


class MeshInfo(BaseModel):
    mesh_id: str
    num_vertices: int
    num_faces: int
    surface_area_m2: float
    bounds_min: list[float]
    bounds_max: list[float]
    download_url: str


class WSProgress(BaseModel):
    task_id: str
    step: str
    progress: float = Field(ge=0, le=1)
    message: str


class ChatResponse(BaseModel):
    response: str
    session_id: str | None = None


class ReportResponse(BaseModel):
    markdown: str
    download_urls: dict[str, str] = Field(
        description="Format -> URL mapping for downloads"
    )


class ConfigResponse(BaseModel):
    location: dict
    simulation: dict
    panel: dict
    electricity: dict
    mesh: dict


# ── Solar Animation ──────────────────────────────────────────────────────────


class SunPositionEntry(BaseModel):
    time: str = Field(description="HH:MM format")
    azimuth: float = Field(description="Degrees, 0=North, clockwise")
    elevation: float = Field(description="Degrees above horizon")
    direction_y_up: list[float] = Field(
        description="Unit vector [x, y, z] in Three.js Y-up coords"
    )


class SunPositionsResponse(BaseModel):
    date: str
    latitude: float
    longitude: float
    freq_minutes: int
    positions: list[SunPositionEntry]


class ShadowTimelineResponse(BaseModel):
    date: str
    mesh_source: str
    n_faces: int
    n_steps: int
    times: list[str]
    shadow_matrix: list[list[bool]] = Field(
        description="(n_steps x n_faces) — True if face is illuminated"
    )


# ── Reconstruction ──────────────────────────────────────────────────────────


class ReconstructionStatus(BaseModel):
    task_id: str
    status: str = Field(description="pending | running | complete | failed")
    progress: float = Field(0.0, ge=0, le=1)
    step: str | None = None
    message: str | None = None
    mesh_id: str | None = None


# ── Waypoint / World Model ──────────────────────────────────────────────────


class WaypointMetricSchema(BaseModel):
    annual_generation_kwh: float
    co2_reduction_tons: float
    installed_capacity_kw: float
    estimated_payback_years: float


class WaypointVariantSchema(BaseModel):
    id: str
    view_name: str
    label: str
    image_data_url: str


class WaypointGenerateResponse(BaseModel):
    request_id: str
    status: str = Field(description="queued | running | complete | failed")
    mock_mode: bool = True
    prompt: str
    negative_prompt: str | None = None
    view_name: str
    seed_image_data_url: str
    result_image_data_url: str
    metrics: WaypointMetricSchema
    variants: list[WaypointVariantSchema]
    latency_ms: int


class WaypointStatusResponse(BaseModel):
    service_status: str = Field(description="ready | loading | unavailable")
    mock_mode: bool = True
    model_name: str
    device: str
    gpu_available: bool
    model_loaded: bool
    queue_depth: int = Field(0, ge=0)
    vram_used_gb: float | None = None


# ── Rust Inspection (OpenVLA) ────────────────────────────────────────────────


class RustInspectionRunRequest(BaseModel):
    seed: int = Field(42, description="Random seed for mock data generation")
    grid_rows: int = Field(5, ge=3, le=10, description="Grid rows (patch divisions)")
    grid_cols: int = Field(5, ge=3, le=10, description="Grid cols (patch divisions)")
    max_steps: int = Field(200, ge=10, le=1000, description="Max agent steps")
    coverage_threshold: float = Field(0.95, ge=0.5, le=1.0, description="Target coverage rate")


class RustInspectionMetricsSchema(BaseModel):
    coverage_rate: float = Field(description="Visited rust patches / total rust patches")
    total_steps: int = Field(description="Total agent steps taken")
    backtrack_count: int = Field(description="Number of backtracks")
    component_jumps: int = Field(description="Inter-component jump count")
    rust_patch_count: int = Field(description="Total rust patches detected")
    visited_rust_count: int = Field(description="Rust patches successfully visited")
    grid_rows: int
    grid_cols: int


class RustInspectionResultSchema(BaseModel):
    trajectory_image_data_url: str = Field(description="SVG data URL of trajectory visualization")
    source_image_data_url: str | None = Field(None, description="Source image if uploaded")
    metrics: RustInspectionMetricsSchema


class RustInspectionRunResponse(BaseModel):
    request_id: str
    status: str = Field(description="complete | failed")
    mock_mode: bool = True
    result: RustInspectionResultSchema


class RustInspectionStatusResponse(BaseModel):
    service_status: str = Field(description="ready | loading | unavailable")
    mock_mode: bool = True
    model_name: str
    device: str
    gpu_available: bool
    model_loaded: bool
    vram_used_gb: float | None = None
