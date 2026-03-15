"""Mock Waypoint API routes for frontend integration."""

from __future__ import annotations

import asyncio
import hashlib
import json
from urllib.parse import quote

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..schemas import (
    WaypointGenerateRequest,
    WaypointGenerateResponse,
    WaypointMetricSchema,
    WaypointStatusResponse,
    WaypointVariantSchema,
)

router = APIRouter(prefix="/api/waypoint", tags=["waypoint"])

_VIEW_LABELS = {
    "bird": "Bird View",
    "south": "South View",
    "west": "West View",
}

_VIEW_PALETTES = {
    "bird": {"base": "#0f766e", "accent": "#34d399"},
    "south": {"base": "#1d4ed8", "accent": "#60a5fa"},
    "west": {"base": "#b45309", "accent": "#f59e0b"},
}


def _build_svg_data_url(
    *,
    title: str,
    subtitle: str,
    base_color: str,
    accent_color: str,
    badge_text: str,
) -> str:
    svg = f"""
<svg
  xmlns="http://www.w3.org/2000/svg"
  width="1280"
  height="720"
  viewBox="0 0 1280 720"
  fill="none"
>
  <defs>
    <linearGradient
      id="bg"
      x1="0"
      y1="0"
      x2="1280"
      y2="720"
      gradientUnits="userSpaceOnUse"
    >
      <stop stop-color="{base_color}"/>
      <stop offset="1" stop-color="#020617"/>
    </linearGradient>
    <linearGradient
      id="panel"
      x1="260"
      y1="150"
      x2="980"
      y2="560"
      gradientUnits="userSpaceOnUse"
    >
      <stop stop-color="{accent_color}" stop-opacity="0.95"/>
      <stop offset="1" stop-color="#E2E8F0" stop-opacity="0.15"/>
    </linearGradient>
  </defs>
  <rect width="1280" height="720" rx="36" fill="url(#bg)"/>
  <circle cx="1070" cy="120" r="78" fill="{accent_color}" fill-opacity="0.35"/>
  <path
    d="M160 560L470 250L700 420L980 190L1120 320"
    stroke="#E2E8F0"
    stroke-opacity="0.25"
    stroke-width="8"
  />
  <path d="M170 540L470 240L700 410L990 180L1120 300" stroke="{accent_color}" stroke-width="3"/>
  <path
    d="M170 540L470 240L700 410L990 180L1120 300"
    stroke="{accent_color}"
    stroke-width="3"
  />
  <rect
    x="250"
    y="190"
    width="690"
    height="310"
    rx="30"
    fill="#0F172A"
    fill-opacity="0.58"
    stroke="#E2E8F0"
    stroke-opacity="0.12"
  />
  <path
    d="M300 445L520 270L690 335L905 220"
    stroke="#E2E8F0"
    stroke-opacity="0.28"
    stroke-width="10"
  />
  <path
    d="M300 465L520 290L690 355L905 240"
    stroke="url(#panel)"
    stroke-width="22"
    stroke-linecap="round"
  />
  <rect x="96" y="92" width="264" height="40" rx="20" fill="#0F172A" fill-opacity="0.66"/>
  <text
    x="122"
    y="118"
    fill="#E2E8F0"
    font-size="21"
    font-family="Arial, Helvetica, sans-serif"
  >
    {badge_text}
  </text>
  <text
    x="96"
    y="616"
    fill="#F8FAFC"
    font-size="54"
    font-weight="700"
    font-family="Arial, Helvetica, sans-serif"
  >
    {title}
  </text>
  <text
    x="96"
    y="662"
    fill="#CBD5E1"
    font-size="28"
    font-family="Arial, Helvetica, sans-serif"
  >
    {subtitle}
  </text>
</svg>
""".strip()
    return f"data:image/svg+xml;charset=UTF-8,{quote(svg)}"


def _estimate_metrics(req: WaypointGenerateRequest) -> WaypointMetricSchema:
    digest = hashlib.sha256(
        (
            f"{req.prompt}|{req.view_name}|{req.steps}|"
            f"{req.guidance_scale}|{req.strength}"
        ).encode("utf-8")
    ).hexdigest()
    score = int(digest[:8], 16)
    annual_generation = 182_000 + (score % 48_000)
    capacity_kw = 118 + (score % 52)
    co2_tons = round(annual_generation * 0.000445, 1)
    payback_years = round(5.1 + ((score >> 5) % 18) / 10, 1)
    return WaypointMetricSchema(
        annual_generation_kwh=float(annual_generation),
        co2_reduction_tons=co2_tons,
        installed_capacity_kw=float(capacity_kw),
        estimated_payback_years=payback_years,
    )


def _build_seed_image(req: WaypointGenerateRequest) -> str:
    if req.seed_image_data_url:
        return req.seed_image_data_url

    palette = _VIEW_PALETTES[req.view_name]
    return _build_svg_data_url(
        title=f"Seed / {_VIEW_LABELS[req.view_name]}",
        subtitle="Factory roof snapshot for world model prompting",
        base_color=palette["base"],
        accent_color="#cbd5e1",
        badge_text="Seed Image",
    )


def _build_variants(req: WaypointGenerateRequest) -> list[WaypointVariantSchema]:
    metrics = _estimate_metrics(req)
    variants: list[WaypointVariantSchema] = []
    for view_name, label in _VIEW_LABELS.items():
        palette = _VIEW_PALETTES[view_name]
        image_data_url = _build_svg_data_url(
            title=f"After / {label}",
            subtitle=(
                f"{metrics.annual_generation_kwh:,.0f} kWh/yr | "
                f"{metrics.installed_capacity_kw:.0f} kW"
            ),
            base_color=palette["base"],
            accent_color=palette["accent"],
            badge_text="Waypoint Mock",
        )
        variants.append(
            WaypointVariantSchema(
                id=f"{view_name}-{req.steps}-{int(req.guidance_scale * 10)}",
                view_name=view_name,
                label=label,
                image_data_url=image_data_url,
            )
        )
    return variants


def _build_response(req: WaypointGenerateRequest) -> WaypointGenerateResponse:
    metrics = _estimate_metrics(req)
    seed_image = _build_seed_image(req)
    variants = _build_variants(req)
    selected = next(
        (variant for variant in variants if variant.view_name == req.view_name),
        variants[0],
    )
    latency_ms = 900 + req.steps * 60 + int(req.guidance_scale * 25)
    return WaypointGenerateResponse(
        request_id=f"waypoint-{hashlib.md5(req.prompt.encode('utf-8')).hexdigest()[:10]}",
        status="complete",
        mock_mode=True,
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        view_name=req.view_name,
        seed_image_data_url=seed_image,
        result_image_data_url=selected.image_data_url,
        metrics=metrics,
        variants=variants,
        latency_ms=latency_ms,
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


@router.get("/status", response_model=WaypointStatusResponse)
async def waypoint_status() -> WaypointStatusResponse:
    gpu_available, device, vram_used_gb = _detect_gpu()
    return WaypointStatusResponse(
        service_status="ready",
        mock_mode=True,
        model_name="Overworld/Waypoint-1-Small (mock bridge)",
        device=device,
        gpu_available=gpu_available,
        model_loaded=False,
        queue_depth=0,
        vram_used_gb=vram_used_gb,
    )


@router.post("/generate", response_model=WaypointGenerateResponse)
async def generate_waypoint(req: WaypointGenerateRequest) -> WaypointGenerateResponse:
    return _build_response(req)


@router.post("/stream")
async def stream_waypoint(req: WaypointGenerateRequest) -> StreamingResponse:
    async def event_stream():
        progress_events = [
            {
                "event": "progress",
                "status": "queued",
                "progress": 0.1,
                "message": "Seed image accepted",
            },
            {
                "event": "progress",
                "status": "running",
                "progress": 0.45,
                "message": "Conditioning prompt",
            },
            {
                "event": "progress",
                "status": "running",
                "progress": 0.82,
                "message": "Rendering mock frames",
            },
        ]
        for payload in progress_events:
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(0.12)

        result = _build_response(req).model_dump(mode="json")
        payload = {"event": "complete", "status": "complete", "result": result}
        yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
