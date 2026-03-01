"""ROI (Return on Investment) calculator for solar panel installation.

Computes financial metrics including payback period, NPV, and annual savings
based on irradiance simulation results and economic parameters.
"""

from dataclasses import dataclass

import numpy as np

from .irradiance import FaceIrradiance


@dataclass
class PanelProposal:
    """Solar panel installation proposal for a specific face."""

    face_id: int
    area_m2: float
    annual_generation_kwh: float
    installed_capacity_kw: float
    installation_cost_jpy: float
    annual_savings_jpy: float
    payback_years: float
    npv_25y_jpy: float
    irr_percent: float
    priority_rank: int  # 1 = best


@dataclass
class ROIReport:
    """Overall ROI report for the facility."""

    proposals: list[PanelProposal]
    total_area_m2: float
    total_capacity_kw: float
    total_annual_generation_kwh: float
    total_installation_cost_jpy: float
    total_annual_savings_jpy: float
    overall_payback_years: float
    overall_npv_25y_jpy: float


def calculate_panel_proposal(
    face: FaceIrradiance,
    panel_efficiency: float = 0.20,
    cost_per_kw_jpy: float = 250_000,
    electricity_price_jpy: float = 30,
    annual_price_increase: float = 0.02,
    degradation_rate: float = 0.005,
    lifespan_years: int = 25,
    discount_rate: float = 0.03,
    usable_area_ratio: float = 0.7,
) -> PanelProposal | None:
    """Calculate ROI for installing solar panels on a specific face.

    Args:
        face: Irradiance data for the face.
        panel_efficiency: Solar panel conversion efficiency (0-1).
        cost_per_kw_jpy: Installation cost per kW in JPY.
        electricity_price_jpy: Current electricity price per kWh in JPY.
        annual_price_increase: Annual electricity price increase rate.
        degradation_rate: Annual panel degradation rate.
        lifespan_years: Panel lifespan in years.
        discount_rate: Discount rate for NPV calculation.
        usable_area_ratio: Fraction of face area usable for panels.

    Returns:
        PanelProposal or None if face is not suitable.
    """
    usable_area = face.area_m2 * usable_area_ratio

    if usable_area < 0.001 or face.annual_irradiance_kwh_m2 < 100:
        return None

    # 1 kWp ≈ ~5-7 m^2 of panel area (assuming ~150-200 W/m^2 panel rating)
    panel_rating_w_per_m2 = 200  # typical modern panel
    installed_capacity_kw = usable_area * panel_rating_w_per_m2 / 1000

    annual_generation_kwh = face.annual_irradiance_kwh_m2 * usable_area * panel_efficiency
    installation_cost = installed_capacity_kw * cost_per_kw_jpy

    # NPV calculation
    npv = -installation_cost
    cumulative_cash = -installation_cost
    payback_years = float("inf")

    for year in range(1, lifespan_years + 1):
        degraded_generation = annual_generation_kwh * (1 - degradation_rate) ** year
        price = electricity_price_jpy * (1 + annual_price_increase) ** year
        annual_saving = degraded_generation * price
        npv += annual_saving / (1 + discount_rate) ** year
        cumulative_cash += annual_saving
        if cumulative_cash >= 0 and payback_years == float("inf"):
            # Linear interpolation for fractional year
            prev_cash = cumulative_cash - annual_saving
            payback_years = year - 1 + (-prev_cash / annual_saving)

    # Simple IRR approximation
    first_year_saving = annual_generation_kwh * electricity_price_jpy
    if installation_cost > 0:
        irr = (first_year_saving / installation_cost) * 100
    else:
        irr = 0.0

    return PanelProposal(
        face_id=face.face_id,
        area_m2=usable_area,
        annual_generation_kwh=round(annual_generation_kwh, 1),
        installed_capacity_kw=round(installed_capacity_kw, 2),
        installation_cost_jpy=round(installation_cost),
        annual_savings_jpy=round(first_year_saving),
        payback_years=round(payback_years, 1),
        npv_25y_jpy=round(npv),
        irr_percent=round(irr, 1),
        priority_rank=0,
    )


def generate_roi_report(
    irradiance_results: list[FaceIrradiance],
    min_irradiance_kwh_m2: float = 800,
    **kwargs,
) -> ROIReport:
    """Generate comprehensive ROI report for all suitable faces.

    Args:
        irradiance_results: Per-face irradiance data.
        min_irradiance_kwh_m2: Minimum annual irradiance to consider.
        **kwargs: Additional parameters passed to calculate_panel_proposal.
    """
    proposals = []
    for face in irradiance_results:
        if face.annual_irradiance_kwh_m2 < min_irradiance_kwh_m2:
            continue
        proposal = calculate_panel_proposal(face, **kwargs)
        if proposal is not None:
            proposals.append(proposal)

    # Rank by NPV (best first)
    proposals.sort(key=lambda p: p.npv_25y_jpy, reverse=True)
    for i, p in enumerate(proposals):
        p.priority_rank = i + 1

    total_area = sum(p.area_m2 for p in proposals)
    total_cap = sum(p.installed_capacity_kw for p in proposals)
    total_gen = sum(p.annual_generation_kwh for p in proposals)
    total_cost = sum(p.installation_cost_jpy for p in proposals)
    total_savings = sum(p.annual_savings_jpy for p in proposals)
    total_npv = sum(p.npv_25y_jpy for p in proposals)

    if total_savings > 0:
        overall_payback = total_cost / total_savings
    else:
        overall_payback = float("inf")

    return ROIReport(
        proposals=proposals,
        total_area_m2=round(total_area, 1),
        total_capacity_kw=round(total_cap, 2),
        total_annual_generation_kwh=round(total_gen, 1),
        total_installation_cost_jpy=round(total_cost),
        total_annual_savings_jpy=round(total_savings),
        overall_payback_years=round(overall_payback, 1),
        overall_npv_25y_jpy=round(total_npv),
    )
