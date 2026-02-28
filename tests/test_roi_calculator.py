"""Tests for ROI calculator."""

import pytest

from src.simulation.irradiance import FaceIrradiance
from src.simulation.roi_calculator import (
    calculate_panel_proposal,
    generate_roi_report,
)


def make_test_face(
    face_id: int = 0,
    irradiance: float = 1200,
    area: float = 100,
) -> FaceIrradiance:
    return FaceIrradiance(
        face_id=face_id,
        annual_irradiance_kwh_m2=irradiance,
        annual_direct_kwh_m2=irradiance * 0.8,
        annual_diffuse_kwh_m2=irradiance * 0.2,
        area_m2=area,
        normal=(0, 0, 1),
        sun_hours=2000,
    )


class TestCalculatePanelProposal:
    def test_basic_calculation(self):
        face = make_test_face(irradiance=1200, area=100)
        proposal = calculate_panel_proposal(face)
        assert proposal is not None
        assert proposal.annual_generation_kwh > 0
        assert proposal.installation_cost_jpy > 0
        assert proposal.payback_years > 0
        assert proposal.payback_years < 30

    def test_high_irradiance_better_roi(self):
        face_high = make_test_face(irradiance=1500, area=100)
        face_low = make_test_face(irradiance=800, area=100)
        p_high = calculate_panel_proposal(face_high)
        p_low = calculate_panel_proposal(face_low)
        assert p_high is not None and p_low is not None
        assert p_high.npv_25y_jpy > p_low.npv_25y_jpy

    def test_too_small_area_returns_none(self):
        face = make_test_face(area=0.5)
        proposal = calculate_panel_proposal(face)
        assert proposal is None

    def test_low_irradiance_returns_none(self):
        face = make_test_face(irradiance=50)
        proposal = calculate_panel_proposal(face)
        assert proposal is None


class TestGenerateROIReport:
    def test_report_with_mixed_faces(self):
        faces = [
            make_test_face(0, irradiance=1200, area=100),
            make_test_face(1, irradiance=1000, area=50),
            make_test_face(2, irradiance=200, area=100),   # too low
            make_test_face(3, irradiance=1100, area=0.5),   # too small
        ]
        report = generate_roi_report(faces)
        assert len(report.proposals) == 2  # only face 0 and 1
        assert report.proposals[0].priority_rank == 1
        assert report.total_capacity_kw > 0

    def test_empty_report(self):
        faces = [make_test_face(0, irradiance=100)]
        report = generate_roi_report(faces)
        assert len(report.proposals) == 0
