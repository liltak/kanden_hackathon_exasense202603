"""Report generation API routes."""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models import Simulation
from ..schemas import ReportGenerateRequest, ReportResponse

router = APIRouter(prefix="/api/report", tags=["report"])

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "simulation_results"


def _try_upload_to_minio(task_id: str, report: str, json_data: list) -> bool:
    """Upload report files to MinIO. Returns True on success."""
    try:
        from ..storage import BUCKET_REPORTS, upload_bytes

        upload_bytes(
            BUCKET_REPORTS,
            f"reports/{task_id}/report.md",
            report.encode("utf-8"),
            content_type="text/markdown; charset=utf-8",
        )
        upload_bytes(
            BUCKET_REPORTS,
            f"reports/{task_id}/irradiance_data.json",
            json.dumps(json_data, indent=2, ensure_ascii=False).encode("utf-8"),
            content_type="application/json",
        )
        return True
    except Exception:
        return False


@router.post("/generate", response_model=ReportResponse)
async def generate_report(req: ReportGenerateRequest | None = None, db: AsyncSession = Depends(get_db)):
    """Generate a comprehensive markdown report from simulation results."""
    from .. import _sim_store

    # Find simulation data — check in-memory first, then DB
    sim = None
    task_id = req.task_id if req else None

    if task_id and task_id in _sim_store:
        sim = _sim_store[task_id]
    elif not task_id:
        for key in reversed(list(_sim_store.keys())):
            if key.startswith("_"):
                continue
            sim = _sim_store[key]
            task_id = key
            break

    # Fall back to DB if not in memory
    if sim is None:
        stmt = select(Simulation).order_by(Simulation.created_at.desc())
        if task_id:
            stmt = stmt.where(Simulation.task_id == task_id)
        sim_row = (await db.execute(stmt)).scalars().first()
        if sim_row:
            task_id = sim_row.task_id
            # Build a compatible sim dict from DB data
            sim = {
                "irradiance": sim_row.irradiance_data,
                "roi": sim_row.roi_data,
                "config": sim_row.config,
                "monthly_ghi": sim_row.monthly_ghi,
                "_from_db": True,
            }

    if sim is None:
        raise HTTPException(status_code=404, detail="No simulation results found. Run a simulation first.")

    # Handle both in-memory objects and DB dicts
    from_db = sim.get("_from_db", False)

    if from_db:
        roi = sim["roi"]
        irr = sim["irradiance"]
        config = sim["config"]
        total_capacity_kw = roi["total_capacity_kw"]
        total_annual_gen = roi["total_annual_generation_kwh"]
        total_annual_sav = roi["total_annual_savings_jpy"]
        total_install_cost = roi["total_installation_cost_jpy"]
        overall_payback = roi["overall_payback_years"]
        overall_npv = roi["overall_npv_25y_jpy"]
        total_area = roi["total_area_m2"]
        proposals = roi["proposals"]
    else:
        roi = sim["roi"]
        irr = sim["irradiance"]
        config = sim["config"]
        total_capacity_kw = roi.total_capacity_kw
        total_annual_gen = roi.total_annual_generation_kwh
        total_annual_sav = roi.total_annual_savings_jpy
        total_install_cost = roi.total_installation_cost_jpy
        overall_payback = roi.overall_payback_years
        overall_npv = roi.overall_npv_25y_jpy
        total_area = roi.total_area_m2
        proposals = roi.proposals

    loc = config["location"]
    co2 = total_annual_gen * 0.000453

    report = f"""# ExaSense 太陽光パネル設置提案レポート

**生成日時**: {time.strftime('%Y年%m月%d日 %H:%M')}
**対象地点**: {loc['latitude']}°N, {loc['longitude']}°E

---

## 1. エグゼクティブサマリー

本レポートは、対象施設における太陽光パネル設置の最適化分析結果をまとめたものです。

| 指標 | 値 |
|------|-----|
| 設置可能容量 | **{total_capacity_kw:.1f} kW** |
| 年間発電量 | **{total_annual_gen:,.0f} kWh** ({total_annual_gen / 1000:.1f} MWh) |
| 年間電力コスト削減額 | **¥{total_annual_sav:,.0f}** |
| 年間CO2削減量 | **{co2:.1f} t-CO2** |
| 初期投資額 | ¥{total_install_cost:,.0f} |
| 投資回収期間 | **{overall_payback:.1f} 年** |
| 25年間NPV | **¥{overall_npv:,.0f}** |

## 2. 施設分析

- 解析対象メッシュ面数: {len(irr)} 面
- パネル設置適合面数: {len(proposals)} 面
- 合計設置可能面積: {total_area:.0f} m²

## 3. 設置優先順位

| 順位 | 面ID | 面積(m²) | 年間日射量(kWh/m²) | 発電量(kWh/年) | NPV(万円) | 回収(年) |
|:----:|:----:|--------:|-----------------:|-----------:|--------:|-------:|
"""

    for p in (proposals[:10] if isinstance(proposals, list) else list(proposals)[:10]):
        if from_db:
            face_id = p["face_id"]
            area = p["area_m2"]
            gen = p["annual_generation_kwh"]
            npv = p["npv_25y_jpy"]
            payback = p["payback_years"]
            rank = p["priority_rank"]
            face = next((r for r in irr if r["face_id"] == face_id), None)
            irr_val = face["annual_irradiance_kwh_m2"] if face else 0
        else:
            face_id = p.face_id
            area = p.area_m2
            gen = p.annual_generation_kwh
            npv = p.npv_25y_jpy
            payback = p.payback_years
            rank = p.priority_rank
            face = next((r for r in irr if r.face_id == face_id), None)
            irr_val = face.annual_irradiance_kwh_m2 if face else 0

        report += (
            f"| {rank} | {face_id} | {area:.0f} | "
            f"{irr_val:,.0f} | "
            f"{gen:,.0f} | "
            f"{npv / 10000:.0f} | "
            f"{payback:.1f} |\n"
        )

    report += f"""
## 4. 経済性分析

### 4.1 前提条件
- パネル効率: {config['panel']['efficiency'] * 100:.0f}%
- 電気料金: ¥{config['electricity']['price_per_kwh']}/kWh
- 電気料金上昇率: {config['electricity']['annual_price_increase'] * 100:.1f}%/年
- パネル劣化率: {config['panel']['degradation_rate'] * 100:.1f}%/年
- 設置単価: ¥{config['panel']['cost_per_kw']:,.0f}/kW
- 評価期間: {config['panel']['lifespan_years']}年

### 4.2 投資効果
- 25年間の累計発電量: {total_annual_gen * 25 * 0.94:,.0f} kWh（劣化考慮）
- 25年間の累計コスト削減: ¥{total_annual_sav * 25 * 1.25:,.0f}（電気料金上昇考慮）

## 5. 推奨アクション

1. **即時実施**: 優先順位1-3の面へのパネル設置（最も投資効率が高い）
2. **補助金申請**: 自家消費型太陽光発電設備の補助金を確認
3. **詳細設計**: 構造計算・電気設計の実施
4. **段階導入**: 第1期で優先度上位面に設置し、効果確認後に拡大

---

*本レポートはExaSenseシミュレーションエンジンにより自動生成されました。*
*実際の設置には専門家による現地調査と詳細設計が必要です。*
"""

    # Build JSON data
    if from_db:
        json_data = [
            {
                "face_id": r["face_id"],
                "annual_irradiance_kwh_m2": round(r["annual_irradiance_kwh_m2"], 2),
                "area_m2": round(r["area_m2"], 4),
                "sun_hours": round(r["sun_hours"], 1),
            }
            for r in irr
        ]
    else:
        json_data = [
            {
                "face_id": r.face_id,
                "annual_irradiance_kwh_m2": round(r.annual_irradiance_kwh_m2, 2),
                "area_m2": round(r.area_m2, 4),
                "sun_hours": round(r.sun_hours, 1),
            }
            for r in irr
        ]

    # Upload to MinIO (best-effort) + save locally as fallback
    _try_upload_to_minio(task_id, report, json_data)

    output_dir = RESULTS_DIR / "latest"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.md").write_text(report)
    (output_dir / "irradiance_data.json").write_text(
        json.dumps(json_data, indent=2, ensure_ascii=False)
    )

    return ReportResponse(
        markdown=report,
        download_urls={
            "markdown": f"/api/report/{task_id}/download/md",
            "json": f"/api/report/{task_id}/download/json",
        },
    )


@router.get("/{task_id}/download/{fmt}")
async def download_report(task_id: str, fmt: str):
    """Download report in specified format (md or json)."""
    # Try MinIO first
    try:
        from ..storage import BUCKET_REPORTS, download_bytes

        if fmt == "md":
            data = download_bytes(BUCKET_REPORTS, f"reports/{task_id}/report.md")
            return Response(
                content=data,
                media_type="text/markdown",
                headers={"Content-Disposition": 'attachment; filename="exasense_report.md"'},
            )
        elif fmt == "json":
            data = download_bytes(BUCKET_REPORTS, f"reports/{task_id}/irradiance_data.json")
            return Response(
                content=data,
                media_type="application/json",
                headers={"Content-Disposition": 'attachment; filename="irradiance_data.json"'},
            )
    except Exception:
        pass

    # Fall back to local file
    output_dir = RESULTS_DIR / "latest"
    if fmt == "md":
        path = output_dir / "report.md"
        media_type = "text/markdown"
        filename = "exasense_report.md"
    elif fmt == "json":
        path = output_dir / "irradiance_data.json"
        media_type = "application/json"
        filename = "irradiance_data.json"
    else:
        raise HTTPException(status_code=400, detail="Format must be 'md' or 'json'")

    if not path.exists():
        raise HTTPException(status_code=404, detail="Report not generated yet")

    content = path.read_bytes()
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
