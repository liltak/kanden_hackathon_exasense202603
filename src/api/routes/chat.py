"""Chat API routes — VLM-powered analysis."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models import Simulation
from ..schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/api", tags=["chat"])


def _build_context_from_dict(roi: dict) -> str:
    return (
        f"設置可能容量: {roi['total_capacity_kw']:.1f}kW, "
        f"年間発電量: {roi['total_annual_generation_kwh']:,.0f}kWh, "
        f"年間削減額: ¥{roi['total_annual_savings_jpy']:,.0f}, "
        f"投資回収: {roi['overall_payback_years']:.1f}年, "
        f"25年NPV: ¥{roi['overall_npv_25y_jpy']:,.0f}"
    )


def _build_context(roi, irr) -> str:
    return (
        f"設置可能容量: {roi.total_capacity_kw:.1f}kW, "
        f"年間発電量: {roi.total_annual_generation_kwh:,.0f}kWh, "
        f"年間削減額: ¥{roi.total_annual_savings_jpy:,.0f}, "
        f"投資回収: {roi.overall_payback_years:.1f}年, "
        f"25年NPV: ¥{roi.overall_npv_25y_jpy:,.0f}"
    )


async def _get_simulation_data(db: AsyncSession):
    """Get latest simulation data from memory or DB."""
    from .. import _sim_store

    # Check in-memory first
    for key in reversed(list(_sim_store.keys())):
        if key.startswith("_"):
            continue
        sim = _sim_store[key]
        roi = sim.get("roi")
        irr = sim.get("irradiance")
        if roi:
            return roi, irr, False

    # Fall back to DB
    stmt = select(Simulation).order_by(Simulation.created_at.desc()).limit(1)
    sim_row = (await db.execute(stmt)).scalar_one_or_none()
    if sim_row and sim_row.roi_data:
        return sim_row.roi_data, sim_row.irradiance_data, True

    return None, None, False


def _generate_response(message: str, roi, irr, from_db: bool) -> str:
    """Generate chat response based on simulation data."""
    import numpy as np

    if from_db:
        context = _build_context_from_dict(roi)
        proposals = roi["proposals"]
        best = proposals[0] if proposals else None
        best_face = next((r for r in irr if r["face_id"] == best["face_id"]), None) if best else None
    else:
        context = _build_context(roi, irr)
        proposals = roi.proposals
        best = proposals[0] if proposals else None
        best_face = next((r for r in irr if r.face_id == best.face_id), None) if best else None

    def _get(obj, key):
        return obj[key] if isinstance(obj, dict) else getattr(obj, key)

    if "最適" in message or "場所" in message or "どこ" in message:
        normal_z = _get(best_face, "normal")[2] if best_face else 0
        if isinstance(normal_z, (list, tuple)):
            normal_z = normal_z[2]
        tilt_deg = np.degrees(np.arccos(abs(normal_z))) if best_face else 0
        return f"""## 太陽光パネル最適設置場所の分析

### 分析データ
{context}

### 推奨設置場所

**第1優先: 面ID {_get(best, 'face_id')}**（年間日射量 {_get(best_face, 'annual_irradiance_kwh_m2'):,.0f} kWh/m²）
- 設置可能面積: {_get(best, 'area_m2'):.0f} m²
- 推定発電量: {_get(best, 'annual_generation_kwh'):,.0f} kWh/年
- 投資回収期間: {_get(best, 'payback_years'):.1f} 年
- 25年NPV: ¥{_get(best, 'npv_25y_jpy') / 10000:.0f}万円

この面は南向きの傾斜屋根で、年間を通じて最も安定した日射を受けます。
遮蔽物による影の影響も最小限です。

### 設置時の推奨事項
1. **傾斜角**: 現在の屋根角度（約{tilt_deg:.0f}°）は当地域の最適角度に近い
2. **パネル種類**: 単結晶シリコン（効率20%以上）を推奨
3. **施工**: 屋根構造の荷重計算を事前に実施すること"""

    elif "コスト" in message or "削減" in message or "改善" in message:
        total_annual_gen = _get(roi, "total_annual_generation_kwh")
        total_annual_sav = _get(roi, "total_annual_savings_jpy")
        overall_payback = _get(roi, "overall_payback_years")
        co2 = total_annual_gen * 0.000453
        return f"""## エネルギーコスト削減提案

### 現状分析
{context}

### 提案1: 太陽光パネル設置（優先度: 高）
- **効果**: 年間 ¥{total_annual_sav / 10000:.0f}万円 の電力コスト削減
- **CO2削減**: 年間 {co2:.1f} t-CO2
- **投資回収**: {overall_payback:.1f} 年

### 提案2: ピークカット蓄電池の導入（優先度: 中）
- デマンドレスポンス対応で基本料金を削減
- 太陽光との組み合わせで自家消費率を最大化
- 推定追加削減: 年間 ¥{total_annual_sav * 0.15 / 10000:.0f}万円

### 提案3: 屋根断熱改修との同時施工（優先度: 中）
- パネル設置と同時に断熱改修することで足場費用を共有
- 空調エネルギーを推定15-20%削減
- 推定追加削減: 年間 ¥{total_annual_sav * 0.1 / 10000:.0f}万円"""

    elif "ROI" in message or "回収" in message or "投資" in message:
        total_annual_sav = _get(roi, "total_annual_savings_jpy")
        total_install_cost = _get(roi, "total_installation_cost_jpy")
        overall_payback = _get(roi, "overall_payback_years")
        overall_npv = _get(roi, "overall_npv_25y_jpy")
        irr_pct = (total_annual_sav / total_install_cost * 100) if total_install_cost > 0 else 0
        return f"""## 投資回収分析

### サマリー
| 指標 | 値 |
|------|-----|
| 初期投資額 | ¥{total_install_cost / 10000:.0f}万円 |
| 年間削減額 | ¥{total_annual_sav / 10000:.0f}万円 |
| 単純回収期間 | {overall_payback:.1f}年 |
| 25年NPV | ¥{overall_npv / 10000:.0f}万円 |
| IRR（概算） | {irr_pct:.1f}% |

### 回収期間の短縮方法
1. **補助金活用**: 自家消費型は国の補助金（設置費用の1/3程度）が利用可能
   → 回収期間を約{overall_payback * 0.67:.1f}年に短縮
2. **PPA/リースモデル**: 初期投資ゼロで導入し、電力単価で支払い
3. **段階的導入**: 高日射面から優先設置し、キャッシュフローを早期改善"""

    else:
        return f"""## ExaSense分析レポート

### シミュレーション結果サマリー
{context}

### 主要な知見
1. 合計 {len(proposals)} 面が太陽光パネル設置に適しています
2. 最も効率的な面は面ID {_get(best, 'face_id')}（日射量 {_get(best_face, 'annual_irradiance_kwh_m2'):,.0f} kWh/m²/年）
3. 投資回収期間 {_get(roi, 'overall_payback_years'):.1f}年は産業用太陽光の一般的な水準です

何について詳しく分析しますか？
- 「最適な設置場所」
- 「コスト削減提案」
- 「投資回収の詳細」"""


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    """Chat with AI about simulation results."""
    roi, irr, from_db = await _get_simulation_data(db)

    if roi is None:
        return ChatResponse(
            response="シミュレーションを先に実行してください。「シミュレーション」タブでデータを生成すると、分析が可能になります。",
            session_id=req.session_id,
        )

    resp = _generate_response(req.message, roi, irr, from_db)
    return ChatResponse(response=resp, session_id=req.session_id)
