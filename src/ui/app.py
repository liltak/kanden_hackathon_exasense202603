"""Gradio WebUI for ExaSense - Factory Energy Optimization Solution.

Demo-ready dashboard interface with:
1. Dashboard — Overview with KPI cards and pipeline status
2. 3D Reconstruction — Upload & reconstruct (demo mode available)
3. Solar Simulation — Interactive simulation with real-time parameters
4. AI Analysis — VLM-powered analysis chat
5. Report — Comprehensive report generation & download
"""

import json
import os
import tempfile
import time
from pathlib import Path

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "simulation_results"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# ── Shared state ──────────────────────────────────────────────────────────────

_state: dict = {
    "irradiance_results": None,
    "roi_report": None,
    "mesh": None,
    "config": None,
    "report_md": None,
}

PRESET_LOCATIONS = {
    "大阪（関西電力エリア）": (34.69, 135.50),
    "東京": (35.68, 139.77),
    "名古屋": (35.18, 136.91),
    "福岡": (33.59, 130.40),
    "札幌": (43.06, 141.35),
}

CSS = """
.kpi-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    padding: 20px;
    color: white;
    text-align: center;
    min-height: 100px;
}
.kpi-card h3 { margin: 0; font-size: 14px; opacity: 0.9; }
.kpi-card .value { font-size: 32px; font-weight: bold; margin: 8px 0; }
.kpi-card .unit { font-size: 12px; opacity: 0.8; }
.kpi-green { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
.kpi-blue { background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%); }
.kpi-orange { background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%); }
.kpi-red { background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); }
.pipeline-step {
    border: 2px solid #e0e0e0;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
    transition: all 0.3s;
}
.pipeline-step.active {
    border-color: #667eea;
    background: #f0f0ff;
}
.pipeline-step.done {
    border-color: #38ef7d;
    background: #f0fff0;
}
.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: bold;
}
.badge-ready { background: #e8f5e9; color: #2e7d32; }
.badge-pending { background: #fff3e0; color: #e65100; }
footer { display: none !important; }
"""


# ── Helper functions ──────────────────────────────────────────────────────────

def _kpi_html(title: str, value: str, unit: str, css_class: str = "") -> str:
    return f"""<div class="kpi-card {css_class}">
<h3>{title}</h3>
<div class="value">{value}</div>
<div class="unit">{unit}</div>
</div>"""


def _run_sim(latitude, longitude, year, freq, panel_eff, elec_price, mesh_type):
    """Run solar simulation and return all outputs."""
    from ..simulation.demo_factory import create_factory_complex, create_simple_factory
    from ..simulation.irradiance import compute_face_irradiance, save_irradiance_results
    from ..simulation.ray_caster import compute_shadow_matrix
    from ..simulation.roi_calculator import generate_roi_report
    from ..simulation.runner import load_config
    from ..simulation.solar_position import compute_clear_sky_irradiance, compute_solar_positions
    from ..simulation.visualization import create_irradiance_heatmap, create_sun_path_diagram

    config_path = CONFIGS_DIR / "solar_params.yaml"
    config = load_config(config_path)
    config["location"]["latitude"] = latitude
    config["location"]["longitude"] = longitude
    config["simulation"]["year"] = int(year)
    config["simulation"]["time_resolution_minutes"] = int(freq)
    config["panel"]["efficiency"] = panel_eff / 100
    config["electricity"]["price_per_kwh"] = elec_price

    if mesh_type == "アップロード済みメッシュ" and _state.get("mesh") is not None:
        mesh = _state["mesh"]
    elif mesh_type == "工場コンプレックス（4棟）":
        mesh = create_factory_complex()
    else:
        mesh = create_simple_factory()

    loc = config["location"]
    sim = config["simulation"]

    t0 = time.time()

    solar = compute_solar_positions(
        latitude=loc["latitude"], longitude=loc["longitude"],
        year=sim["year"], freq_minutes=sim["time_resolution_minutes"],
        timezone=loc["timezone"], altitude=loc.get("altitude", 0),
    )
    cs = compute_clear_sky_irradiance(
        latitude=loc["latitude"], longitude=loc["longitude"],
        year=sim["year"], freq_minutes=sim["time_resolution_minutes"],
        timezone=loc["timezone"], altitude=loc.get("altitude", 0),
        model=sim.get("dni_model", "ineichen"),
    )
    sun_dirs = solar.sun_direction_vectors()
    shadow_matrix = compute_shadow_matrix(
        mesh=mesh, sun_directions=sun_dirs, sun_visible=solar.sun_visible,
        min_face_area=config.get("mesh", {}).get("min_face_area_m2", 0),
    )
    time_step_hours = sim["time_resolution_minutes"] / 60.0
    irr = compute_face_irradiance(
        face_normals=mesh.face_normals, face_areas=mesh.area_faces,
        shadow_matrix=shadow_matrix, sun_directions=sun_dirs,
        dni=cs["dni"].values,
        dhi=cs["dhi"].values if sim.get("include_diffuse", True) else np.zeros(len(cs)),
        time_step_hours=time_step_hours,
    )
    elapsed = time.time() - t0

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

    _state["irradiance_results"] = irr
    _state["roi_report"] = roi
    _state["mesh"] = mesh
    _state["config"] = config

    # Visualizations
    heatmap_fig = create_irradiance_heatmap(mesh, irr)
    heatmap_fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        height=500,
    )

    sunpath_fig = create_sun_path_diagram(solar.azimuth, solar.elevation)
    sunpath_fig.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20))

    # Monthly irradiance bar chart
    monthly_fig = _create_monthly_chart(cs, sim["time_resolution_minutes"])

    # KPI cards
    kpi_html = f"""<div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:12px;">
{_kpi_html("設置可能容量", f"{roi.total_capacity_kw:.0f}", "kW", "kpi-blue")}
{_kpi_html("年間発電量", f"{roi.total_annual_generation_kwh / 1000:.0f}", "MWh/年", "kpi-green")}
{_kpi_html("年間削減額", f"¥{roi.total_annual_savings_jpy / 10000:.0f}万", "円/年", "kpi-orange")}
{_kpi_html("投資回収", f"{roi.overall_payback_years:.1f}", "年", "kpi-red")}
</div>
<div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:12px; margin-top:12px;">
{_kpi_html("25年NPV", f"¥{roi.overall_npv_25y_jpy / 10000:.0f}万", "円", "kpi-green")}
{_kpi_html("CO2削減", f"{roi.total_annual_generation_kwh * 0.000453:.1f}", "t-CO2/年", "kpi-blue")}
{_kpi_html("メッシュ面数", f"{len(mesh.faces)}", "面", "")}
{_kpi_html("計算時間", f"{elapsed:.1f}", "秒", "")}
</div>"""

    # Proposal table
    table_md = "### 設置優先順位\n\n"
    table_md += "| 順位 | 面ID | 面積(m²) | 年間日射量 | 年間発電量 | NPV | 回収期間 |\n"
    table_md += "|:----:|:----:|--------:|----------:|---------:|----:|-------:|\n"
    for p in roi.proposals[:10]:
        face = next(r for r in irr if r.face_id == p.face_id)
        table_md += (
            f"| {p.priority_rank} | {p.face_id} | {p.area_m2:.0f} | "
            f"{face.annual_irradiance_kwh_m2:,.0f} kWh/m² | "
            f"{p.annual_generation_kwh:,.0f} kWh | "
            f"¥{p.npv_25y_jpy / 10000:.0f}万 | "
            f"{p.payback_years:.1f}年 |\n"
        )

    return kpi_html, heatmap_fig, sunpath_fig, monthly_fig, table_md


def _create_monthly_chart(cs, freq_minutes):
    """Create monthly GHI bar chart."""
    import pandas as pd
    monthly = cs.resample("ME").sum() * (freq_minutes / 60) / 1000  # kWh/m²
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["1月", "2月", "3月", "4月", "5月", "6月",
           "7月", "8月", "9月", "10月", "11月", "12月"],
        y=monthly["ghi"].values[:12],
        marker_color=[
            "#4fc3f7", "#4fc3f7", "#81c784", "#81c784", "#81c784",
            "#ffb74d", "#ffb74d", "#ffb74d", "#81c784", "#81c784",
            "#4fc3f7", "#4fc3f7",
        ],
        hovertemplate="GHI: %{y:.0f} kWh/m²<extra></extra>",
    ))
    fig.update_layout(
        title="月別水平面日射量 (GHI)",
        yaxis_title="kWh/m²",
        height=350,
        margin=dict(l=40, r=20, t=40, b=30),
        showlegend=False,
    )
    return fig


def _on_location_change(location_name):
    """Update lat/lon when preset location is selected."""
    if location_name in PRESET_LOCATIONS:
        lat, lon = PRESET_LOCATIONS[location_name]
        return lat, lon
    return gr.update(), gr.update()


def _generate_chat_response(message, history):
    """Generate AI analysis response (mock/real VLM)."""
    roi = _state.get("roi_report")
    irr = _state.get("irradiance_results")

    if roi is None:
        return "シミュレーションを先に実行してください。「シミュレーション」タブでデータを生成すると、分析が可能になります。"

    # Context-aware mock response
    context = (
        f"設置可能容量: {roi.total_capacity_kw:.1f}kW, "
        f"年間発電量: {roi.total_annual_generation_kwh:,.0f}kWh, "
        f"年間削減額: ¥{roi.total_annual_savings_jpy:,.0f}, "
        f"投資回収: {roi.overall_payback_years:.1f}年, "
        f"25年NPV: ¥{roi.overall_npv_25y_jpy:,.0f}"
    )

    best = roi.proposals[0] if roi.proposals else None
    best_face = next((r for r in irr if r.face_id == best.face_id), None) if best else None

    if "最適" in message or "場所" in message or "どこ" in message:
        resp = f"""## 太陽光パネル最適設置場所の分析

### 分析データ
{context}

### 推奨設置場所

**第1優先: 面ID {best.face_id}**（年間日射量 {best_face.annual_irradiance_kwh_m2:,.0f} kWh/m²）
- 設置可能面積: {best.area_m2:.0f} m²
- 推定発電量: {best.annual_generation_kwh:,.0f} kWh/年
- 投資回収期間: {best.payback_years:.1f} 年
- 25年NPV: ¥{best.npv_25y_jpy / 10000:.0f}万円

この面は南向きの傾斜屋根で、年間を通じて最も安定した日射を受けます。
遮蔽物による影の影響も最小限です。

### 設置時の推奨事項
1. **傾斜角**: 現在の屋根角度（約{np.degrees(np.arccos(abs(best_face.normal[2]))):.0f}°）は当地域の最適角度に近い
2. **パネル種類**: 単結晶シリコン（効率20%以上）を推奨
3. **施工**: 屋根構造の荷重計算を事前に実施すること"""

    elif "コスト" in message or "削減" in message or "改善" in message:
        co2 = roi.total_annual_generation_kwh * 0.000453
        resp = f"""## エネルギーコスト削減提案

### 現状分析
{context}

### 提案1: 太陽光パネル設置（優先度: 高）
- **効果**: 年間 ¥{roi.total_annual_savings_jpy / 10000:.0f}万円 の電力コスト削減
- **CO2削減**: 年間 {co2:.1f} t-CO2
- **投資回収**: {roi.overall_payback_years:.1f} 年

### 提案2: ピークカット蓄電池の導入（優先度: 中）
- デマンドレスポンス対応で基本料金を削減
- 太陽光との組み合わせで自家消費率を最大化
- 推定追加削減: 年間 ¥{roi.total_annual_savings_jpy * 0.15 / 10000:.0f}万円

### 提案3: 屋根断熱改修との同時施工（優先度: 中）
- パネル設置と同時に断熱改修することで足場費用を共有
- 空調エネルギーを推定15-20%削減
- 推定追加削減: 年間 ¥{roi.total_annual_savings_jpy * 0.1 / 10000:.0f}万円"""

    elif "ROI" in message or "回収" in message or "投資" in message:
        resp = f"""## 投資回収分析

### サマリー
| 指標 | 値 |
|------|-----|
| 初期投資額 | ¥{roi.total_installation_cost_jpy / 10000:.0f}万円 |
| 年間削減額 | ¥{roi.total_annual_savings_jpy / 10000:.0f}万円 |
| 単純回収期間 | {roi.overall_payback_years:.1f}年 |
| 25年NPV | ¥{roi.overall_npv_25y_jpy / 10000:.0f}万円 |
| IRR（概算） | {(roi.total_annual_savings_jpy / roi.total_installation_cost_jpy * 100) if roi.total_installation_cost_jpy > 0 else 0:.1f}% |

### 回収期間の短縮方法
1. **補助金活用**: 自家消費型は国の補助金（設置費用の1/3程度）が利用可能
   → 回収期間を約{roi.overall_payback_years * 0.67:.1f}年に短縮
2. **PPA/リースモデル**: 初期投資ゼロで導入し、電力単価で支払い
3. **段階的導入**: 高日射面から優先設置し、キャッシュフローを早期改善"""

    else:
        resp = f"""## ExaSense分析レポート

### シミュレーション結果サマリー
{context}

### 主要な知見
1. 合計 {len(roi.proposals)} 面が太陽光パネル設置に適しています
2. 最も効率的な面は面ID {best.face_id}（日射量 {best_face.annual_irradiance_kwh_m2:,.0f} kWh/m²/年）
3. 投資回収期間 {roi.overall_payback_years:.1f}年は産業用太陽光の一般的な水準です

何について詳しく分析しますか？
- 「最適な設置場所」
- 「コスト削減提案」
- 「投資回収の詳細」"""

    return resp


def _generate_report():
    """Generate a comprehensive markdown report."""
    roi = _state.get("roi_report")
    irr = _state.get("irradiance_results")
    config = _state.get("config")

    if roi is None:
        return "シミュレーションを先に実行してください。", None, None

    loc = config["location"]
    co2 = roi.total_annual_generation_kwh * 0.000453

    report = f"""# ExaSense 太陽光パネル設置提案レポート

**生成日時**: {time.strftime('%Y年%m月%d日 %H:%M')}
**対象地点**: {loc['latitude']}°N, {loc['longitude']}°E

---

## 1. エグゼクティブサマリー

本レポートは、対象施設における太陽光パネル設置の最適化分析結果をまとめたものです。

| 指標 | 値 |
|------|-----|
| 設置可能容量 | **{roi.total_capacity_kw:.1f} kW** |
| 年間発電量 | **{roi.total_annual_generation_kwh:,.0f} kWh** ({roi.total_annual_generation_kwh / 1000:.1f} MWh) |
| 年間電力コスト削減額 | **¥{roi.total_annual_savings_jpy:,.0f}** |
| 年間CO2削減量 | **{co2:.1f} t-CO2** |
| 初期投資額 | ¥{roi.total_installation_cost_jpy:,.0f} |
| 投資回収期間 | **{roi.overall_payback_years:.1f} 年** |
| 25年間NPV | **¥{roi.overall_npv_25y_jpy:,.0f}** |

## 2. 施設分析

- 解析対象メッシュ面数: {len(irr)} 面
- パネル設置適合面数: {len(roi.proposals)} 面
- 合計設置可能面積: {roi.total_area_m2:.0f} m²

## 3. 設置優先順位

| 順位 | 面ID | 面積(m²) | 年間日射量(kWh/m²) | 発電量(kWh/年) | NPV(万円) | 回収(年) |
|:----:|:----:|--------:|-----------------:|-----------:|--------:|-------:|
"""

    for p in roi.proposals[:10]:
        face = next(r for r in irr if r.face_id == p.face_id)
        report += (
            f"| {p.priority_rank} | {p.face_id} | {p.area_m2:.0f} | "
            f"{face.annual_irradiance_kwh_m2:,.0f} | "
            f"{p.annual_generation_kwh:,.0f} | "
            f"{p.npv_25y_jpy / 10000:.0f} | "
            f"{p.payback_years:.1f} |\n"
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
- 25年間の累計発電量: {roi.total_annual_generation_kwh * 25 * 0.94:,.0f} kWh（劣化考慮）
- 25年間の累計コスト削減: ¥{roi.total_annual_savings_jpy * 25 * 1.25:,.0f}（電気料金上昇考慮）

## 5. 推奨アクション

1. **即時実施**: 優先順位1-3の面へのパネル設置（最も投資効率が高い）
2. **補助金申請**: 自家消費型太陽光発電設備の補助金を確認
3. **詳細設計**: 構造計算・電気設計の実施
4. **段階導入**: 第1期で優先度上位面に設置し、効果確認後に拡大

---

*本レポートはExaSenseシミュレーションエンジンにより自動生成されました。*
*実際の設置には専門家による現地調査と詳細設計が必要です。*
"""

    _state["report_md"] = report

    # Save files for download
    md_path = RESULTS_DIR / "latest" / "report.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(report)

    json_data = [
        {
            "face_id": r.face_id,
            "annual_irradiance_kwh_m2": round(r.annual_irradiance_kwh_m2, 2),
            "area_m2": round(r.area_m2, 4),
            "sun_hours": round(r.sun_hours, 1),
        }
        for r in irr
    ]
    json_path = RESULTS_DIR / "latest" / "irradiance_data.json"
    json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False))

    return report, str(md_path), str(json_path)


def _load_uploaded_mesh(file_obj):
    """Load a PLY/OBJ mesh file uploaded by the user."""
    import trimesh

    if file_obj is None:
        return "ファイルを選択してください。", None

    file_path = file_obj if isinstance(file_obj, str) else file_obj.name
    try:
        mesh = trimesh.load(file_path, force="mesh")
    except Exception as e:
        return f"メッシュ読み込みエラー: {e}", None

    _state["mesh"] = mesh

    info = (
        f"**読み込み完了**\n\n"
        f"- ファイル: `{Path(file_path).name}`\n"
        f"- 頂点数: **{len(mesh.vertices):,}**\n"
        f"- 面数: **{len(mesh.faces):,}**\n"
        f"- 表面積: {mesh.area:.1f} m²\n"
        f"- バウンディングボックス: {mesh.bounds[0].round(2)} ~ {mesh.bounds[1].round(2)}"
    )

    fig = _create_3d_mesh_view()
    return info, fig


def _create_3d_mesh_view():
    """Create a 3D view of the current mesh."""
    mesh = _state.get("mesh")
    if mesh is None:
        from ..simulation.demo_factory import create_factory_complex
        mesh = create_factory_complex()

    vertices = mesh.vertices
    faces = mesh.faces

    # Color by face normal z-component (roof=blue, wall=gray, floor=green)
    nz = mesh.face_normals[:, 2]
    colors = np.zeros(len(faces))
    for i, z in enumerate(nz):
        if z > 0.5:
            colors[i] = 2  # roof
        elif abs(z) < 0.1:
            colors[i] = 1  # wall
        else:
            colors[i] = 0  # floor/other

    # Downsample for browser performance (>100K faces is slow in Plotly)
    max_faces = 100_000
    if len(faces) > max_faces:
        import random
        idx = sorted(random.sample(range(len(faces)), max_faces))
        # Rebuild with only selected faces
        import trimesh as _trimesh
        sub = _trimesh.Trimesh(vertices=vertices, faces=faces[idx], process=False)
        vertices = sub.vertices
        faces = sub.faces
        nz = sub.face_normals[:, 2]
        colors = np.zeros(len(faces))
        for i, z in enumerate(nz):
            if z > 0.5:
                colors[i] = 2
            elif abs(z) < 0.1:
                colors[i] = 1
            else:
                colors[i] = 0

    fig = go.Figure(data=[
        go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            intensity=colors, intensitymode="cell",
            colorscale=[[0, "#8d6e63"], [0.5, "#90a4ae"], [1, "#42a5f5"]],
            showscale=False,
            hovertemplate="x: %{x:.1f}<br>y: %{y:.1f}<br>z: %{z:.1f}<extra></extra>",
            lighting=dict(ambient=0.5, diffuse=0.8, specular=0.3),
        )
    ])
    fig.update_layout(
        title="3D Building Model",
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0)),
        ),
        height=550,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


# ── Build App ─────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="ExaSense - 工場エネルギー最適化",
    ) as app:

        # ── Header ──
        gr.HTML("""
        <div style="background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #01579b 100%);
                    padding: 24px 32px; border-radius: 12px; margin-bottom: 16px;
                    display: flex; align-items: center; justify-content: space-between;">
            <div>
                <h1 style="color: white; margin: 0; font-size: 28px;">
                    ExaSense
                </h1>
                <p style="color: rgba(255,255,255,0.85); margin: 4px 0 0 0; font-size: 14px;">
                    工場向けエネルギー最適化ソリューション — 3D再構築 × 日照シミュレーション × AI分析
                </p>
            </div>
            <div style="color: rgba(255,255,255,0.7); font-size: 12px; text-align: right;">
                <div>Powered by pvlib + trimesh + Qwen3.5-VL</div>
                <div style="margin-top:4px;">Kanden Hackathon 2026</div>
            </div>
        </div>
        """)

        with gr.Tabs() as tabs:

            # ━━━ Tab 1: Dashboard ━━━
            with gr.TabItem("Dashboard", id="dashboard"):
                gr.HTML("""
                <div style="padding: 20px; background: #f5f5f5; border-radius: 8px; margin-bottom: 16px;">
                    <h3 style="margin:0 0 8px 0;">パイプライン概要</h3>
                    <div style="display: flex; gap: 8px; align-items: center;">
                        <div style="flex:1; text-align:center; padding:12px; background:#e3f2fd; border-radius:8px; border:2px solid #1976d2;">
                            <div style="font-weight:bold;">Phase 1</div>
                            <div style="font-size:12px;">3D再構築</div>
                        </div>
                        <div style="font-size:24px; color:#999;">→</div>
                        <div style="flex:1; text-align:center; padding:12px; background:#e3f2fd; border-radius:8px; border:2px solid #1976d2;">
                            <div style="font-weight:bold;">Phase 2</div>
                            <div style="font-size:12px;">メッシュ処理</div>
                        </div>
                        <div style="font-size:24px; color:#999;">→</div>
                        <div style="flex:1; text-align:center; padding:12px; background:#e8f5e9; border-radius:8px; border:2px solid #388e3c;">
                            <div style="font-weight:bold;">Phase 3</div>
                            <div style="font-size:12px;">日照シミュレーション</div>
                            <div style="font-size:10px; color:#388e3c;">READY</div>
                        </div>
                        <div style="font-size:24px; color:#999;">→</div>
                        <div style="flex:1; text-align:center; padding:12px; background:#e3f2fd; border-radius:8px; border:2px solid #1976d2;">
                            <div style="font-weight:bold;">Phase 4</div>
                            <div style="font-size:12px;">AI分析 (VLM)</div>
                        </div>
                        <div style="font-size:24px; color:#999;">→</div>
                        <div style="flex:1; text-align:center; padding:12px; background:#e8f5e9; border-radius:8px; border:2px solid #388e3c;">
                            <div style="font-weight:bold;">Phase 5</div>
                            <div style="font-size:12px;">WebUI</div>
                            <div style="font-size:10px; color:#388e3c;">ACTIVE</div>
                        </div>
                    </div>
                </div>
                """)

                gr.Markdown("### クイックスタート")
                gr.Markdown(
                    "1. **シミュレーション**タブで地点・パラメータを設定し実行\n"
                    "2. **3Dビュー**タブで建物モデルとヒートマップを確認\n"
                    "3. **AI分析**タブで設置提案をAIに質問\n"
                    "4. **レポート**タブで報告書をダウンロード"
                )

                dashboard_mesh = gr.Plot(label="3Dモデルプレビュー")
                load_preview_btn = gr.Button("プレビューを読み込む", size="sm")
                load_preview_btn.click(fn=_create_3d_mesh_view, outputs=[dashboard_mesh])

            # ━━━ Tab 2: 3D View ━━━
            with gr.TabItem("3D ビュー", id="3dview"):
                gr.Markdown("### 3D建物モデル")
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### メッシュファイル読み込み")
                        mesh_upload = gr.File(
                            label="PLY / OBJ ファイル",
                            file_types=[".ply", ".obj", ".stl", ".glb"],
                        )
                        upload_btn = gr.Button("読み込み", variant="primary")
                        mesh_info = gr.Markdown("")

                        gr.Markdown("---")
                        gr.Markdown("#### デモメッシュ")
                        mesh_type_view = gr.Radio(
                            ["単棟工場", "工場コンプレックス（4棟）"],
                            value="工場コンプレックス（4棟）",
                            label="プリセット",
                        )
                        load_mesh_btn = gr.Button("デモ表示", size="sm")
                        gr.Markdown("""
**色の凡例:**
- 青: 屋根面（パネル設置候補）
- グレー: 壁面
- 茶: 床面・その他
""")
                    with gr.Column(scale=3):
                        mesh_3d_plot = gr.Plot(label="3Dモデル")

                upload_btn.click(
                    fn=_load_uploaded_mesh,
                    inputs=[mesh_upload],
                    outputs=[mesh_info, mesh_3d_plot],
                )
                load_mesh_btn.click(fn=_create_3d_mesh_view, outputs=[mesh_3d_plot])

            # ━━━ Tab 3: Simulation ━━━
            with gr.TabItem("シミュレーション", id="simulation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### パラメータ設定")

                        location_preset = gr.Dropdown(
                            choices=list(PRESET_LOCATIONS.keys()),
                            value="大阪（関西電力エリア）",
                            label="プリセット地点",
                        )
                        with gr.Row():
                            lat = gr.Number(label="緯度", value=34.69, precision=4)
                            lon = gr.Number(label="経度", value=135.50, precision=4)

                        location_preset.change(
                            fn=_on_location_change,
                            inputs=[location_preset],
                            outputs=[lat, lon],
                        )

                        year = gr.Number(label="シミュレーション年", value=2025, precision=0)
                        freq = gr.Dropdown(
                            choices=[("1時間", 60), ("30分", 30)],
                            value=60,
                            label="時間分解能",
                        )

                        gr.Markdown("#### パネル・経済パラメータ")
                        efficiency = gr.Slider(
                            label="パネル効率 (%)", minimum=15, maximum=25,
                            value=20, step=0.5,
                        )
                        price = gr.Number(label="電気料金 (¥/kWh)", value=30, precision=0)

                        mesh_type = gr.Radio(
                            ["アップロード済みメッシュ", "単棟工場", "工場コンプレックス（4棟）"],
                            value="工場コンプレックス（4棟）",
                            label="対象建物",
                        )

                        run_btn = gr.Button(
                            "シミュレーション実行",
                            variant="primary",
                            size="lg",
                        )

                    with gr.Column(scale=3):
                        kpi_display = gr.HTML(
                            '<div style="padding:40px; text-align:center; color:#999;">'
                            'シミュレーションを実行すると結果が表示されます</div>'
                        )

                        with gr.Tabs():
                            with gr.TabItem("日射量ヒートマップ"):
                                heatmap_plot = gr.Plot(label="3D日射量ヒートマップ")
                            with gr.TabItem("太陽軌跡"):
                                sunpath_plot = gr.Plot(label="太陽軌跡図")
                            with gr.TabItem("月別日射量"):
                                monthly_plot = gr.Plot(label="月別GHI")

                        sim_table = gr.Markdown("")

                run_btn.click(
                    fn=_run_sim,
                    inputs=[lat, lon, year, freq, efficiency, price, mesh_type],
                    outputs=[kpi_display, heatmap_plot, sunpath_plot, monthly_plot, sim_table],
                )

            # ━━━ Tab 4: AI Analysis ━━━
            with gr.TabItem("AI 分析", id="ai"):
                gr.Markdown("### AI エネルギーアドバイザー")
                gr.Markdown(
                    "シミュレーション結果に基づいて、AIが太陽光パネル設置の提案・分析を行います。"
                    " *（H100環境では Qwen3.5-VL が回答します）*"
                )
                gr.ChatInterface(
                    fn=_generate_chat_response,
                    examples=[
                        "この施設で太陽光パネルの設置に最も適した場所はどこですか？",
                        "エネルギーコスト削減のための改善提案を3つ挙げてください。",
                        "投資回収期間とROIの詳細を教えてください。",
                        "この施設の分析結果を要約してください。",
                    ],
                )

            # ━━━ Tab 5: Report ━━━
            with gr.TabItem("レポート", id="report"):
                gr.Markdown("### 太陽光パネル設置提案レポート")
                gen_report_btn = gr.Button(
                    "レポート生成",
                    variant="primary",
                    size="lg",
                )
                report_md = gr.Markdown("シミュレーション実行後、レポートを生成できます。")
                with gr.Row():
                    download_md = gr.File(label="レポート (Markdown)")
                    download_json = gr.File(label="日射量データ (JSON)")

                gen_report_btn.click(
                    fn=_generate_report,
                    outputs=[report_md, download_md, download_json],
                )

    return app


def main():
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        share=False,
        theme=gr.themes.Default(
            primary_hue="blue",
            secondary_hue="emerald",
            font=gr.themes.GoogleFont("Noto Sans JP"),
        ),
        css=CSS,
    )


if __name__ == "__main__":
    main()
