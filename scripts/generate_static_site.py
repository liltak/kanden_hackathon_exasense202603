"""Generate a self-contained static HTML dashboard from simulation results."""

import json
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
CONFIGS_DIR = PROJECT_ROOT / "configs"

def main():
    from src.simulation.demo_factory import create_factory_complex
    from src.simulation.irradiance import compute_face_irradiance
    from src.simulation.ray_caster import compute_shadow_matrix
    from src.simulation.roi_calculator import generate_roi_report
    from src.simulation.runner import load_config
    from src.simulation.solar_position import compute_clear_sky_irradiance, compute_solar_positions
    from src.simulation.visualization import create_irradiance_heatmap, create_sun_path_diagram

    config = load_config(CONFIGS_DIR / "solar_params.yaml")
    loc = config["location"]
    sim = config["simulation"]
    panel_cfg = config["panel"]
    elec_cfg = config["electricity"]

    mesh = create_factory_complex()
    print(f"Mesh: {len(mesh.faces)} faces")

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

    roi = generate_roi_report(
        irr,
        panel_efficiency=panel_cfg.get("efficiency", 0.20),
        cost_per_kw_jpy=panel_cfg.get("cost_per_kw", 250_000),
        electricity_price_jpy=elec_cfg.get("price_per_kwh", 30),
        annual_price_increase=elec_cfg.get("annual_price_increase", 0.02),
        degradation_rate=panel_cfg.get("degradation_rate", 0.005),
        lifespan_years=panel_cfg.get("lifespan_years", 25),
    )

    # --- Generate plotly figures ---
    heatmap_fig = create_irradiance_heatmap(mesh, irr)
    heatmap_fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=500)

    sunpath_fig = create_sun_path_diagram(solar.azimuth, solar.elevation)
    sunpath_fig.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20))

    # Monthly chart
    import pandas as pd
    monthly = cs.resample("ME").sum() * (sim["time_resolution_minutes"] / 60) / 1000
    monthly_fig = go.Figure()
    monthly_fig.add_trace(go.Bar(
        x=["1月","2月","3月","4月","5月","6月","7月","8月","9月","10月","11月","12月"],
        y=monthly["ghi"].values[:12],
        marker_color=["#4fc3f7","#4fc3f7","#81c784","#81c784","#81c784",
                       "#ffb74d","#ffb74d","#ffb74d","#81c784","#81c784",
                       "#4fc3f7","#4fc3f7"],
        hovertemplate="GHI: %{y:.0f} kWh/m²<extra></extra>",
    ))
    monthly_fig.update_layout(
        title="月別水平面日射量 (GHI)", yaxis_title="kWh/m²",
        height=350, margin=dict(l=40, r=20, t=40, b=30), showlegend=False,
    )

    # 3D mesh view
    vertices = mesh.vertices
    faces = mesh.faces
    nz = mesh.face_normals[:, 2]
    colors = np.zeros(len(faces))
    for i, z in enumerate(nz):
        if z > 0.5:
            colors[i] = 2
        elif abs(z) < 0.1:
            colors[i] = 1
    mesh_fig = go.Figure(data=[
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
    mesh_fig.update_layout(
        title="3D Building Model (工場コンプレックス 4棟)",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                   aspectmode="data", camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0))),
        height=550, margin=dict(l=0, r=0, t=40, b=0),
    )

    # Convert to HTML fragments
    heatmap_html = heatmap_fig.to_html(full_html=False, include_plotlyjs=False)
    sunpath_html = sunpath_fig.to_html(full_html=False, include_plotlyjs=False)
    monthly_html = monthly_fig.to_html(full_html=False, include_plotlyjs=False)
    mesh_html = mesh_fig.to_html(full_html=False, include_plotlyjs=False)

    # Priority table
    table_rows = ""
    for p in roi.proposals[:10]:
        face = next(r for r in irr if r.face_id == p.face_id)
        table_rows += (
            f"<tr><td>{p.priority_rank}</td><td>{p.face_id}</td>"
            f"<td>{p.area_m2:.0f}</td><td>{face.annual_irradiance_kwh_m2:,.0f}</td>"
            f"<td>{p.annual_generation_kwh:,.0f}</td>"
            f"<td>¥{p.npv_25y_jpy / 10000:.0f}万</td>"
            f"<td>{p.payback_years:.1f}年</td></tr>"
        )

    co2 = roi.total_annual_generation_kwh * 0.000453

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ExaSense - 工場エネルギー最適化ソリューション</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: 'Helvetica Neue', 'Noto Sans JP', sans-serif; background:#f5f7fa; color:#333; }}
.header {{
    background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #01579b 100%);
    padding: 28px 40px; color: white;
}}
.header h1 {{ font-size: 32px; margin-bottom: 4px; }}
.header p {{ opacity: 0.85; font-size: 14px; }}
.header .meta {{ opacity: 0.7; font-size: 12px; margin-top: 8px; }}
.container {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
.kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 24px 0; }}
.kpi {{ border-radius: 12px; padding: 20px; color: white; text-align: center; }}
.kpi h3 {{ font-size: 13px; opacity: 0.9; margin-bottom: 8px; }}
.kpi .value {{ font-size: 36px; font-weight: bold; }}
.kpi .unit {{ font-size: 12px; opacity: 0.8; margin-top: 4px; }}
.kpi-blue {{ background: linear-gradient(135deg, #2193b0, #6dd5ed); }}
.kpi-green {{ background: linear-gradient(135deg, #11998e, #38ef7d); }}
.kpi-orange {{ background: linear-gradient(135deg, #f7971e, #ffd200); }}
.kpi-red {{ background: linear-gradient(135deg, #eb3349, #f45c43); }}
.kpi-purple {{ background: linear-gradient(135deg, #667eea, #764ba2); }}
.kpi-teal {{ background: linear-gradient(135deg, #00b4db, #0083b0); }}
section {{ background: white; border-radius: 12px; padding: 24px; margin-bottom: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
section h2 {{ font-size: 20px; margin-bottom: 16px; color: #1a237e; border-bottom: 2px solid #e3f2fd; padding-bottom: 8px; }}
.tabs {{ display: flex; gap: 8px; margin-bottom: 16px; }}
.tab {{ padding: 8px 20px; border-radius: 6px; cursor: pointer; border: 1px solid #ddd; background: #fafafa; font-size: 14px; }}
.tab.active {{ background: #1a237e; color: white; border-color: #1a237e; }}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}
table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
th {{ background: #f0f4ff; padding: 10px 12px; text-align: left; font-weight: 600; border-bottom: 2px solid #ddd; }}
td {{ padding: 10px 12px; border-bottom: 1px solid #eee; }}
tr:hover td {{ background: #f8f9ff; }}
td:nth-child(n+3) {{ text-align: right; }}
th:nth-child(n+3) {{ text-align: right; }}
.pipeline {{ display: flex; gap: 8px; align-items: center; margin: 16px 0; flex-wrap: wrap; }}
.pipe-step {{ flex: 1; min-width: 140px; text-align: center; padding: 14px 8px; border-radius: 8px; border: 2px solid #1976d2; background: #e3f2fd; }}
.pipe-step.ready {{ border-color: #388e3c; background: #e8f5e9; }}
.pipe-step .label {{ font-weight: bold; font-size: 14px; }}
.pipe-step .desc {{ font-size: 11px; color: #555; margin-top: 2px; }}
.pipe-arrow {{ font-size: 24px; color: #999; }}
.grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
.report-section {{ background: #f9fafb; padding: 20px; border-radius: 8px; border-left: 4px solid #1a237e; }}
@media (max-width: 900px) {{
    .grid-2 {{ grid-template-columns: 1fr; }}
    .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
}}
</style>
</head>
<body>

<div class="header">
    <h1>ExaSense</h1>
    <p>工場向けエネルギー最適化ソリューション — 3D再構築 × 日照シミュレーション × AI分析</p>
    <div class="meta">Powered by VGGT + pvlib + trimesh + Open3D | Kanden Hackathon 2026</div>
</div>

<div class="container">

<!-- Pipeline -->
<section>
<h2>パイプライン概要</h2>
<div class="pipeline">
    <div class="pipe-step ready"><div class="label">Phase 1-2</div><div class="desc">3D再構築 + メッシュ処理</div></div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step ready"><div class="label">Phase 3</div><div class="desc">日照シミュレーション</div></div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step"><div class="label">Phase 4</div><div class="desc">VLM AI分析</div></div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step ready"><div class="label">Phase 5</div><div class="desc">WebUI ダッシュボード</div></div>
</div>
</section>

<!-- KPIs -->
<div class="kpi-grid">
    <div class="kpi kpi-blue"><h3>設置可能容量</h3><div class="value">{roi.total_capacity_kw:.0f}</div><div class="unit">kW</div></div>
    <div class="kpi kpi-green"><h3>年間発電量</h3><div class="value">{roi.total_annual_generation_kwh / 1000:.0f}</div><div class="unit">MWh/年</div></div>
    <div class="kpi kpi-orange"><h3>年間コスト削減</h3><div class="value">¥{roi.total_annual_savings_jpy / 10000:.0f}万</div><div class="unit">円/年</div></div>
    <div class="kpi kpi-red"><h3>投資回収期間</h3><div class="value">{roi.overall_payback_years:.1f}</div><div class="unit">年</div></div>
    <div class="kpi kpi-purple"><h3>25年NPV</h3><div class="value">¥{roi.overall_npv_25y_jpy / 10000:.0f}万</div><div class="unit">円</div></div>
    <div class="kpi kpi-teal"><h3>CO2削減</h3><div class="value">{co2:.1f}</div><div class="unit">t-CO2/年</div></div>
</div>

<!-- 3D Visualization -->
<section>
<h2>3D 可視化</h2>
<div class="tabs">
    <div class="tab active" onclick="switchTab('heatmap')">日射量ヒートマップ</div>
    <div class="tab" onclick="switchTab('mesh')">3Dモデル</div>
    <div class="tab" onclick="switchTab('sunpath')">太陽軌跡</div>
    <div class="tab" onclick="switchTab('monthly')">月別日射量</div>
</div>
<div id="tab-heatmap" class="tab-content active">{heatmap_html}</div>
<div id="tab-mesh" class="tab-content">{mesh_html}</div>
<div id="tab-sunpath" class="tab-content">{sunpath_html}</div>
<div id="tab-monthly" class="tab-content">{monthly_html}</div>
</section>

<!-- Priority Table -->
<section>
<h2>設置優先順位 TOP 10</h2>
<table>
<thead>
<tr><th>順位</th><th>面ID</th><th>面積 (m²)</th><th>年間日射量 (kWh/m²)</th><th>年間発電量 (kWh)</th><th>25年NPV</th><th>回収期間</th></tr>
</thead>
<tbody>{table_rows}</tbody>
</table>
</section>

<!-- Report -->
<section>
<h2>分析レポート</h2>
<div class="grid-2">
<div class="report-section">
<h3 style="margin-bottom:12px;">エグゼクティブサマリー</h3>
<p>対象地点: <strong>{loc['latitude']}°N, {loc['longitude']}°E</strong> (大阪・関西電力エリア)</p>
<p>解析対象: 工場コンプレックス (4棟) — <strong>{len(mesh.faces)} 面</strong></p>
<p style="margin-top:8px;">パネル設置適合面: <strong>{len(roi.proposals)} 面</strong> / 合計面積: <strong>{roi.total_area_m2:.0f} m²</strong></p>
</div>
<div class="report-section">
<h3 style="margin-bottom:12px;">経済性分析</h3>
<p>初期投資額: <strong>¥{roi.total_installation_cost_jpy / 10000:.0f}万円</strong></p>
<p>年間削減額: <strong>¥{roi.total_annual_savings_jpy / 10000:.0f}万円</strong></p>
<p>25年間累計削減: <strong>¥{roi.total_annual_savings_jpy * 25 * 1.25 / 100000000:.1f}億円</strong> (電気料金上昇考慮)</p>
<p style="margin-top:8px;">パネル効率: {panel_cfg['efficiency']*100:.0f}% / 電気料金: ¥{elec_cfg['price_per_kwh']}/kWh</p>
</div>
</div>

<div style="margin-top:20px; padding:16px; background:#fff3e0; border-radius:8px;">
<h3 style="margin-bottom:8px;">推奨アクション</h3>
<ol style="padding-left:20px; line-height:1.8;">
<li><strong>即時実施</strong>: 優先順位1-3の面へのパネル設置（最も投資効率が高い）</li>
<li><strong>補助金申請</strong>: 自家消費型太陽光発電設備の補助金を確認</li>
<li><strong>詳細設計</strong>: 構造計算・電気設計の実施</li>
<li><strong>段階導入</strong>: 第1期で優先度上位面に設置し、効果確認後に拡大</li>
</ol>
</div>
</section>

<!-- Tech Stack -->
<section>
<h2>技術スタック</h2>
<div class="grid-2">
<div>
<h3 style="margin-bottom:8px;">Phase 1-2: 3D再構築 + メッシュ処理</h3>
<ul style="padding-left:20px; line-height:1.8;">
<li><strong>VGGT-1B</strong> (Meta) — 画像→3D点群変換</li>
<li><strong>Open3D</strong> — Poisson再構成 + メッシュ品質改善</li>
<li>改善パイプライン: 1.7M点 → 20K面 (28x圧縮)</li>
</ul>
</div>
<div>
<h3 style="margin-bottom:8px;">Phase 3: 日照シミュレーション</h3>
<ul style="padding-left:20px; line-height:1.8;">
<li><strong>pvlib</strong> — 太陽位置・日射量計算</li>
<li><strong>trimesh</strong> — レイキャスティング影計算</li>
<li><strong>Plotly</strong> — 3Dヒートマップ可視化</li>
</ul>
</div>
</div>
<div class="grid-2" style="margin-top:16px;">
<div>
<h3 style="margin-bottom:8px;">Phase 4: VLM AI分析</h3>
<ul style="padding-left:20px; line-height:1.8;">
<li><strong>Qwen3.5-VL</strong> — マルチモーダル画像理解</li>
<li><strong>Unsloth</strong> — LoRA ファインチューニング</li>
</ul>
</div>
<div>
<h3 style="margin-bottom:8px;">Phase 5: WebUI</h3>
<ul style="padding-left:20px; line-height:1.8;">
<li><strong>Gradio</strong> — 5タブ対話型ダッシュボード</li>
<li><strong>FastAPI</strong> — REST API バックエンド</li>
</ul>
</div>
</div>
</section>

<footer style="text-align:center; padding:24px; color:#999; font-size:12px;">
ExaSense — Factory Energy Optimization Solution | Generated {time.strftime('%Y-%m-%d %H:%M')} | Kanden Hackathon 2026
</footer>

</div>

<script>
function switchTab(name) {{
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
    document.getElementById('tab-' + name).classList.add('active');
    event.target.classList.add('active');
    // Trigger plotly relayout for proper rendering
    var plotDiv = document.getElementById('tab-' + name).querySelector('.plotly-graph-div');
    if (plotDiv) Plotly.Plots.resize(plotDiv);
}}
</script>
</body>
</html>"""

    DOCS_DIR.mkdir(exist_ok=True)
    out = DOCS_DIR / "index.html"
    out.write_text(html)
    print(f"Static dashboard: {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
