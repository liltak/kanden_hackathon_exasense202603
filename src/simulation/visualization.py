"""3D heatmap visualization for irradiance results.

Generates interactive Plotly 3D visualizations of mesh faces
colored by annual irradiance.
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import trimesh

from .irradiance import FaceIrradiance


def create_irradiance_heatmap(
    mesh: trimesh.Trimesh,
    irradiance_results: list[FaceIrradiance],
    title: str = "Annual Solar Irradiance (kWh/m²)",
    colorscale: str = "YlOrRd",
) -> go.Figure:
    """Create interactive 3D heatmap of irradiance on mesh.

    Args:
        mesh: Triangle mesh.
        irradiance_results: Per-face irradiance data.
        title: Plot title.
        colorscale: Plotly colorscale name.

    Returns:
        Plotly Figure object.
    """
    vertices = mesh.vertices
    faces = mesh.faces

    values = np.zeros(len(faces))
    for r in irradiance_results:
        if r.face_id < len(values):
            values[r.face_id] = r.annual_irradiance_kwh_m2

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                intensity=values,
                intensitymode="cell",
                colorscale=colorscale,
                colorbar=dict(title="kWh/m²/year"),
                hovertemplate=(
                    "Irradiance: %{intensity:.0f} kWh/m²/year<br>"
                    "<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="East (m)",
            yaxis_title="North (m)",
            zaxis_title="Up (m)",
            aspectmode="data",
        ),
        width=1000,
        height=700,
    )

    return fig


def create_sun_path_diagram(
    azimuth: np.ndarray,
    elevation: np.ndarray,
    title: str = "Sun Path Diagram",
) -> go.Figure:
    """Create a sun path diagram (polar plot).

    Args:
        azimuth: Sun azimuth angles in degrees (0=North, clockwise).
        elevation: Sun elevation angles in degrees.
        title: Plot title.

    Returns:
        Plotly Figure object.
    """
    mask = elevation > 0
    az = azimuth[mask]
    el = elevation[mask]

    # Create hour-of-day coloring
    hours = np.arange(len(azimuth))[mask] % (365 * 24)
    month = hours / (30 * 24)

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=90 - el,  # zenith angle (0 = overhead, 90 = horizon)
            theta=az,
            mode="markers",
            marker=dict(
                size=2,
                color=month,
                colorscale="Rainbow",
                colorbar=dict(title="Month"),
            ),
            hovertemplate=(
                "Azimuth: %{theta:.1f}°<br>"
                "Elevation: %{customdata:.1f}°<br>"
                "<extra></extra>"
            ),
            customdata=el,
        )
    )

    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(range=[0, 90], tickvals=[0, 15, 30, 45, 60, 75, 90]),
            angularaxis=dict(
                direction="clockwise",
                rotation=90,  # 0° at top (North)
            ),
        ),
        width=700,
        height=700,
    )

    return fig


def save_heatmap_html(fig: go.Figure, output_path: Path) -> None:
    """Save Plotly figure as standalone HTML."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs=True)
