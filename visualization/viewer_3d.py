"""
Generic 3-D viewer utilities.

Provides convenience wrappers for:
  • Quick point-cloud viewing
  • Plotly fallback for browser-based interactive plots
  • Screenshot / export helpers
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    import pyvista as pv
except ImportError:
    pv = None

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


# ---------------------------------------------------------------------------
# PyVista quick-view
# ---------------------------------------------------------------------------

def quick_plot(
    points: NDArray,
    scalars: NDArray | None = None,
    point_size: float = 2.0,
    cmap: str = "viridis",
    title: str = "3D Viewer",
    background: str = "black",
    window_size: tuple[int, int] = (1200, 900),
    screenshot: str | None = None,
) -> None:
    """Fast point-cloud visualisation with PyVista."""
    if pv is None:
        raise ImportError("PyVista required. pip install pyvista")

    cloud = pv.PolyData(points)
    if scalars is not None:
        cloud["values"] = scalars

    plotter = pv.Plotter(window_size=window_size)
    plotter.set_background(background)
    plotter.add_mesh(
        cloud,
        scalars="values" if scalars is not None else None,
        cmap=cmap,
        point_size=point_size,
        render_points_as_spheres=True,
        opacity=0.6,
        show_scalar_bar=scalars is not None,
        scalar_bar_args={"color": "white"} if scalars is not None else {},
    )
    plotter.add_text(title, font_size=12, color="white")
    plotter.add_axes(color="white")

    # Label the potential well center
    center_pt = np.array([[0.0, 0.0, 0.2]], dtype=np.float32)
    plotter.add_point_labels(
        center_pt,
        ["Potential Center"],
        point_color="yellow",
        point_size=5,
        font_size=16,
        text_color="yellow",
        shape="rounded_rect",
        shape_color="black",
        shape_opacity=0.4,
        always_visible=True,
    )

    # Estimate extents for the probability cloud label
    try:
        max_extent = float(np.max(np.abs(points)))
    except ValueError:
        max_extent = 10.0
    
    # Place wave-function label
    cloud_pt = np.array([[max_extent * 0.6, max_extent * 0.6, max_extent * 0.6]], dtype=np.float32)
    plotter.add_point_labels(
        cloud_pt,
        ["Wavefunction Cloud"],
        point_color="white",
        point_size=5,
        font_size=14,
        text_color="white",
        shape="rounded_rect",
        shape_color="black",
        shape_opacity=0.4,
        always_visible=True,
    )

    if screenshot:
        plotter.show(screenshot=screenshot, auto_close=False)
    else:
        plotter.show(auto_close=False)
    
    plotter.close()


# ---------------------------------------------------------------------------
# Plotly fallback (browser-based)
# ---------------------------------------------------------------------------

def plotly_scatter3d(
    points: NDArray,
    scalars: NDArray | None = None,
    title: str = "3D Viewer",
    colorscale: str = "Viridis",
    marker_size: float = 1.0,
    opacity: float = 0.5,
    max_points: int = 50_000,
) -> "go.Figure":
    """
    Create an interactive Plotly 3-D scatter plot.

    The figure is returned (call ``.show()`` to display).
    If there are more than ``max_points`` points, a random subsample is used.
    """
    if go is None:
        raise ImportError("Plotly required. pip install plotly")

    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
        if scalars is not None:
            scalars = scalars[idx]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=scalars if scalars is not None else "cyan",
                    colorscale=colorscale if scalars is not None else None,
                    opacity=opacity,
                    colorbar=dict(title="Value") if scalars is not None else None,
                ),
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (a₀)",
            yaxis_title="Y (a₀)",
            zaxis_title="Z (a₀)",
            bgcolor="black",
        ),
        paper_bgcolor="black",
        font_color="white",
    )
    return fig


# ---------------------------------------------------------------------------
# Export helper
# ---------------------------------------------------------------------------

def export_points(points: NDArray, scalars: NDArray | None, path: str) -> None:
    """
    Export a point cloud to a CSV file.

    Columns: x, y, z [, value]
    """
    if scalars is not None:
        data = np.column_stack([points, scalars])
        header = "x,y,z,value"
    else:
        data = points
        header = "x,y,z"
    np.savetxt(path, data, delimiter=",", header=header, comments="")
