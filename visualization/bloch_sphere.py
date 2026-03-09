"""
Bloch sphere visualisation using PyVista.

Draws the unit Bloch sphere with axes labels (|0⟩, |1⟩, |+⟩, |−⟩, |i⟩, |−i⟩)
and renders a qubit state vector.
"""

from __future__ import annotations

import numpy as np

try:
    import pyvista as pv
except ImportError:
    pv = None


def bloch_vector_from_angles(theta: float, phi: float) -> tuple[float, float, float]:
    """Convert Bloch sphere angles to Cartesian (x, y, z)."""
    return (
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    )


def render_bloch_sphere(
    theta: float = 0.0,
    phi: float = 0.0,
    title: str = "Bloch Sphere",
    background: str = "black",
    window_size: tuple[int, int] = (900, 900),
    screenshot: str | None = None,
) -> None:
    """
    Render an interactive Bloch sphere with the qubit state
    |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩.

    Parameters
    ----------
    theta : float – polar angle on the Bloch sphere [0, π]
    phi   : float – azimuthal angle [0, 2π)
    """
    if pv is None:
        raise ImportError("PyVista is required. pip install pyvista")

    plotter = pv.Plotter(window_size=window_size)
    plotter.set_background(background)

    # Semi-transparent sphere
    sphere = pv.Sphere(radius=1.0, theta_resolution=60, phi_resolution=60)
    plotter.add_mesh(sphere, color="steelblue", opacity=0.15, smooth_shading=True)

    # Wireframe rings (equator + meridians)
    for angle in [0, np.pi / 2]:
        ring = pv.Circle(radius=1.0, resolution=120)
        if angle == np.pi / 2:
            ring = ring.rotate_x(90, inplace=False)
        plotter.add_mesh(ring, color="gray", line_width=1, opacity=0.4)

    ring_yz = pv.Circle(radius=1.0, resolution=120).rotate_y(90, inplace=False)
    plotter.add_mesh(ring_yz, color="gray", line_width=1, opacity=0.4)

    # Axes lines
    axis_len = 1.3
    for direction, color, label_pos, label in [
        ([axis_len, 0, 0], "red",    (axis_len + 0.15, 0, 0), "|+⟩"),
        ([-axis_len, 0, 0], "red",   (-axis_len - 0.25, 0, 0), "|−⟩"),
        ([0, axis_len, 0], "green",  (0, axis_len + 0.15, 0), "|i⟩"),
        ([0, -axis_len, 0], "green", (0, -axis_len - 0.25, 0), "|−i⟩"),
        ([0, 0, axis_len], "blue",   (0, 0, axis_len + 0.15), "|0⟩"),
        ([0, 0, -axis_len], "blue",  (0, 0, -axis_len - 0.25), "|1⟩"),
    ]:
        line = pv.Line((0, 0, 0), direction)
        plotter.add_mesh(line, color=color, line_width=2)
        plotter.add_point_labels(
            [label_pos], [label], font_size=16, text_color="white",
            point_size=0, shape=None, always_visible=True,
        )

    # State vector arrow
    bx, by, bz = bloch_vector_from_angles(theta, phi)
    arrow = pv.Arrow(start=(0, 0, 0), direction=(bx, by, bz), scale=1.0)
    plotter.add_mesh(arrow, color="cyan", opacity=1.0)

    # State point
    state_point = pv.Sphere(radius=0.05, center=(bx, by, bz))
    plotter.add_mesh(state_point, color="cyan")

    plotter.add_text(title, font_size=14, color="white")

    if screenshot:
        plotter.show(screenshot=screenshot, auto_close=False)
    else:
        plotter.show(auto_close=False)
    
    plotter.close()
