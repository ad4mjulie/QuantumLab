"""
Orbital renderer — PyVista-based 3-D visualisation of hydrogen orbitals.

Renders point clouds produced by the Monte-Carlo sampler with colour
mapping based on wavefunction phase or probability density.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    import pyvista as pv
except ImportError:
    pv = None  # graceful fallback if PyVista unavailable


def render_orbital(
    points: NDArray,
    values: NDArray,
    title: str = "Hydrogen Orbital",
    point_size: float = 1.5,
    opacity: float = 0.6,
    cmap: str = "coolwarm",
    show_nucleus: bool = True,
    background: str = "black",
    window_size: tuple[int, int] = (1200, 900),
    screenshot: str | None = None,
) -> None:
    """
    Render an orbital point cloud with PyVista.

    Parameters
    ----------
    points      : (N, 3) Cartesian coordinates
    values      : (N,) scalar values for colour mapping (density or phase)
    title       : window title
    point_size  : size of each point glyph
    opacity     : cloud opacity
    cmap        : matplotlib colour-map name
    show_nucleus: draw a sphere at the origin
    background  : background colour
    window_size : (width, height)
    screenshot  : if given, save a PNG to this path
    """
    if pv is None:
        raise ImportError("PyVista is required for 3-D rendering. pip install pyvista")

    cloud = pv.PolyData(points)
    cloud["scalars"] = values

    plotter = pv.Plotter(window_size=window_size)
    plotter.set_background(background)

    plotter.add_mesh(
        cloud,
        scalars="scalars",
        cmap=cmap,
        point_size=point_size,
        opacity=opacity,
        render_points_as_spheres=True,
        show_scalar_bar=True,
        scalar_bar_args={"title": "|ψ|²", "color": "white"},
    )

    if show_nucleus:
        nucleus = pv.Sphere(radius=0.15, center=(0, 0, 0))
        plotter.add_mesh(nucleus, color="yellow", opacity=1.0)

    plotter.add_text(title, font_size=14, color="white")
    plotter.add_axes(color="white")

    if screenshot:
        plotter.show(screenshot=screenshot, auto_close=True)
    else:
        plotter.show()


def render_orbital_isosurface(
    x: NDArray,
    y: NDArray,
    z: NDArray,
    density: NDArray,
    iso_value: float | None = None,
    title: str = "Orbital Isosurface",
    cmap: str = "plasma",
    opacity: float = 0.5,
    show_nucleus: bool = True,
    background: str = "black",
    window_size: tuple[int, int] = (1200, 900),
    screenshot: str | None = None,
) -> None:
    """
    Render an isosurface of |ψ|² on a structured grid.

    Parameters
    ----------
    x, y, z  : 1-D coordinate arrays
    density  : 3-D array of |ψ|² values (shape = len(x) × len(y) × len(z))
    iso_value: isosurface threshold (auto-detected if None)
    """
    if pv is None:
        raise ImportError("PyVista is required for 3-D rendering.")

    grid = pv.ImageData(
        dimensions=(len(x), len(y), len(z)),
        spacing=(x[1] - x[0], y[1] - y[0], z[1] - z[0]),
        origin=(x[0], y[0], z[0]),
    )
    grid["density"] = density.flatten(order="F")

    if iso_value is None:
        iso_value = float(np.max(density)) * 0.1

    contour = grid.contour([iso_value], scalars="density")

    plotter = pv.Plotter(window_size=window_size)
    plotter.set_background(background)

    if contour.n_points > 0:
        plotter.add_mesh(
            contour,
            cmap=cmap,
            opacity=opacity,
            show_scalar_bar=True,
            scalar_bar_args={"title": "|ψ|²", "color": "white"},
        )

    if show_nucleus:
        nucleus = pv.Sphere(radius=0.15, center=(0, 0, 0))
        plotter.add_mesh(nucleus, color="yellow", opacity=1.0)

    plotter.add_text(title, font_size=14, color="white")
    plotter.add_axes(color="white")

    if screenshot:
        plotter.show(screenshot=screenshot, auto_close=True)
    else:
        plotter.show()
