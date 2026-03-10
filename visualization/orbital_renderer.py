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

from physics.hydrogen_solver import HydrogenSolver
from physics.orbitals import get_orbital
from quantum.hydrogen_register import measure_electron


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
        nucleus = pv.Sphere(radius=0.15, center=(0.0, 0.0, 0.0))
        plotter.add_mesh(nucleus, color="yellow", opacity=1.0)
        
        # Place label slightly above the nucleus sphere to prevent clipping
        nuc_pt = np.array([[0.0, 0.0, 0.2]], dtype=np.float32)
        plotter.add_point_labels(
            nuc_pt,
            ["Nucleus"],
            point_color="yellow",
            point_size=5,
            font_size=18,
            text_color="yellow",
            shape="rounded_rect",
            shape_color="black",
            shape_opacity=0.4,
            always_visible=True,
        )

    # Estimate extents for standard label placement
    try:
        max_extent = float(np.max(np.abs(points)))
    except ValueError:
        max_extent = 10.0
    
    # Place cloud label at a clear visual location
    cloud_pt = np.array([[max_extent * 0.6, max_extent * 0.6, max_extent * 0.6]], dtype=np.float32)
    plotter.add_point_labels(
        cloud_pt,
        ["Electron Cloud / Orbital"],
        point_color="white",
        point_size=5,
        font_size=16,
        text_color="white",
        shape="rounded_rect",
        shape_color="black",
        shape_opacity=0.4,
        always_visible=True,
    )

    plotter.add_text(title, font_size=14, color="white", name="plot_title")
    plotter.add_axes(color="white")

    # ------------------------------------------------------------------
    # "Measure Electron" Keyboard Shortcut (H)
    # ------------------------------------------------------------------
    
    def on_measure():
        """Callback to perform quantum measurement and collapse state."""
        # Visual feedback for start
        plotter.add_text(
            "MEASURING...", position="lower_right", font_size=12, 
            color="yellow", name="status_label"
        )
        plotter.render() # Force update
        
        collapsed_name = measure_electron()
        print(f"\n  [Quantum Measurement] Collapsed to: {collapsed_name}")
        
        # Optimize performance: use half the points for the reactive update if n > 50k
        target_pts = len(points)
        if target_pts > 40_000:
            target_pts = 40_000
            
        # Re-solve for the collapsed orbital
        solver = HydrogenSolver()
        n, l, m = get_orbital(collapsed_name)
        new_points, new_density = solver.sample_points_mc(n, l, m, n_points=target_pts)
        
        # Update mesh data
        cloud.points = new_points
        cloud["scalars"] = new_density
        
        # Update UI labels
        plotter.add_text(
            f"{collapsed_name} (Collapsed State)", 
            font_size=14, color="cyan", name="plot_title"
        )
        plotter.add_text(
            f"MEASURED: {collapsed_name} ({target_pts:,} pts)",
            position="lower_right",
            font_size=12,
            color="lime",
            name="status_label"
        )

    plotter.add_key_event("h", on_measure)
    plotter.add_text(
        "Press 'H' to Measure Electron", 
        position="upper_right", 
        font_size=10, 
        color="#8b949e"
    )

    if screenshot:
        plotter.show(screenshot=screenshot, auto_close=False)
    else:
        plotter.show(auto_close=False)
    
    plotter.close()


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
        nucleus = pv.Sphere(radius=0.15, center=(0.0, 0.0, 0.0))
        plotter.add_mesh(nucleus, color="yellow", opacity=1.0)
        
        nuc_pt = np.array([[0.0, 0.0, 0.2]], dtype=np.float32)
        plotter.add_point_labels(
            nuc_pt,
            ["Nucleus"],
            point_color="yellow",
            point_size=5,
            font_size=18,
            text_color="yellow",
            shape="rounded_rect",
            shape_color="black",
            shape_opacity=0.4,
            always_visible=True,
        )

    # Add label for the Isosurface/Cloud
    extent_x = float(np.max(np.abs(x)))
    cloud_pt = np.array([[extent_x * 0.6, extent_x * 0.6, extent_x * 0.6]], dtype=np.float32)
    plotter.add_point_labels(
        cloud_pt,
        ["Probability Density Isosurface"],
        point_color="white",
        point_size=5,
        font_size=16,
        text_color="white",
        shape="rounded_rect",
        shape_color="black",
        shape_opacity=0.4,
        always_visible=True,
    )

    plotter.add_text(title, font_size=14, color="white", name="plot_title")
    plotter.add_axes(color="white")

    # ------------------------------------------------------------------
    # "Measure Electron" Keyboard Shortcut (H)
    # ------------------------------------------------------------------
    
    def on_measure_iso():
        """Callback for isosurface measurement."""
        collapsed_name = measure_electron()
        print(f"\n  [Quantum Measurement] Collapsed to: {collapsed_name}")
        
        solver = HydrogenSolver()
        n, l, m = get_orbital(collapsed_name)
        
        # Isosurface requires grid data, not points
        res = solver.compute_orbital(n, l, m, grid_size=len(x))
        new_density = res["density"]
        
        # Update the contour by regenerating it from new grid data
        # Note: In PyVista we often replace the actor or clear and re-add
        grid["density"] = new_density.flatten(order="F")
        new_contour = grid.contour([iso_value], scalars="density")
        
        # Update existing contour mesh
        contour.copy_from(new_contour)
        
        plotter.add_text(
            f"{collapsed_name} (Isosurface)", 
            font_size=14, color="cyan", name="plot_title"
        )

    plotter.add_key_event("h", on_measure_iso)

    if screenshot:
        plotter.show(screenshot=screenshot, auto_close=False)
    else:
        plotter.show(auto_close=False)
    
    plotter.close()
