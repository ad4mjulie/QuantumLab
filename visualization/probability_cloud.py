"""
Probability cloud generator.

High-level functions that glue the physics solver to the visualisation
layer, producing colour-mapped point clouds from Monte-Carlo samples.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from physics.hydrogen_solver import HydrogenSolver


def generate_cloud(
    n: int,
    l: int,
    m: int,
    n_points: int = 100_000,
    r_max: float | None = None,
) -> dict[str, NDArray]:
    """
    Generate a point cloud for the (n, l, m) hydrogen orbital.

    Returns
    -------
    dict with keys:
        points  : (n_points, 3) Cartesian coordinates
        density : (n_points,)   |ψ|² at each point
        phase   : (n_points,)   arg(ψ) at each point (radians)
    """
    solver = HydrogenSolver()
    points, density = solver.sample_points_mc(n, l, m, n_points, r_max=r_max)
    phase = solver.compute_phase(n, l, m, points)
    return {"points": points, "density": density, "phase": phase}


def density_color_values(density: NDArray) -> NDArray:
    """Normalise density to [0, 1] for colourmap mapping."""
    d_min, d_max = density.min(), density.max()
    if d_max - d_min < 1e-30:
        return np.zeros_like(density)
    return (density - d_min) / (d_max - d_min)


def phase_color_values(phase: NDArray) -> NDArray:
    """Map phase from [−π, π] → [0, 1] for colourmap mapping."""
    return (phase + np.pi) / (2.0 * np.pi)
