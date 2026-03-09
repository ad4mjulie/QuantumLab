"""
Hydrogen atom solver.

Provides the ``HydrogenSolver`` class which:
  • evaluates the wavefunction / probability density on a 3-D grid
  • performs Monte-Carlo rejection sampling to build point clouds
  • returns analytic energy levels
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from physics.wavefunctions import (
    hydrogen_wavefunction,
    probability_density,
    cartesian_to_spherical,
    spherical_to_cartesian,
)


class HydrogenSolver:
    """High-level solver for hydrogen-like atom orbitals."""

    # Hydrogen ground-state energy in eV
    _E1 = -13.6  # eV

    # ------------------------------------------------------------------
    # Energy
    # ------------------------------------------------------------------

    @staticmethod
    def get_energy(n: int) -> float:
        """Return the energy level Eₙ = −13.6 / n² eV."""
        if n < 1:
            raise ValueError(f"n must be ≥ 1, got {n}")
        return -13.6 / n ** 2

    # ------------------------------------------------------------------
    # Grid evaluation
    # ------------------------------------------------------------------

    def compute_orbital(
        self,
        n: int,
        l: int,
        m: int,
        grid_size: int = 80,
        r_max: float | None = None,
    ) -> dict:
        """
        Evaluate ψ and |ψ|² on a uniform Cartesian grid.

        Parameters
        ----------
        n, l, m    : quantum numbers
        grid_size  : number of points per axis (total = grid_size³)
        r_max      : half-width of the cube in Bohr radii (auto-scaled if None)

        Returns
        -------
        dict with keys:
            x, y, z         – 1-D coordinate arrays
            psi             – complex 3-D array of wavefunction values
            density         – real 3-D array of |ψ|²
            grid_shape      – (grid_size, grid_size, grid_size)
        """
        if r_max is None:
            r_max = float(2 * n ** 2 + 10)

        lin = np.linspace(-r_max, r_max, grid_size)
        X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")

        r, theta, phi = cartesian_to_spherical(X, Y, Z)

        psi = hydrogen_wavefunction(n, l, m, r, theta, phi)
        density = np.real(psi * np.conj(psi))

        return {
            "x": lin,
            "y": lin,
            "z": lin,
            "psi": psi,
            "density": density,
            "grid_shape": (grid_size, grid_size, grid_size),
        }

    # ------------------------------------------------------------------
    # Monte-Carlo sampling
    # ------------------------------------------------------------------

    def sample_points_mc(
        self,
        n: int,
        l: int,
        m: int,
        n_points: int = 100_000,
        r_max: float | None = None,
        seed: int | None = None,
    ) -> tuple[NDArray, NDArray]:
        """
        Monte-Carlo rejection sampling of the probability density |ψ|².

        Returns points distributed according to the electron probability cloud.

        Parameters
        ----------
        n, l, m   : quantum numbers
        n_points  : desired number of accepted points
        r_max     : maximum radial distance to sample
        seed      : optional random seed for reproducibility

        Returns
        -------
        points : (n_points, 3) array of Cartesian coordinates
        values : (n_points,) array of |ψ|² at each point
        """
        rng = np.random.default_rng(seed)

        if r_max is None:
            r_max = float(2 * n ** 2 + 10)

        accepted_pts: list[NDArray] = []
        accepted_vals: list[NDArray] = []
        total_accepted = 0

        # Estimate peak density from a small pilot sample for envelope
        pilot_r = np.linspace(0.01, r_max, 500)
        pilot_theta = np.linspace(0, np.pi, 100)
        pilot_phi = np.zeros_like(pilot_theta)
        RR, TT = np.meshgrid(pilot_r, pilot_theta, indexing="ij")
        pilot_density = probability_density(
            n, l, m, RR.ravel(), TT.ravel(), np.zeros(RR.size)
        )
        max_density = float(np.max(pilot_density)) * 1.2  # safety margin

        if max_density == 0:
            max_density = 1e-10

        batch = max(n_points, 200_000)

        while total_accepted < n_points:
            # Uniform sampling in a cube
            x = rng.uniform(-r_max, r_max, batch)
            y = rng.uniform(-r_max, r_max, batch)
            z = rng.uniform(-r_max, r_max, batch)

            r, theta, phi = cartesian_to_spherical(x, y, z)
            density = probability_density(n, l, m, r, theta, phi)

            # Rejection step
            u = rng.uniform(0, max_density, batch)
            mask = u < density

            accepted_pts.append(np.column_stack([x[mask], y[mask], z[mask]]))
            accepted_vals.append(density[mask])
            total_accepted += int(np.sum(mask))

        points = np.vstack(accepted_pts)[:n_points]
        values = np.concatenate(accepted_vals)[:n_points]
        return points, values

    # ------------------------------------------------------------------
    # Wavefunction phase (for colour mapping)
    # ------------------------------------------------------------------

    def compute_phase(
        self,
        n: int,
        l: int,
        m: int,
        points: NDArray,
    ) -> NDArray:
        """
        Return the complex phase angle of ψ at each (x, y, z) point.

        Useful for phase-based colour mapping in visualisations.
        """
        r, theta, phi = cartesian_to_spherical(
            points[:, 0], points[:, 1], points[:, 2]
        )
        psi = hydrogen_wavefunction(n, l, m, r, theta, phi)
        return np.angle(psi)
