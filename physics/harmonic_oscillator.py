"""
Quantum harmonic oscillator.

Provides analytic eigenstates of the 1-D and 3-D isotropic quantum harmonic
oscillator, using Hermite polynomials from ``scipy.special``.

In natural units (ℏ = m = ω = 1):
    ψₙ(x) = (2ⁿ n! √π)^{-1/2} · Hₙ(x) · exp(-x²/2)
    Eₙ    = n + 1/2             (1-D)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import hermite
from scipy.special import factorial


# ---------------------------------------------------------------------------
# 1-D quantum harmonic oscillator
# ---------------------------------------------------------------------------

def harmonic_wavefunction(n: int, x: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Evaluate the n-th eigenstate ψₙ(x) of the 1-D harmonic oscillator.

    Parameters
    ----------
    n : int    – quantum number (n ≥ 0)
    x : array  – position values

    Returns
    -------
    psi : array – wavefunction values
    """
    if n < 0:
        raise ValueError(f"n must be ≥ 0, got {n}")
    H_n = hermite(n)
    norm = (2 ** n * factorial(n, exact=True) * np.sqrt(np.pi)) ** (-0.5)
    return norm * H_n(x) * np.exp(-x ** 2 / 2.0)


def harmonic_probability(n: int, x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return |ψₙ(x)|² for the 1-D oscillator."""
    psi = harmonic_wavefunction(n, x)
    return psi ** 2


def harmonic_energy(n: int) -> float:
    """Energy of the n-th level: Eₙ = n + 1/2 (natural units)."""
    return n + 0.5


# ---------------------------------------------------------------------------
# 3-D isotropic harmonic oscillator (for point-cloud visualisation)
# ---------------------------------------------------------------------------

def harmonic_3d_density(
    nx: int,
    ny: int,
    nz: int,
    x: NDArray,
    y: NDArray,
    z: NDArray,
) -> NDArray[np.floating]:
    """
    Product separable 3-D isotropic harmonic oscillator density.

    ρ(x,y,z) = |ψ_{nx}(x)|² |ψ_{ny}(y)|² |ψ_{nz}(z)|²
    """
    return (
        harmonic_probability(nx, x)
        * harmonic_probability(ny, y)
        * harmonic_probability(nz, z)
    )


def sample_harmonic_3d(
    nx: int = 0,
    ny: int = 0,
    nz: int = 0,
    n_points: int = 100_000,
    x_max: float = 6.0,
) -> tuple[NDArray, NDArray]:
    """
    Monte-Carlo rejection sampling for the 3-D harmonic oscillator.

    Returns
    -------
    points : (n_points, 3) Cartesian coordinates
    values : (n_points,)   density at each point
    """
    accepted_pts: list[NDArray] = []
    accepted_vals: list[NDArray] = []
    total = 0

    # Estimate peak density
    x_pilot = np.linspace(-x_max, x_max, 500)
    peak = float(np.max(harmonic_probability(nx, x_pilot)))
    peak *= float(np.max(harmonic_probability(ny, x_pilot)))
    peak *= float(np.max(harmonic_probability(nz, x_pilot)))
    peak *= 1.2  # safety margin
    if peak == 0:
        peak = 1e-10

    batch = max(n_points, 200_000)

    while total < n_points:
        xs = np.random.uniform(-x_max, x_max, batch)
        ys = np.random.uniform(-x_max, x_max, batch)
        zs = np.random.uniform(-x_max, x_max, batch)

        density = harmonic_3d_density(nx, ny, nz, xs, ys, zs)
        u = np.random.uniform(0, peak, batch)
        mask = u < density

        accepted_pts.append(np.column_stack([xs[mask], ys[mask], zs[mask]]))
        accepted_vals.append(density[mask])
        total += int(np.sum(mask))

    points = np.vstack(accepted_pts)[:n_points]
    values = np.concatenate(accepted_vals)[:n_points]
    return points, values
