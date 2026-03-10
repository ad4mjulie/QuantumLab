"""
Quantum wavefunctions for the hydrogen atom.

Implements the analytic solution ψ(r,θ,φ) = Rₙₗ(r) · Yₗᵐ(θ,φ) where:
  - Rₙₗ(r) is the radial wavefunction built from associated Laguerre polynomials
  - Yₗᵐ(θ,φ) are the spherical harmonics

All coordinates use atomic units (Bohr radii a₀ = 1).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import sph_harm_y, factorial, assoc_laguerre


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
A0 = 1.0  # Bohr radius in atomic units


# ---------------------------------------------------------------------------
# Radial wavefunction  Rₙₗ(r)
# ---------------------------------------------------------------------------

def radial_wavefunction(
    n: int, l: int, r: NDArray[np.floating]
) -> NDArray[np.floating]:
    """
    Compute the radial wavefunction Rₙₗ(r) for hydrogen.

    Parameters
    ----------
    n : int   – principal quantum number (n ≥ 1)
    l : int   – angular momentum quantum number (0 ≤ l < n)
    r : array – radial distances in units of a₀

    Returns
    -------
    R : array – radial wavefunction evaluated at each r
    """
    if n < 1:
        raise ValueError(f"n must be ≥ 1, got {n}")
    if l < 0 or l >= n:
        raise ValueError(f"l must satisfy 0 ≤ l < n, got l={l}, n={n}")

    rho = 2.0 * r / (n * A0)

    # Normalisation constant
    norm = np.sqrt(
        (2.0 / (n * A0)) ** 3
        * factorial(n - l - 1, exact=True)
        / (2.0 * n * factorial(n + l, exact=True))
    )

    # Associated Laguerre polynomial L^{2l+1}_{n-l-1}(rho)
    laguerre_vals = assoc_laguerre(rho, n - l - 1, 2 * l + 1)

    R = norm * np.exp(-rho / 2.0) * rho ** l * laguerre_vals
    return R


# ---------------------------------------------------------------------------
# Spherical harmonic wrapper  Yₗᵐ(θ, φ)
# ---------------------------------------------------------------------------

def spherical_harmonic(
    l: int,
    m: int,
    theta: NDArray[np.floating],
    phi: NDArray[np.floating],
) -> NDArray[np.complexfloating]:
    """
    Compute the complex spherical harmonic Yₗᵐ(θ, φ).

    Uses the Condon–Shortley phase convention (included in scipy).

    Parameters
    ----------
    l     : int   – degree  (l ≥ 0)
    m     : int   – order   (−l ≤ m ≤ l)
    theta : array – polar angle  [0, π]
    phi   : array – azimuthal angle [0, 2π)

    Returns
    -------
    Y : complex array
    """
    # scipy 1.15+ convention: sph_harm_y(l, m, theta, phi)
    return sph_harm_y(l, m, theta, phi)


# ---------------------------------------------------------------------------
# Full hydrogen wavefunction  ψ(r, θ, φ)
# ---------------------------------------------------------------------------

def hydrogen_wavefunction(
    n: int,
    l: int,
    m: int,
    r: NDArray[np.floating],
    theta: NDArray[np.floating],
    phi: NDArray[np.floating],
) -> NDArray[np.complexfloating]:
    """
    Full hydrogen wavefunction ψₙₗₘ(r, θ, φ).

    Returns
    -------
    psi : complex array – wavefunction values
    """
    return radial_wavefunction(n, l, r) * spherical_harmonic(l, m, theta, phi)


# ---------------------------------------------------------------------------
# Probability density |ψ|²
# ---------------------------------------------------------------------------

def probability_density(
    n: int,
    l: int,
    m: int,
    r: NDArray[np.floating],
    theta: NDArray[np.floating],
    phi: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Probability density |ψₙₗₘ|² at the given spherical coordinates.
    """
    psi = hydrogen_wavefunction(n, l, m, r, theta, phi)
    return np.real(psi * np.conj(psi))


# ---------------------------------------------------------------------------
# Cartesian ↔ Spherical helpers
# ---------------------------------------------------------------------------

def cartesian_to_spherical(
    x: NDArray, y: NDArray, z: NDArray
) -> tuple[NDArray, NDArray, NDArray]:
    """Convert Cartesian (x, y, z) → spherical (r, θ, φ)."""
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(np.clip(z / np.maximum(r, 1e-30), -1, 1))
    phi = np.arctan2(y, x)
    return r, theta, phi


def spherical_to_cartesian(
    r: NDArray, theta: NDArray, phi: NDArray
) -> tuple[NDArray, NDArray, NDArray]:
    """Convert spherical (r, θ, φ) → Cartesian (x, y, z)."""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z
