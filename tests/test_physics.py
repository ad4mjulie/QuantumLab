"""Wavefunction and physics unit tests."""

import numpy as np
import pytest
from physics.wavefunctions import (
    radial_wavefunction, 
    spherical_harmonic, 
    hydrogen_wavefunction, 
    probability_density
)
from physics.validation import check_normalization


def test_radial_wavefunction_normalization():
    """Test if radial wavefunction R10 is correctly calculated at r=0."""
    # R10(0) = 2 * (1/a0)^1.5
    r = np.array([0.0])
    R = radial_wavefunction(1, 0, r)
    assert np.isclose(R[0], 2.0)


def test_hydrogen_1s_normalization():
    """
    Verify 1s orbital normalization using radial quadrature.

    ∫|ψ|² dV = ∫₀^∞ |R(r)|² r² dr = 1  (radial part; Ylm are normalised separately)

    This is far more accurate than a Cartesian grid integral because it uses
    scipy's adaptive quadrature which handles the exponential tail cleanly.
    """
    from scipy.integrate import quad
    from physics.wavefunctions import radial_wavefunction

    def radial_integrand(r: float) -> float:
        rr = np.array([r])
        R = radial_wavefunction(1, 0, rr)
        return float(R[0] ** 2 * r ** 2)

    result, error = quad(radial_integrand, 0, np.inf, limit=200)
    # Should integrate to 1.0 with very high precision
    assert abs(result - 1.0) < 1e-6, f"Normalization integral = {result:.8f}, error = {error:.2e}"


@pytest.mark.parametrize("n, l, m", [
    (1, 0, 0),
    (2, 1, 0),
    (3, 2, 1),
])
def test_wavefunction_shapes(n, l, m):
    """Ensure wavefunctions return correct shapes for array inputs."""
    r = np.linspace(0.1, 10, 10)
    theta = np.linspace(0, np.pi, 10)
    phi = np.linspace(0, 2*np.pi, 10)
    
    psi = hydrogen_wavefunction(n, l, m, r, theta, phi)
    assert psi.shape == (10,)
