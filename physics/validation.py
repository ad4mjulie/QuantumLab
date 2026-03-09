"""Scientific validation utilities for quantum physics simulations."""

import numpy as np
from typing import Callable


def numerical_integration_3d(
    func: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    r_max: float,
    grid_size: int = 100
) -> float:
    """
    Perform 3D numerical integration of a function over a cube [-r_max, r_max]^3.
    Useful for checking wavefunction normalization.
    """
    lin = np.linspace(-r_max, r_max, grid_size)
    dx = lin[1] - lin[0]
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    
    values = func(X, Y, Z)
    return float(np.sum(values) * (dx**3))


def check_normalization(
    density_func: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    r_max: float
) -> float:
    """Check if the probability density integrates to approximately 1.0."""
    return numerical_integration_3d(density_func, r_max)
