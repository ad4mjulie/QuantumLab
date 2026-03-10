"""Service layer for physics simulations."""

from __future__ import annotations
from typing import Any
import numpy as np
from core.models import OrbitalParams, SimulationResult
from core.exceptions import ResourceNotFoundError
from physics.hydrogen_solver import HydrogenSolver
from physics.orbitals import get_orbital


class PhysicsService:
    """Orchestrates physics simulations and bundles results."""

    def __init__(self) -> None:
        self.solver = HydrogenSolver()

    def list_available_orbitals(self) -> list[dict[str, any]]:
        """Return a list of all supported hydrogen orbitals."""
        from physics.orbitals import ORBITAL_CATALOG
        return [
            {"name": name, "n": q[0], "l": q[1], "m": q[2]}
            for name, q in ORBITAL_CATALOG.items()
        ]

    def run_orbital_simulation(self, params: OrbitalParams) -> SimulationResult:
        """Run a hydrogen orbital simulation and return bundled result."""
        try:
            n, l, m = get_orbital(params.name)
        except KeyError:
            raise ResourceNotFoundError(f"Orbital '{params.name}' not found.")

        # Run Monte-Carlo sampling
        points, density = self.solver.sample_points_mc(
            n, l, m, n_points=params.n_points, seed=params.seed
        )
        
        return SimulationResult(
            title=f"Hydrogen Orbital {params.name}",
            n_points=len(points),
            energy_ev=self.solver.get_energy(n),
            points=points.tolist(),
            values=density.tolist(),
            metadata={"n": n, "l": l, "m": m}
        )
