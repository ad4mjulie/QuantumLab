"""Service layer for physics simulations."""

from __future__ import annotations
from typing import Any
import numpy as np
from core.models import OrbitalParams, SimulationResult, OrbitalInfo
from core.exceptions import ResourceNotFoundError
from physics.hydrogen_solver import HydrogenSolver
from physics.orbitals import get_orbital


class PhysicsService:
    """Orchestrates physics simulations and bundles results."""

    def __init__(self) -> None:
        self.solver = HydrogenSolver()

    def list_available_orbitals(self) -> list[OrbitalInfo]:
        """Return a list of all supported hydrogen orbitals."""
        from physics.orbitals import ORBITAL_CATALOG
        return [
            OrbitalInfo(name=name, n=q[0], l=q[1], m=q[2])
            for name, q in ORBITAL_CATALOG.items()
        ]

    def run_orbital_simulation(self, params: OrbitalParams) -> SimulationResult:
        """Run a hydrogen orbital simulation or superposition and return bundled result."""
        if params.name.lower() == "superposition":
            return self.run_superposition_simulation(params)
            
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

    def run_superposition_simulation(self, params: OrbitalParams) -> SimulationResult:
        """
        Prepare a superposition of the first 5 basis states and sample density.
        1s, 2s, 2px, 2py, 2pz
        """
        basis = ["1s", "2s", "2px", "2py", "2pz"]
        pts_per_orbital = params.n_points // len(basis)
        
        all_points = []
        all_densities = []
        
        for i, name in enumerate(basis):
            n, l, m = get_orbital(name)
            # Use a slightly different seed for each component if a seed is provided
            comp_seed = params.seed + i if params.seed is not None else None
            p, d = self.solver.sample_points_mc(n, l, m, n_points=pts_per_orbital, seed=comp_seed)
            all_points.append(p)
            all_densities.append(d)
            
        points = np.vstack(all_points)
        densities = np.concatenate(all_densities)
        
        return SimulationResult(
            title="Hydrogen Orbital Superposition",
            n_points=len(points),
            energy_ev=None,  # Mixed state has no single energy level
            points=points.tolist(),
            values=densities.tolist(),
            metadata={"basis": basis, "is_superposition": True}
        )
