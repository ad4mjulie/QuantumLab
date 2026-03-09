"""Service layer for quantum simulations."""

from __future__ import annotations
from core.models import GroverParams, VQEParams
from quantum.grover import GroverSearch
from quantum.vqe import VQESolver


class QuantumService:
    """Orchestrates quantum algorithms and formats results."""

    def __init__(self):
        self.grover = GroverSearch()

    def run_grover(self, params: GroverParams) -> dict:
        """Run Grover's algorithm and return counts and probabilities."""
        counts = self.grover.run(
            params.n_qubits, params.target, shots=params.shots
        )
        total = sum(counts.values())
        return {
            "counts": counts,
            "probabilities": {k: v / total for k, v in counts.items()},
            "found": max(counts, key=counts.get)
        }

    def run_vqe(self, params: VQEParams) -> dict:
        """Run VQE optimization and return report."""
        solver = VQESolver(
            n_qubits=params.n_qubits, 
            depth=params.depth, 
            shots=params.shots
        )
        # Note: we need to update VQESolver to support seeds if needed
        result = solver.optimize(maxiter=params.maxiter)
        return {
            "optimal_energy": result["optimal_energy"],
            "n_iterations": result["n_iterations"],
            "history": result["history"]
        }
