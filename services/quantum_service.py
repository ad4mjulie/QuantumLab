"""Service layer for quantum simulations."""

from __future__ import annotations
from core.models import GroverParams, VQEParams, GroverResult, VQEResult
from quantum.grover import GroverSearch
from quantum.vqe import VQESolver


class QuantumService:
    """Orchestrates quantum algorithms and formats results."""

    def __init__(self):
        self.grover = GroverSearch()

    def run_grover(self, params: GroverParams) -> GroverResult:
        """Run Grover's algorithm and return counts and probabilities."""
        counts = self.grover.run(
            params.n_qubits, params.target, shots=params.shots, seed=params.seed
        )
        total = sum(counts.values())
        return GroverResult(
            counts=counts,
            probabilities={k: v / total for k, v in counts.items()},
            found=max(counts, key=counts.get)
        )

    def run_vqe(self, params: VQEParams) -> VQEResult:
        """Run VQE optimization and return report."""
        solver = VQESolver(
            n_qubits=params.n_qubits, 
            depth=params.depth, 
            shots=params.shots,
            seed=params.seed
        )
        result = solver.optimize(maxiter=params.maxiter)
        return VQEResult(
            optimal_energy=result["optimal_energy"],
            n_iterations=result["n_iterations"],
            history=result["history"]
        )

    def measure_electron(self, seed: int | None = None) -> dict[str, Any]:
        """
        Execute a quantum measurement of the hydrogen electron register.
        
        Returns a result containing the collapsed orbital name and bitstring.
        """
        from quantum.hydrogen_register import get_hydrogen_measurement_circuit, BITSTRING_TO_ORBITAL
        from quantum.circuits import run_circuit
        
        qc = get_hydrogen_measurement_circuit()
        counts = run_circuit(qc, shots=1, seed=seed)
        bitstring = list(counts.keys())[0]
        
        # Determine the orbital
        orbital = BITSTRING_TO_ORBITAL.get(bitstring, "1s")
        
        return {
            "orbital": orbital,
            "bitstring": bitstring,
            "metadata": {"shots": 1, "seed": seed}
        }
