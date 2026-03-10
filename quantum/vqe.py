"""
Variational Quantum Eigensolver (VQE).

Implements a simple VQE loop:
  1. Build a parameterised ansatz  (RY + CNOT layers)
  2. Evaluate ⟨ψ(θ)|H|ψ(θ)⟩ via measurement on the Aer simulator
  3. Minimise with ``scipy.optimize.minimize``

The default Hamiltonian is the 2-qubit Heisenberg-like diagonal:
    H = Z⊗Z  (diagonal → easy to measure in computational basis)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator





class VQESolver:
    """Simple VQE solver for diagonal Hamiltonians."""

    def __init__(
        self, 
        n_qubits: int = 2, 
        depth: int = 2, 
        shots: int = 4096,
        seed: int | None = None
    ):
        self.n_qubits = n_qubits
        self.depth = depth
        self.shots = shots
        self.seed = seed

    # ------------------------------------------------------------------
    # Ansatz
    # ------------------------------------------------------------------

    def build_ansatz(self, params: NDArray) -> QuantumCircuit:
        """
        Construct an RY + CNOT layered ansatz.

        Parameters
        ----------
        params : 1-D array of length ``n_qubits * depth``

        Returns
        -------
        QuantumCircuit (without measurement gates)
        """
        qc = QuantumCircuit(self.n_qubits)
        idx = 0
        for _layer in range(self.depth):
            for q in range(self.n_qubits):
                qc.ry(params[idx], q)
                idx += 1
            for q in range(self.n_qubits - 1):
                qc.cx(q, q + 1)
        return qc

    # ------------------------------------------------------------------
    # Expectation value  ⟨Z⊗Z⊗…⊗Z⟩
    # ------------------------------------------------------------------

    def _measure_zz(self, qc: QuantumCircuit) -> float:
        """
        Measure ⟨Z⊗Z⟩ by running in the computational basis and assigning
        eigenvalues  +1 / −1  to each bitstring.
        """
        from quantum.circuits import _SIMULATOR
        
        meas_qc = qc.copy()
        meas_qc.measure_all()

        job = _SIMULATOR.run(meas_qc, shots=self.shots, seed_simulator=self.seed)
        counts = job.result().get_counts()

        expectation = 0.0
        total = sum(counts.values())
        for bitstring, count in counts.items():
            eigenvalue = 1
            for bit in bitstring.replace(" ", ""):
                eigenvalue *= (-1) ** int(bit)
            expectation += eigenvalue * count / total
        return expectation

    # ------------------------------------------------------------------
    # Cost function
    # ------------------------------------------------------------------

    def cost_function(self, params: NDArray) -> float:
        """Evaluate ⟨ψ(θ)|H|ψ(θ)⟩ for H = Z⊗Z."""
        qc = self.build_ansatz(params)
        return self._measure_zz(qc)

    # ------------------------------------------------------------------
    # Optimisation loop
    # ------------------------------------------------------------------

    def optimize(
        self,
        method: str = "COBYLA",
        maxiter: int = 200,
    ) -> dict:
        """
        Run the full VQE optimisation.
        """
        n_params = self.n_qubits * self.depth
        
        # Reproducible initial parameters
        if self.seed is not None:
            rng = np.random.default_rng(self.seed)
            initial_params = rng.uniform(0, 2 * np.pi, n_params)
        else:
            initial_params = np.random.uniform(0, 2 * np.pi, n_params)

        history: list[float] = []

        def _callback(xk: NDArray) -> None:
            energy = self.cost_function(xk)
            history.append(float(energy))

        result = minimize(
            self.cost_function,
            initial_params,
            method=method,
            callback=_callback,
            options={"maxiter": maxiter},
        )

        final_energy = float(result.fun)
        if not history or abs(history[-1] - final_energy) > 1e-6:
            history.append(final_energy)

        return {
            "optimal_params": result.x,
            "optimal_energy": final_energy,
            "n_iterations": result.nfev,
            "history": history,
            "success": result.success,
            "message": result.message
        }
