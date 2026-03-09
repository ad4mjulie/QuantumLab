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


_SIMULATOR = AerSimulator()


class VQESolver:
    """Simple VQE solver for diagonal Hamiltonians."""

    def __init__(self, n_qubits: int = 2, depth: int = 2, shots: int = 4096):
        self.n_qubits = n_qubits
        self.depth = depth
        self.shots = shots

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

        For a generic diagonal Hamiltonian expressed in the Z basis,
        the eigenvalue of bitstring b is  ∏ᵢ (−1)^{bᵢ}.
        """
        meas_qc = qc.copy()
        meas_qc.measure_all()

        job = _SIMULATOR.run(meas_qc, shots=self.shots)
        counts = job.result().get_counts()

        expectation = 0.0
        total = sum(counts.values())
        for bitstring, count in counts.items():
            # eigenvalue = product of (-1)^bit
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

        Returns
        -------
        dict with keys:
            optimal_params : NDArray  – best parameter vector
            optimal_energy : float    – minimised energy
            n_iterations   : int
            history        : list[float] – energy at each iteration
        """
        n_params = self.n_qubits * self.depth
        initial_params = np.random.uniform(0, 2 * np.pi, n_params)

        history: list[float] = []

        def _callback(xk: NDArray) -> None:
            energy = self.cost_function(xk)
            history.append(float(energy))
            # print(f"  Iteration {len(history)}: {energy:.6f}") # debug

        result = minimize(
            self.cost_function,
            initial_params,
            method=method,
            callback=_callback,
            options={"maxiter": maxiter},
        )

        # Ensure final energy is in history
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
