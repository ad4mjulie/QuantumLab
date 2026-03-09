"""
Grover's search algorithm implementation.

Builds the oracle + diffuser circuit for an arbitrary target bitstring and
executes it on the Aer simulator.
"""

from __future__ import annotations

import math

from qiskit import QuantumCircuit

from quantum.circuits import run_circuit, counts_to_probabilities


class GroverSearch:
    """Run Grover's algorithm to find a marked bitstring."""

    # ------------------------------------------------------------------
    # Oracle
    # ------------------------------------------------------------------

    @staticmethod
    def build_oracle(n_qubits: int, target: str) -> QuantumCircuit:
        """
        Build a phase-flip oracle that marks ``target``.

        The oracle applies a Z-phase to the target basis state using
        X gates to condition on 0-bits followed by a multi-controlled Z.

        Parameters
        ----------
        n_qubits : int – number of qubits
        target   : str – bitstring to mark, e.g. ``"110"``
        """
        if len(target) != n_qubits:
            raise ValueError(
                f"Target length ({len(target)}) ≠ n_qubits ({n_qubits})"
            )

        oracle = QuantumCircuit(n_qubits, name="Oracle")

        # Flip qubits whose target bit is '0' so that the target maps to |11…1⟩
        for i, bit in enumerate(reversed(target)):
            if bit == "0":
                oracle.x(i)

        # Multi-controlled Z = H on last qubit + MCX + H on last qubit
        if n_qubits == 1:
            oracle.z(0)
        else:
            oracle.h(n_qubits - 1)
            oracle.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            oracle.h(n_qubits - 1)

        # Undo the X flips
        for i, bit in enumerate(reversed(target)):
            if bit == "0":
                oracle.x(i)

        return oracle

    # ------------------------------------------------------------------
    # Diffuser (Grover's diffusion operator)
    # ------------------------------------------------------------------

    @staticmethod
    def build_diffuser(n_qubits: int) -> QuantumCircuit:
        """Grover diffusion operator  2|s⟩⟨s| − I  where |s⟩ = H^n|0⟩."""
        diffuser = QuantumCircuit(n_qubits, name="Diffuser")

        diffuser.h(range(n_qubits))
        diffuser.x(range(n_qubits))

        # Multi-controlled Z
        if n_qubits == 1:
            diffuser.z(0)
        else:
            diffuser.h(n_qubits - 1)
            diffuser.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            diffuser.h(n_qubits - 1)

        diffuser.x(range(n_qubits))
        diffuser.h(range(n_qubits))

        return diffuser

    # ------------------------------------------------------------------
    # Full circuit
    # ------------------------------------------------------------------

    def build_circuit(self, n_qubits: int, target: str) -> QuantumCircuit:
        """
        Assemble the full Grover circuit with the optimal number of
        iterations  ⌊π/4 · √N⌋.
        """
        N = 2 ** n_qubits
        n_iterations = max(1, int(math.pi / 4 * math.sqrt(N)))

        qc = QuantumCircuit(n_qubits, n_qubits)

        # Superposition
        qc.h(range(n_qubits))

        # Grover iterations
        oracle = self.build_oracle(n_qubits, target)
        diffuser = self.build_diffuser(n_qubits)

        for _ in range(n_iterations):
            qc.compose(oracle, inplace=True)
            qc.compose(diffuser, inplace=True)

        qc.measure(range(n_qubits), range(n_qubits))
        return qc

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        n_qubits: int,
        target: str,
        shots: int = 1024,
    ) -> dict[str, int]:
        """
        Execute Grover's algorithm and return measurement counts.

        Parameters
        ----------
        n_qubits : int – number of qubits
        target   : str – target bitstring
        shots    : int – number of shots

        Returns
        -------
        counts : dict[str, int]
        """
        qc = self.build_circuit(n_qubits, target)
        return run_circuit(qc, shots=shots)

    def run_probabilities(
        self,
        n_qubits: int,
        target: str,
        shots: int = 1024,
    ) -> dict[str, float]:
        """Run Grover's and return normalised probabilities."""
        counts = self.run(n_qubits, target, shots)
        return counts_to_probabilities(counts)
