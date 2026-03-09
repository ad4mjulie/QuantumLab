"""
Quantum circuit utilities.

Thin wrapper around Qiskit for building and executing circuits on the
Aer simulator, plus helpers for interpreting results.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# Simulator singleton
# ---------------------------------------------------------------------------
_SIMULATOR = AerSimulator()


def run_circuit(
    circuit: QuantumCircuit,
    shots: int = 1024,
    seed: int | None = None,
) -> dict[str, int]:
    """
    Execute a quantum circuit on the Aer simulator and return raw counts.

    Parameters
    ----------
    circuit : QuantumCircuit – must already contain measurements
    shots   : int            – number of measurement shots
    seed     : int            – optional seed for reproducibility

    Returns
    -------
    counts : dict[str, int]  – e.g. {'00': 512, '11': 512}
    """
    job = _SIMULATOR.run(circuit, shots=shots, seed_simulator=seed)
    return job.result().get_counts()


def counts_to_probabilities(counts: dict[str, int]) -> dict[str, float]:
    """Normalise measurement counts to probabilities."""
    total = sum(counts.values())
    return {state: c / total for state, c in counts.items()}


# ---------------------------------------------------------------------------
# Demo circuits
# ---------------------------------------------------------------------------

def create_bell_state() -> QuantumCircuit:
    """Build a simple Bell-state circuit |Φ⁺⟩ = (|00⟩ + |11⟩)/√2."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def create_ghz_state(n_qubits: int = 3) -> QuantumCircuit:
    """Build an n-qubit GHZ state."""
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(0)
    for i in range(1, n_qubits):
        qc.cx(0, i)
    qc.measure_all(add_bits=False)
    return qc
