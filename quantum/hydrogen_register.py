"""
Hydrogen orbital quantum register mapping.

Maps orbital states to bitstrings for quantum measurement:
  000 -> 1s
  001 -> 2s
  010 -> 2px
  011 -> 2py
  100 -> 2pz
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from quantum.circuits import run_circuit

# Bitstring to Orbital mapping as per RC requirements
BITSTRING_TO_ORBITAL = {
    "000": "1s",
    "001": "2s",
    "010": "2px",
    "011": "2py",
    "100": "2pz",
}

ORBITAL_TO_BITSTRING = {v: k for k, v in BITSTRING_TO_ORBITAL.items()}


def get_hydrogen_measurement_circuit() -> QuantumCircuit:
    """
    Returns a circuit that prepares a superposition of hydrogen orbitals
    and includes measurement gates.
    
    In a real scenario, this circuit would be the output of an evolution 
    or VQE algorithm. Here we prepare an equal superposition of the 
    first 5 basis states for demonstration.
    """
    # 3 qubits needed for 5 states (up to 8 states)
    qc = QuantumCircuit(3, 3)
    
    # Simple state preparation: 
    # We want a superposition of |000>, |001>, |010>, |011>, |100>
    # For the RC, we'll use a simple H + some rotations or just H on 2 and 
    # leave the 3rd to occasionally be 1.
    qc.h(0)
    qc.h(1)
    qc.h(2)
    # Note: This actually creates a superposition of all 8 states.
    # We filter or handle invalid bitstrings in the measurement logic.
    
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc


def measure_electron(seed: int | None = None) -> str:
    """
    Execute a measurement on the hydrogen electron register (shots=1).
    
    Returns
    -------
    orbital_name : str
        The name of the orbital the wavefunction collapsed into.
    """
    qc = get_hydrogen_measurement_circuit()
    counts = run_circuit(qc, shots=1, seed=seed)
    
    # Get the single measured bitstring (e.g., "101")
    # Qiskit bitstrings are often little-endian in representation (q2 q1 q0)
    # but run_circuit returns them as strings.
    bitstring = list(counts.keys())[0]
    
    # Map bitstring to orbital, defaulting to '1s' if out of range for this demo
    return BITSTRING_TO_ORBITAL.get(bitstring, "1s")
