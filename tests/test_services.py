"""Integration tests for the Service Layer."""

import pytest
from services.physics_service import PhysicsService
from services.quantum_service import QuantumService
from core.models import OrbitalParams, GroverParams


def test_physics_service_deterministic():
    """Verify that same seed produces identical orbital point clouds."""
    service = PhysicsService()
    params = OrbitalParams(name="1s", n_points=1000, seed=42)
    
    res1 = service.run_orbital_simulation(params)
    res2 = service.run_orbital_simulation(params)
    
    assert res1.points == res2.points
    assert res1.values == res2.values


def test_quantum_service_grover():
    """Verify Grover service returns correct target state."""
    service = QuantumService()
    params = GroverParams(n_qubits=3, target="101", shots=512, seed=42)
    
    result = service.run_grover(params)
    assert result.found == "101"
    assert result.probabilities["101"] > 0.8
