import pytest
import numpy as np
from services.physics_service import PhysicsService
from services.quantum_service import QuantumService
from core.models import OrbitalParams, GroverParams, VQEParams

def test_physics_determinism():
    service = PhysicsService()
    params = OrbitalParams(name="2p0", n_points=5000, seed=123)
    
    res1 = service.run_orbital_simulation(params)
    res2 = service.run_orbital_simulation(params)
    
    assert np.allclose(res1.points, res2.points)
    assert np.allclose(res1.values, res2.values)

def test_grover_determinism():
    service = QuantumService()
    params = GroverParams(n_qubits=3, target="110", shots=1024, seed=123)
    
    res1 = service.run_grover(params)
    res2 = service.run_grover(params)
    
    assert res1.counts == res2.counts

def test_vqe_determinism():
    service = QuantumService()
    params = VQEParams(n_qubits=2, depth=1, maxiter=5, seed=123)
    
    res1 = service.run_vqe(params)
    res2 = service.run_vqe(params)
    
    assert np.allclose(res1.optimal_energy, res2.optimal_energy)
    assert res1.history == res2.history
