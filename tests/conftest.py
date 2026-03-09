"""Pytest configuration and common fixtures."""
import pytest
from services import PhysicsService, QuantumService

@pytest.fixture
def physics_service():
    return PhysicsService()

@pytest.fixture
def quantum_service():
    return QuantumService()
