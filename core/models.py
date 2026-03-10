"""Shared domain models and schemas for QuantumLab."""

from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class OrbitalParams(BaseModel):
    """Parameters for hydrogen orbital simulation."""
    name: str = Field(..., description="Orbital name (e.g., '2p0')")
    n_points: int = Field(100000, ge=1000, le=1000000)
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class HydrogenMeasurementParams(BaseModel):
    """Parameters for one-shot quantum measurement of the electron."""
    seed: Optional[int] = Field(None, description="Random seed for measurement")


class HydrogenMeasurementResult(BaseModel):
    """Result of a single-shot quantum measurement of the electron."""
    orbital: str = Field(..., description="The collapsed orbital name")
    bitstring: str = Field(..., description="The measured bitstring")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SimulationResult(BaseModel):
    """Generic container for simulation output point clouds."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    title: str
    n_points: int
    energy_ev: Optional[float] = None
    points: List[List[float]] = Field(..., description="N x 3 list of coordinates")
    values: List[float] = Field(..., description="Scalar values at each point (density or phase)")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GroverParams(BaseModel):
    """Parameters for Grover's algorithm."""
    n_qubits: int = Field(..., ge=1, le=10)
    target: str
    shots: int = 1024
    seed: Optional[int] = None


class VQEParams(BaseModel):
    """Parameters for VQE solver."""
    n_qubits: int = Field(2, ge=2, le=6)
    depth: int = Field(2, ge=1)
    maxiter: int = 200
    shots: int = 4096
    seed: Optional[int] = None

class GroverResult(BaseModel):
    """Result of Grover's search algorithm."""
    counts: Dict[str, int]
    probabilities: Dict[str, float]
    found: str


class VQEResult(BaseModel):
    """Result of VQE optimization."""
    optimal_energy: float
    n_iterations: int
    history: List[float]


class OrbitalInfo(BaseModel):
    """Metadata for an available orbital."""
    name: str
    n: int
    l: int
    m: int


class OrbitalListResponse(BaseModel):
    """Wrapper for the list of available orbitals."""
    orbitals: List[OrbitalInfo]


# Ensure all types are resolved for Pydantic V2
SimulationResult.model_rebuild()
GroverResult.model_rebuild()
VQEResult.model_rebuild()
OrbitalInfo.model_rebuild()
OrbitalListResponse.model_rebuild()
HydrogenMeasurementResult.model_rebuild()
