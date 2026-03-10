"""Shared domain models and schemas for QuantumLab."""

from __future__ import annotations
from typing import List, Optional, Dict, Any
import typing
from pydantic import BaseModel, Field, ConfigDict


class OrbitalParams(BaseModel):
    """Parameters for hydrogen orbital simulation."""
    name: str = Field(..., description="Orbital name (e.g., '2p0')")
    n_points: int = Field(100000, ge=1000, le=1000000)
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class SimulationResult(BaseModel):
    """Generic container for simulation output point clouds."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    title: str
    n_points: int
    energy_ev: Optional[float] = None
    points: List[List[float]] = Field(..., description="N x 3 list of coordinates")
    values: List[float] = Field(..., description="Scalar values at each point (density or phase)")
    metadata: typing.Dict[str, typing.Any] = Field(default_factory=dict)


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

# Ensure all types are resolved for Pydantic V2
SimulationResult.model_rebuild()
