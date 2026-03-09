"""
FastAPI backend for the Quantum Simulation Lab.

Exposes REST endpoints to run simulations programmatically and serves
the minimal web interface.
"""

from __future__ import annotations

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from physics.hydrogen_solver import HydrogenSolver
from physics.orbitals import list_orbitals, get_orbital, ORBITAL_CATALOG
from physics.harmonic_oscillator import sample_harmonic_3d
from quantum.grover import GroverSearch
from quantum.vqe import VQESolver

app = FastAPI(
    title="QuantumLab API",
    description="Integrated Quantum Simulation Lab — REST API",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "QuantumLab"}


# ---------------------------------------------------------------------------
# Orbitals catalogue
# ---------------------------------------------------------------------------

@app.get("/orbitals")
def orbitals():
    """List all available hydrogen orbitals."""
    return {
        "orbitals": [
            {"name": name, "n": q[0], "l": q[1], "m": q[2]}
            for name, q in ORBITAL_CATALOG.items()
        ]
    }


# ---------------------------------------------------------------------------
# Orbital simulation
# ---------------------------------------------------------------------------

class OrbitalRequest(BaseModel):
    orbital: str = Field("1s", description="Orbital name, e.g. '2p0', '3d+1'")
    n_points: int = Field(50000, ge=1000, le=500000)


@app.post("/simulate/orbital")
def simulate_orbital(req: OrbitalRequest):
    """Sample a hydrogen orbital and return the point cloud as JSON."""
    n, l, m = get_orbital(req.orbital)
    solver = HydrogenSolver()
    points, values = solver.sample_points_mc(n, l, m, req.n_points)
    phase = solver.compute_phase(n, l, m, points)

    return {
        "orbital": req.orbital,
        "n": n, "l": l, "m": m,
        "energy_eV": solver.get_energy(n),
        "n_points": len(points),
        "points": points.tolist(),
        "density": values.tolist(),
        "phase": phase.tolist(),
    }


# ---------------------------------------------------------------------------
# Harmonic oscillator
# ---------------------------------------------------------------------------

class HarmonicRequest(BaseModel):
    nx: int = Field(0, ge=0, le=10)
    ny: int = Field(0, ge=0, le=10)
    nz: int = Field(0, ge=0, le=10)
    n_points: int = Field(50000, ge=1000, le=500000)


@app.post("/simulate/harmonic")
def simulate_harmonic(req: HarmonicRequest):
    """Sample a 3-D harmonic oscillator eigenstate."""
    points, values = sample_harmonic_3d(req.nx, req.ny, req.nz, req.n_points)
    return {
        "quantum_numbers": {"nx": req.nx, "ny": req.ny, "nz": req.nz},
        "n_points": len(points),
        "points": points.tolist(),
        "density": values.tolist(),
    }


# ---------------------------------------------------------------------------
# Grover
# ---------------------------------------------------------------------------

class GroverRequest(BaseModel):
    n_qubits: int = Field(3, ge=1, le=8)
    target: str = Field("101", description="Target bitstring")
    shots: int = Field(1024, ge=1, le=100000)


@app.post("/simulate/grover")
def simulate_grover(req: GroverRequest):
    """Run Grover's algorithm and return measurement counts."""
    g = GroverSearch()
    counts = g.run(req.n_qubits, req.target, shots=req.shots)
    total = sum(counts.values())
    return {
        "n_qubits": req.n_qubits,
        "target": req.target,
        "shots": req.shots,
        "counts": counts,
        "probabilities": {k: v / total for k, v in counts.items()},
        "found": max(counts, key=counts.get),
    }


# ---------------------------------------------------------------------------
# VQE
# ---------------------------------------------------------------------------

class VQERequest(BaseModel):
    n_qubits: int = Field(2, ge=2, le=6)
    depth: int = Field(2, ge=1, le=8)
    shots: int = Field(4096, ge=1, le=100000)
    maxiter: int = Field(100, ge=10, le=1000)


@app.post("/simulate/vqe")
def simulate_vqe(req: VQERequest):
    """Run VQE optimisation and return results."""
    solver = VQESolver(n_qubits=req.n_qubits, depth=req.depth, shots=req.shots)
    result = solver.optimize(maxiter=req.maxiter)
    return {
        "n_qubits": req.n_qubits,
        "depth": req.depth,
        "optimal_energy": result["optimal_energy"],
        "n_function_evals": result["n_iterations"],
        "optimal_params": result["optimal_params"].tolist(),
    }


# ---------------------------------------------------------------------------
# Web UI (served as HTML)
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def web_root():
    """Serve the minimal web interface."""
    from ui.web_interface import get_html
    return HTMLResponse(content=get_html())
