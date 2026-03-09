"""
FastAPI backend for the Quantum Simulation Lab (v2).
Utilizes a Service Layer for clean separation of concerns.
"""

from __future__ import annotations
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import HTMLResponse

from core import OrbitalParams, SimulationResult, GroverParams, VQEParams
from core import settings
from services import PhysicsService, QuantumService

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
)

# Shared services
def get_physics_service(): return PhysicsService()
def get_quantum_service(): return QuantumService()


@app.get("/health")
def health():
    return {"status": "ok", "service": "QuantumLab"}


@app.get("/orbitals")
def orbitals():
    from physics.orbitals import ORBITAL_CATALOG
    return {
        "orbitals": [
            {"name": name, "n": q[0], "l": q[1], "m": q[2]}
            for name, q in ORBITAL_CATALOG.items()
        ]
    }


@app.post("/simulate/orbital", response_model=SimulationResult)
def simulate_orbital(
    req: OrbitalParams, 
    service: PhysicsService = Depends(get_physics_service)
):
    """Run orbital simulation via PhysicsService."""
    try:
        return service.run_orbital_simulation(req)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/simulate/grover")
def simulate_grover(
    req: GroverParams, 
    service: QuantumService = Depends(get_quantum_service)
):
    """Run Grover algorithm via QuantumService."""
    return service.run_grover(req)


@app.post("/simulate/vqe")
def simulate_vqe(
    req: VQEParams, 
    service: QuantumService = Depends(get_quantum_service)
):
    """Run VQE via QuantumService."""
    return service.run_vqe(req)


@app.get("/", response_class=HTMLResponse)
def web_root():
    """Serve the minimal web interface."""
    from ui.web_interface import get_html
    return HTMLResponse(content=get_html())
