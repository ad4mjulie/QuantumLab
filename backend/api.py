from typing import List, Dict, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse

from core import (
    OrbitalParams, SimulationResult, GroverParams, VQEParams,
    GroverResult, VQEResult, OrbitalInfo, OrbitalListResponse,
    HydrogenMeasurementParams, HydrogenMeasurementResult
)
from core import settings
from core.exceptions import QuantumLabError, ResourceNotFoundError, ValidationError
from services import PhysicsService, QuantumService

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
)

# Shared services
def get_physics_service() -> PhysicsService: return PhysicsService()
def get_quantum_service() -> QuantumService: return QuantumService()

@app.exception_handler(QuantumLabError)
async def quantumlab_exception_handler(request: Request, exc: QuantumLabError):
    if isinstance(exc, ResourceNotFoundError):
        status_code = 404
    elif isinstance(exc, ValidationError):
        status_code = 400
    else:
        status_code = 500
    
    return JSONResponse(
        status_code=status_code,
        content={"detail": str(exc), "type": exc.__class__.__name__},
    )

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "QuantumLab"}

@app.get("/orbitals", response_model=OrbitalListResponse)
def orbitals(service: PhysicsService = Depends(get_physics_service)):
    """List all available hydrogen orbitals."""
    return OrbitalListResponse(orbitals=service.list_available_orbitals())

@app.post("/simulate/orbital", response_model=SimulationResult)
def simulate_orbital(
    req: OrbitalParams, 
    service: PhysicsService = Depends(get_physics_service)
):
    """Run orbital simulation via PhysicsService."""
    return service.run_orbital_simulation(req)

@app.post("/simulate/grover", response_model=GroverResult)
def simulate_grover(
    req: GroverParams, 
    service: QuantumService = Depends(get_quantum_service)
):
    """Run Grover algorithm via QuantumService."""
    return service.run_grover(req)

@app.post("/simulate/vqe", response_model=VQEResult)
def simulate_vqe(
    req: VQEParams, 
    service: QuantumService = Depends(get_quantum_service)
):
    """Run VQE via QuantumService."""
    return service.run_vqe(req)

@app.post("/simulate/hydrogen/measure", response_model=HydrogenMeasurementResult)
def measure_hydrogen(
    req: HydrogenMeasurementParams,
    service: QuantumService = Depends(get_quantum_service)
):
    """Perform a quantum measurement of the hydrogen electron register."""
    return service.measure_electron(seed=req.seed)

@app.get("/", response_class=HTMLResponse)
def web_root():
    """Serve the minimal web interface."""
    from ui.web_interface import get_html
    return HTMLResponse(content=get_html())
