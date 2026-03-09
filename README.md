# ⚛ QuantumLab — Integrated Quantum Simulation Lab

A modular Python system for **quantum physics simulation**, **quantum algorithm execution**, and **interactive 3-D visualisation**.

---

## Features

| Module | Capabilities |
|--------|-------------|
| **Physics** | Hydrogen atom orbital wavefunctions · Probability density computation · Monte-Carlo electron cloud sampling · Quantum harmonic oscillator eigenstates |
| **Quantum** | Grover's search algorithm · Variational Quantum Eigensolver (VQE) · Bell / GHZ state circuits · Qiskit Aer simulation |
| **Visualisation** | PyVista 3-D point clouds · Isosurface rendering · Bloch sphere · Plotly browser fallback · CSV export |
| **Backend** | FastAPI REST API with auto-generated docs |
| **UI** | Interactive terminal CLI · Minimal web dashboard |

---

## Project Structure

```
QuantumLab/
├── main.py                  # Entry point (CLI or --server)
├── requirements.txt
│
├── physics/
│   ├── wavefunctions.py     # ψ(r,θ,φ), radial R_nl, spherical harmonics
│   ├── hydrogen_solver.py   # Grid evaluation & Monte-Carlo sampling
│   ├── orbitals.py          # Orbital catalogue (1s, 2p, 3d, …)
│   └── harmonic_oscillator.py
│
├── quantum/
│   ├── circuits.py          # Aer simulator wrapper, Bell/GHZ states
│   ├── grover.py            # Grover's algorithm
│   └── vqe.py               # VQE with RY+CNOT ansatz
│
├── visualization/
│   ├── orbital_renderer.py  # PyVista orbital rendering
│   ├── probability_cloud.py # Point-cloud generation
│   ├── bloch_sphere.py      # Bloch sphere visualisation
│   └── viewer_3d.py         # Generic 3-D viewer + Plotly fallback
│
├── backend/
│   └── api.py               # FastAPI endpoints
│
└── ui/
    ├── cli.py               # Interactive CLI
    └── web_interface.py     # Self-contained HTML dashboard
```

---

## Quick Start

### 1. Install dependencies

```bash
cd QuantumLab
pip install -r requirements.txt
```

### 2. Run the interactive CLI

```bash
python main.py
```

You will see a menu:

```
  ⚛  Quantum Simulation Lab  ⚛

  Select a simulation:

    1 │ Hydrogen atom orbitals
    2 │ Harmonic oscillator
    3 │ Grover's algorithm
    4 │ VQE (Variational Quantum Eigensolver)
    5 │ Bloch sphere
    6 │ Start API server
    0 │ Exit
```

### 3. Start the REST API

```bash
python main.py --server
# → http://127.0.0.1:8000
# → Swagger docs at http://127.0.0.1:8000/docs
```

---

## Physics Background

### Hydrogen Atom

The wavefunction of the hydrogen atom is:

```
ψₙₗₘ(r, θ, φ) = Rₙₗ(r) · Yₗᵐ(θ, φ)
```

where:

- **Rₙₗ(r)** — radial part, built from associated Laguerre polynomials:

  ```
  Rₙₗ(r) = √[(2/na₀)³ · (n-l-1)! / 2n·(n+l)!] · e^(-ρ/2) · ρˡ · L^{2l+1}_{n-l-1}(ρ)
  ```

  with ρ = 2r / (na₀)

- **Yₗᵐ(θ, φ)** — spherical harmonics (Condon–Shortley convention)

- **Energy levels**: Eₙ = −13.6 / n²  eV

### Supported Orbitals

`1s`, `2s`, `2p`, `3s`, `3p`, `3d`, and all magnetic sub-levels (m = −l … +l).

### Visualisation Method

Electron probability clouds are generated via **Monte-Carlo rejection sampling**:
1. Uniformly sample candidate points in a cube
2. Evaluate |ψ|² at each point
3. Accept with probability proportional to |ψ|² / max(|ψ|²)
4. Render accepted points as a 3-D scatter using PyVista

Colour can be mapped to **probability density** (intensity) or **wavefunction phase** (sign / complex angle).

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Health check |
| `GET`  | `/orbitals` | List available orbitals |
| `POST` | `/simulate/orbital` | Sample a hydrogen orbital |
| `POST` | `/simulate/harmonic` | Sample a harmonic oscillator state |
| `POST` | `/simulate/grover` | Run Grover's algorithm |
| `POST` | `/simulate/vqe` | Run VQE optimisation |
| `GET`  | `/` | Web dashboard |

Full interactive docs are available at `/docs` (Swagger UI).

---

## Quantum Algorithms

### Grover's Algorithm

Searches for a target bitstring in an unsorted space of N = 2ⁿ elements using O(√N) queries.

```python
from quantum.grover import GroverSearch

g = GroverSearch()
counts = g.run(n_qubits=3, target="101", shots=1024)
# → {'101': ~900, ...}
```

### VQE

Finds the ground-state energy of a Z⊗Z Hamiltonian using a parameterised RY + CNOT ansatz
optimised with COBYLA.

```python
from quantum.vqe import VQESolver

solver = VQESolver(n_qubits=2, depth=2, shots=4096)
result = solver.optimize(maxiter=200)
print(result["optimal_energy"])  # → ≈ −1.0
```

---

## Requirements

- Python ≥ 3.10
- numpy, scipy, qiskit, qiskit-aer, pyvista, plotly, fastapi, uvicorn

---

## License

MIT
# QuantumLab
