"""
Interactive command-line interface for the Quantum Simulation Lab (v2).
Refactored to use the Service Layer.
"""

from __future__ import annotations
import sys
import multiprocessing as mp
import numpy as np

from core import OrbitalParams, GroverParams, VQEParams
from services import PhysicsService, QuantumService


def _banner():
    print()
    print("=" * 60)
    print("  ⚛  Quantum Simulation Lab (v2)  ⚛")
    print("=" * 60)
    print()


def _menu():
    print("  Select a simulation:\n")
    print("    1 │ Hydrogen atom orbitals")
    print("    2 │ Harmonic oscillator")
    print("    3 │ Grover's algorithm")
    print("    4 │ VQE (Variational Quantum Eigensolver)")
    print("    5 │ Bloch sphere")
    print("    6 │ Start API server")
    print("    0 │ Exit")
    print()


class CLInterface:
    def __init__(self):
        self.physics = PhysicsService()
        self.quantum = QuantumService()

    def run_hydrogen(self):
        from physics.orbitals import list_orbitals
        from visualization.orbital_renderer import render_orbital

        print("\n  Available orbitals:", ", ".join(list_orbitals()))
        name = input("  Enter orbital name (e.g. 2p0): ").strip()
        n_pts = input("  Number of sample points [100000]: ").strip()
        n_pts = int(n_pts) if n_pts else 100_000
        seed = input("  Seed (optional): ").strip()
        seed = int(seed) if seed else None

        print(f"\n  ➤ Running simulation for {name} ({n_pts:,} points)...")
        
        try:
            params = OrbitalParams(name=name, n_points=n_pts, seed=seed)
            result = self.physics.run_orbital_simulation(params)
            
            print(f"  ➤ Energy level: {result.energy_ev:.4f} eV\n")

            color_choice = input("  Colour by (d)ensity or (p)hase? [d]: ").strip().lower()
            # If phase is needed, we'd call compute_phase, which we should move to service too
            # For now, density is the default value in SimulationResult.values
            
            p = mp.Process(
                target=render_orbital,
                args=(np.array(result.points), np.array(result.values)),
                kwargs={"title": f"{result.title} (Seed: {seed})"},
            )
            p.start()
            p.join()
        except Exception as e:
            print(f"  ✗ Error: {e}")

    def run_harmonic(self):
        # TODO: Move to service layer as well
        from physics.harmonic_oscillator import sample_harmonic_3d, harmonic_energy
        from visualization.viewer_3d import quick_plot

        print("\n  3-D isotropic harmonic oscillator")
        nx = int(input("  nx [0]: ").strip() or "0")
        ny = int(input("  ny [0]: ").strip() or "0")
        nz = int(input("  nz [0]: ").strip() or "0")
        n_pts = int(input("  Number of sample points [100000]: ").strip() or "100000")

        energy = harmonic_energy(nx) + harmonic_energy(ny) + harmonic_energy(nz)
        print(f"\n  ➤ E = {energy:.2f} ℏω")

        points, values = sample_harmonic_3d(nx, ny, nz, n_pts)
        p = mp.Process(
            target=quick_plot,
            args=(points,),
            kwargs={
                "scalars": values,
                "title": f"Harmonic Oscillator ({nx},{ny},{nz})",
                "cmap": "plasma",
            },
        )
        p.start()
        p.join()

    def run_grover(self):
        n_qubits = int(input("\n  Number of qubits [3]: ").strip() or "3")
        target = input(f"  Target bitstring ({n_qubits} bits) [{'1' * n_qubits}]: ").strip()
        if not target:
            target = "1" * n_qubits
        shots = int(input("  Shots [1024]: ").strip() or "1024")
        seed = input("  Seed (optional): ").strip()
        seed = int(seed) if seed else None

        params = GroverParams(n_qubits=n_qubits, target=target, shots=shots, seed=seed)
        print(f"\n  ➤ Running Grover's algorithm for target |{target}⟩...")
        
        result = self.quantum.run_grover(params)
        counts = result["counts"]
        total = sum(counts.values())

        print("  Results:")
        for state, c in sorted(counts.items(), key=lambda x: -x[1]):
            bar_len = int(40 * c / total)
            print(f"    |{state}⟩  {c:5d}  ({100*c/total:5.1f}%)  {'█' * bar_len}")

        print(f"\n  ➤ Most probable state: |{result['found']}⟩")

    def run_vqe(self):
        n_qubits = int(input("\n  Number of qubits [2]: ").strip() or "2")
        depth = int(input("  Ansatz depth [2]: ").strip() or "2")
        maxiter = int(input("  Max iterations [100]: ").strip() or "100")

        params = VQEParams(n_qubits=n_qubits, depth=depth, maxiter=maxiter)
        print("\n  ➤ Running VQE optimisation...")
        
        result = self.quantum.run_vqe(params)
        print(f"  ➤ Optimal energy: {result['optimal_energy']:.6f}")
        print(f"  ➤ Exact ground state (Z⊗Z): −1.0")

    def run_bloch(self):
        from visualization.bloch_sphere import render_bloch_sphere
        theta = float(input("  θ (polar) [0.785]: ").strip() or "0.785")
        phi = float(input("  φ (azimuthal) [0.0]: ").strip() or "0.0")
        p = mp.Process(
            target=render_bloch_sphere,
            kwargs={"theta": theta, "phi": phi},
        )
        p.start()
        p.join()

    def start_server(self):
        import uvicorn
        print("\n  ➤ Starting FastAPI server...")
        uvicorn.run("backend.api:app", host="127.0.0.1", port=8000, reload=False)


def run_cli():
    interface = CLInterface()
    _banner()
    actions = {
        "1": interface.run_hydrogen,
        "2": interface.run_harmonic,
        "3": interface.run_grover,
        "4": interface.run_vqe,
        "5": interface.run_bloch,
        "6": interface.start_server,
    }
    
    while True:
        _menu()
        choice = input("  ▸ ").strip()
        if choice == "0":
            break
        if action := actions.get(choice):
            try:
                action()
            except KeyboardInterrupt:
                print("\n  [Interrupted]")
            except Exception as e:
                print(f"\n  ✗ Error: {e}")
        else:
            print("  Invalid choice.")
