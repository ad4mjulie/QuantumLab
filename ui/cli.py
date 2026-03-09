"""
Interactive command-line interface for the Quantum Simulation Lab.

Provides a menu-driven experience to run simulations and launch
3-D visualisations directly from the terminal.
"""

from __future__ import annotations

import sys


def _banner():
    print()
    print("=" * 60)
    print("  ⚛  Quantum Simulation Lab  ⚛")
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


# ------------------------------------------------------------------
# Hydrogen orbitals
# ------------------------------------------------------------------

def _run_hydrogen():
    from physics.orbitals import list_orbitals, get_orbital
    from physics.hydrogen_solver import HydrogenSolver
    from visualization.orbital_renderer import render_orbital

    print("\n  Available orbitals:", ", ".join(list_orbitals()))
    name = input("  Enter orbital name (e.g. 2p0): ").strip()
    n_pts = input("  Number of sample points [100000]: ").strip()
    n_pts = int(n_pts) if n_pts else 100_000

    n, l, m = get_orbital(name)
    solver = HydrogenSolver()

    print(f"\n  ➤ Sampling {n_pts:,} points for orbital {name}  (n={n}, l={l}, m={m})")
    print(f"  ➤ Energy level: {solver.get_energy(n):.4f} eV\n")

    points, density = solver.sample_points_mc(n, l, m, n_pts)
    phase = solver.compute_phase(n, l, m, points)

    color_choice = input("  Colour by (d)ensity or (p)hase? [d]: ").strip().lower()
    values = phase if color_choice == "p" else density
    cmap = "hsv" if color_choice == "p" else "coolwarm"

    render_orbital(
        points,
        values,
        title=f"Hydrogen Orbital  {name}   (n={n}, l={l}, m={m})",
        cmap=cmap,
    )


# ------------------------------------------------------------------
# Harmonic oscillator
# ------------------------------------------------------------------

def _run_harmonic():
    from physics.harmonic_oscillator import sample_harmonic_3d, harmonic_energy
    from visualization.viewer_3d import quick_plot

    print("\n  3-D isotropic harmonic oscillator")
    nx = int(input("  nx [0]: ").strip() or "0")
    ny = int(input("  ny [0]: ").strip() or "0")
    nz = int(input("  nz [0]: ").strip() or "0")
    n_pts = int(input("  Number of sample points [100000]: ").strip() or "100000")

    energy = harmonic_energy(nx) + harmonic_energy(ny) + harmonic_energy(nz)
    print(f"\n  ➤ E = {energy:.2f} ℏω")
    print(f"  ➤ Sampling {n_pts:,} points …\n")

    points, values = sample_harmonic_3d(nx, ny, nz, n_pts)
    quick_plot(
        points,
        scalars=values,
        title=f"Harmonic Oscillator  ({nx},{ny},{nz})   E = {energy:.1f} ℏω",
        cmap="plasma",
    )


# ------------------------------------------------------------------
# Grover
# ------------------------------------------------------------------

def _run_grover():
    from quantum.grover import GroverSearch

    n_qubits = int(input("\n  Number of qubits [3]: ").strip() or "3")
    target = input(f"  Target bitstring ({n_qubits} bits) [{'1' * n_qubits}]: ").strip()
    if not target:
        target = "1" * n_qubits
    shots = int(input("  Shots [1024]: ").strip() or "1024")

    g = GroverSearch()
    print(f"\n  ➤ Running Grover's algorithm for target |{target}⟩ …\n")
    counts = g.run(n_qubits, target, shots=shots)
    total = sum(counts.values())

    print("  Results:")
    for state, c in sorted(counts.items(), key=lambda x: -x[1]):
        bar_len = int(40 * c / total)
        print(f"    |{state}⟩  {c:5d}  ({100*c/total:5.1f}%)  {'█' * bar_len}")

    found = max(counts, key=counts.get)
    print(f"\n  ➤ Most probable state: |{found}⟩")
    success = "✓" if found == target else "✗"
    print(f"  ➤ Target found: {success}")


# ------------------------------------------------------------------
# VQE
# ------------------------------------------------------------------

def _run_vqe():
    from quantum.vqe import VQESolver

    n_qubits = int(input("\n  Number of qubits [2]: ").strip() or "2")
    depth = int(input("  Ansatz depth [2]: ").strip() or "2")
    shots = int(input("  Shots [4096]: ").strip() or "4096")
    maxiter = int(input("  Max iterations [100]: ").strip() or "100")

    solver = VQESolver(n_qubits=n_qubits, depth=depth, shots=shots)
    print("\n  ➤ Running VQE optimisation …\n")

    result = solver.optimize(maxiter=maxiter)
    print(f"  ➤ Optimal energy: {result['optimal_energy']:.6f}")
    print(f"  ➤ Function evaluations: {result['n_iterations']}")
    print(f"  ➤ Exact ground state (Z⊗Z): −1.0")
    print(f"  ➤ Error: {abs(result['optimal_energy'] - (-1.0)):.6f}")


# ------------------------------------------------------------------
# Bloch sphere
# ------------------------------------------------------------------

def _run_bloch():
    import numpy as np
    from visualization.bloch_sphere import render_bloch_sphere

    print("\n  Bloch sphere state |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩")
    theta = float(input("  θ (polar, 0 to π) [0.7854]: ").strip() or str(np.pi / 4))
    phi = float(input("  φ (azimuthal, 0 to 2π) [0.0]: ").strip() or "0.0")

    render_bloch_sphere(theta=theta, phi=phi, title=f"Bloch Sphere  θ={theta:.3f}  φ={phi:.3f}")


# ------------------------------------------------------------------
# API server
# ------------------------------------------------------------------

def _start_server():
    import uvicorn
    print("\n  ➤ Starting FastAPI server on http://127.0.0.1:8000")
    print("  ➤ Docs at http://127.0.0.1:8000/docs\n")
    uvicorn.run("backend.api:app", host="127.0.0.1", port=8000, reload=False)


# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------

ACTIONS = {
    "1": _run_hydrogen,
    "2": _run_harmonic,
    "3": _run_grover,
    "4": _run_vqe,
    "5": _run_bloch,
    "6": _start_server,
}


def run_cli():
    """Launch the interactive CLI."""
    _banner()
    while True:
        _menu()
        choice = input("  ▸ ").strip()
        if choice == "0":
            print("\n  Goodbye! 🚀\n")
            sys.exit(0)
        action = ACTIONS.get(choice)
        if action:
            try:
                action()
            except KeyboardInterrupt:
                print("\n  [Interrupted]")
            except Exception as exc:
                print(f"\n  ✗ Error: {exc}\n")
        else:
            print("  Invalid choice — try again.\n")
