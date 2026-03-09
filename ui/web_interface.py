"""
Minimal web interface for the Quantum Simulation Lab.

Returns a self-contained HTML page with embedded JavaScript that calls
the FastAPI endpoints and renders results.
"""

from __future__ import annotations


def get_html() -> str:
    """Return the full HTML string for the web UI."""
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>QuantumLab — Integrated Quantum Simulation Lab</title>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #c9d1d9; --accent: #58a6ff; --accent2: #7ee787;
    --font: 'Segoe UI', system-ui, -apple-system, sans-serif;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg); color: var(--text); font-family: var(--font);
    min-height: 100vh; display: flex; flex-direction: column; align-items: center;
    padding: 2rem 1rem;
  }
  h1 { color: var(--accent); font-size: 2rem; margin-bottom: .5rem; }
  .subtitle { color: #8b949e; margin-bottom: 2rem; }
  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.5rem; width: 100%; max-width: 700px;
    margin-bottom: 1.5rem;
  }
  .card h2 { color: var(--accent2); font-size: 1.2rem; margin-bottom: 1rem; }
  label { display: block; color: #8b949e; font-size: .9rem; margin-bottom: .3rem; }
  input, select {
    width: 100%; padding: .6rem .8rem; border-radius: 8px;
    border: 1px solid var(--border); background: var(--bg); color: var(--text);
    font-size: 1rem; margin-bottom: 1rem; outline: none;
  }
  input:focus, select:focus { border-color: var(--accent); }
  button {
    background: var(--accent); color: var(--bg); border: none;
    padding: .7rem 1.5rem; border-radius: 8px; font-size: 1rem;
    cursor: pointer; font-weight: 600; transition: opacity .2s;
  }
  button:hover { opacity: .85; }
  button:disabled { opacity: .4; cursor: default; }
  .result {
    background: var(--bg); border: 1px solid var(--border); border-radius: 8px;
    padding: 1rem; margin-top: 1rem; white-space: pre-wrap; font-family: monospace;
    font-size: .85rem; max-height: 400px; overflow-y: auto;
  }
  .bar-chart { margin-top: 1rem; }
  .bar-row { display: flex; align-items: center; margin: .3rem 0; }
  .bar-label { width: 80px; text-align: right; padding-right: .5rem; font-family: monospace; }
  .bar-fill {
    height: 22px; background: linear-gradient(90deg, var(--accent), var(--accent2));
    border-radius: 4px; transition: width .5s ease;
  }
  .bar-val { margin-left: .5rem; font-size: .85rem; color: #8b949e; }
</style>
</head>
<body>
<h1>⚛ QuantumLab</h1>
<p class="subtitle">Integrated Quantum Simulation Lab</p>

<!-- Orbital card -->
<div class="card">
  <h2>Hydrogen Orbital Simulation</h2>
  <label for="orbital-name">Orbital</label>
  <select id="orbital-name">
    <option>1s</option><option>2s</option><option>2p0</option><option>2p+1</option><option>2p-1</option>
    <option>3s</option><option>3p0</option><option>3d0</option><option>3d+1</option><option>3d+2</option>
  </select>
  <label for="orbital-pts">Sample points</label>
  <input id="orbital-pts" type="number" value="10000" min="1000" max="200000" />
  <button onclick="runOrbital()">Run simulation</button>
  <div id="orbital-result" class="result" style="display:none"></div>
</div>

<!-- Grover card -->
<div class="card">
  <h2>Grover's Algorithm</h2>
  <label for="grover-qubits">Qubits</label>
  <input id="grover-qubits" type="number" value="3" min="1" max="8" />
  <label for="grover-target">Target bitstring</label>
  <input id="grover-target" type="text" value="101" />
  <button onclick="runGrover()">Run Grover</button>
  <div id="grover-result" class="result" style="display:none"></div>
  <div id="grover-chart" class="bar-chart"></div>
</div>

<!-- VQE card -->
<div class="card">
  <h2>VQE Optimisation</h2>
  <label for="vqe-qubits">Qubits</label>
  <input id="vqe-qubits" type="number" value="2" min="2" max="6" />
  <label for="vqe-depth">Ansatz depth</label>
  <input id="vqe-depth" type="number" value="2" min="1" max="8" />
  <label for="vqe-iter">Max iterations</label>
  <input id="vqe-iter" type="number" value="100" min="10" max="1000" />
  <button onclick="runVQE()">Run VQE</button>
  <div id="vqe-result" class="result" style="display:none"></div>
</div>

<script>
async function post(url, body) {
  const res = await fetch(url, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  });
  return res.json();
}

async function runOrbital() {
  const el = document.getElementById('orbital-result');
  el.style.display = 'block'; el.textContent = 'Running…';
  const data = await post('/simulate/orbital', {
    orbital: document.getElementById('orbital-name').value,
    n_points: parseInt(document.getElementById('orbital-pts').value)
  });
  el.textContent = `Orbital: ${data.orbital}  (n=${data.n}, l=${data.l}, m=${data.m})\n`
    + `Energy: ${data.energy_eV.toFixed(4)} eV\n`
    + `Points sampled: ${data.n_points.toLocaleString()}\n\n`
    + `(Point cloud data available via API response)`;
}

async function runGrover() {
  const el = document.getElementById('grover-result');
  const chart = document.getElementById('grover-chart');
  el.style.display = 'block'; el.textContent = 'Running…'; chart.innerHTML = '';
  const data = await post('/simulate/grover', {
    n_qubits: parseInt(document.getElementById('grover-qubits').value),
    target: document.getElementById('grover-target').value,
    shots: 2048
  });
  el.textContent = `Target: |${data.target}⟩   Found: |${data.found}⟩\n`;
  const maxP = Math.max(...Object.values(data.probabilities));
  chart.innerHTML = Object.entries(data.probabilities)
    .sort((a,b) => b[1]-a[1])
    .map(([s, p]) =>
      `<div class="bar-row">
        <span class="bar-label">|${s}⟩</span>
        <div class="bar-fill" style="width:${(p/maxP*300).toFixed(0)}px"></div>
        <span class="bar-val">${(100*p).toFixed(1)}%</span>
      </div>`
    ).join('');
}

async function runVQE() {
  const el = document.getElementById('vqe-result');
  el.style.display = 'block'; el.textContent = 'Running (this may take a moment)…';
  const data = await post('/simulate/vqe', {
    n_qubits: parseInt(document.getElementById('vqe-qubits').value),
    depth: parseInt(document.getElementById('vqe-depth').value),
    maxiter: parseInt(document.getElementById('vqe-iter').value)
  });
  el.textContent =
    `Optimal energy:     ${data.optimal_energy.toFixed(6)}\n`
    + `Exact ground state: -1.000000  (Z⊗Z Hamiltonian)\n`
    + `Error:              ${Math.abs(data.optimal_energy - (-1)).toFixed(6)}\n`
    + `Function evals:     ${data.n_function_evals}`;
}
</script>
</body>
</html>"""
