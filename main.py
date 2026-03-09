#!/usr/bin/env python3
"""
QuantumLab — Integrated Quantum Simulation Lab
================================================

Entry point.  Run with no arguments for the interactive CLI,
or with ``--server`` to start the FastAPI backend.

Usage
-----
    python main.py            # interactive CLI
    python main.py --server   # FastAPI server on http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="QuantumLab — Integrated Quantum Simulation Lab",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Start the FastAPI server instead of the CLI",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server bind host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server bind port (default: 8000)",
    )
    args = parser.parse_args()

    if args.server:
        import uvicorn

        print(f"\n  ⚛  QuantumLab API — http://{args.host}:{args.port}")
        print(f"  ⚛  Docs at http://{args.host}:{args.port}/docs\n")
        uvicorn.run(
            "backend.api:app",
            host=args.host,
            port=args.port,
            reload=False,
        )
    else:
        from ui.cli import run_cli

        run_cli()


if __name__ == "__main__":
    main()
