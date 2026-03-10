"""
Orbital catalogue for the hydrogen atom.

Maps human-readable names (``"1s"``, ``"2p0"``, ``"3d+1"``, …) to their
``(n, l, m)`` quantum numbers and provides convenience look-ups.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Catalogue:  name  →  (n, l, m)
# ---------------------------------------------------------------------------

ORBITAL_CATALOG: dict[str, tuple[int, int, int]] = {
    # n = 1
    "1s":    (1, 0, 0),
    # n = 2
    "2s":    (2, 0, 0),
    "2p-1":  (2, 1, -1),
    "2p0":   (2, 1, 0),
    "2p+1":  (2, 1, 1),
    "2p":    (2, 1, 0),      # alias
    # n = 3
    "3s":    (3, 0, 0),
    "3p-1":  (3, 1, -1),
    "3p0":   (3, 1, 0),
    "3p+1":  (3, 1, 1),
    "3p":    (3, 1, 0),      # alias
    "3d-2":  (3, 2, -2),
    "3d-1":  (3, 2, -1),
    "3d0":   (3, 2, 0),
    "3d+1":  (3, 2, 1),
    "3d+2":  (3, 2, 2),
    "3d":    (3, 2, 0),      # alias
    # Real orbital aliases for quantum measurement mapping
    "2px":   (2, 1, 1),
    "2py":   (2, 1, -1),
    "2pz":   (2, 1, 0),
    # n = 4
    "4s":    (4, 0, 0),
    "4p":    (4, 1, 0),
    "4d":    (4, 2, 0),
    "4f":    (4, 3, 0),
}


def list_orbitals() -> list[str]:
    """Return all available orbital names sorted by (n, l, |m|)."""
    return sorted(ORBITAL_CATALOG.keys(), key=lambda k: ORBITAL_CATALOG[k])


def get_orbital(name: str) -> tuple[int, int, int]:
    """
    Look up quantum numbers by orbital name.

    Parameters
    ----------
    name : str – e.g. ``"2p0"``, ``"3d+1"``

    Returns
    -------
    (n, l, m) : tuple[int, int, int]

    Raises
    ------
    KeyError if the orbital name is not in the catalogue.
    """
    key = name.strip().lower()
    # also accept e.g. "2P0"
    for cat_key, val in ORBITAL_CATALOG.items():
        if cat_key.lower() == key:
            return val
    raise KeyError(
        f"Unknown orbital '{name}'. "
        f"Available: {', '.join(list_orbitals())}"
    )
