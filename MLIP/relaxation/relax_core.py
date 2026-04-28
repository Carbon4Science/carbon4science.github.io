"""Shared relaxation helpers used by both the unified runner
(MLIP/relaxation/run_relaxation.py) and per-model Relax.py scripts
(MLIP/<Model>/Relax.py) that use non-default protocols.

Saves per-structure:
    <out_dir>/final/<material_id>.xyz    — final relaxed structure (extxyz)
    <out_dir>/traces/<material_id>.npz   — per-step energy/forces/max_force

Aggregate metadata is accumulated by the caller and written to a top-level JSON.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_optimizer(optimizer_name: str):
    """Return an ASE optimizer class by name."""
    import ase.optimize as aopt

    name = optimizer_name.upper()
    table = {
        "FIRE": aopt.FIRE,
        "LBFGS": aopt.LBFGS,
        "BFGS": aopt.BFGS,
        "GOQN": getattr(aopt, "GoodOldQuasiNewton", None),
    }
    cls = table.get(name)
    if cls is None:
        raise ValueError(f"Unknown / unavailable optimizer '{optimizer_name}'")
    return cls


def _make_cell_filter(name: str):
    """Return an ASE cell-filter class by name."""
    from ase.filters import FrechetCellFilter, ExpCellFilter, UnitCellFilter

    table = {
        "FrechetCellFilter": FrechetCellFilter,
        "ExpCellFilter": ExpCellFilter,
        "UnitCellFilter": UnitCellFilter,
    }
    if name not in table:
        raise ValueError(f"Unknown cell filter '{name}'")
    return table[name]


def relax_one_structure(
    atoms,
    calculator,
    *,
    optimizer: str = "FIRE",
    cell_filter: str = "FrechetCellFilter",
    fmax: float = 0.05,
    max_steps: int = 500,
    traces_dir: Optional[Path] = None,
    final_dir: Optional[Path] = None,
    material_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single relaxation with the given protocol and record per-step
    energy and forces via an observer. Returns a metadata dict.

    The input `atoms` is deep-copied so the caller's object is not mutated.
    """
    from ase.io import write as ase_write

    work_atoms = atoms.copy()
    work_atoms.calc = calculator

    CellFilter = _make_cell_filter(cell_filter)
    Optimizer = _make_optimizer(optimizer)

    filtered = CellFilter(work_atoms)

    # Per-step history: length equals number of optimizer iterations actually taken.
    energies: list[float] = []
    forces: list[np.ndarray] = []
    max_forces: list[float] = []

    def observer():
        e = float(work_atoms.get_potential_energy())
        f = np.asarray(work_atoms.get_forces()).copy()
        energies.append(e)
        forces.append(f)
        max_forces.append(float(np.linalg.norm(f, axis=1).max()))

    t0 = time.perf_counter()
    error: Optional[str] = None
    converged = False
    steps_taken = 0

    opt = None
    try:
        opt = Optimizer(filtered, logfile=None)
        opt.attach(observer, interval=1)
        opt.run(fmax=fmax, steps=max_steps)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    # Derive convergence from the final per-atom max-force rather than
    # opt.converged() because ASE 3.27+ requires passing the gradient
    # explicitly to converged(); we already track per-step max-force.
    if max_forces:
        converged = bool(max_forces[-1] <= fmax)
    if opt is not None:
        steps_taken = int(getattr(opt, "nsteps", max(len(energies) - 1, 0)))

    wall_seconds = time.perf_counter() - t0

    n_atoms = len(work_atoms)
    n_rec = len(energies)
    forces_arr = (
        np.stack(forces, axis=0)
        if n_rec > 0
        else np.zeros((0, n_atoms, 3), dtype=np.float32)
    )

    result: Dict[str, Any] = {
        "material_id": material_id,
        "n_atoms": n_atoms,
        "steps": steps_taken,
        "recorded_steps": n_rec,
        "converged": converged,
        "final_fmax": max_forces[-1] if max_forces else None,
        "final_energy": energies[-1] if energies else None,
        "wall_seconds": wall_seconds,
        "error": error,
    }

    if material_id is None:
        return result

    if final_dir is not None and error is None:
        final_dir.mkdir(parents=True, exist_ok=True)
        ase_write(final_dir / f"{material_id}.xyz", work_atoms, format="extxyz")

    if traces_dir is not None and n_rec > 0:
        traces_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            traces_dir / f"{material_id}.npz",
            energies=np.asarray(energies, dtype=np.float64),
            forces=forces_arr.astype(np.float32),
            max_force=np.asarray(max_forces, dtype=np.float32),
        )

    return result


def import_calculator(model_name: str, device: Optional[str] = None, checkpoint_path: Optional[str] = None):
    """Import a model's `_get_calculator` from MLIP/<Model>/Inference.py."""
    import importlib

    mod = importlib.import_module(f"MLIP.{model_name}.Inference")
    return mod._get_calculator(device=device, checkpoint_path=checkpoint_path)


def load_subset(xyz_path: Path) -> list:
    """Load structures from the prepared extxyz subset."""
    from ase.io import read as ase_read

    atoms_list = ase_read(str(xyz_path), index=":", format="extxyz")
    for i, at in enumerate(atoms_list):
        if "material_id" not in at.info:
            at.info["material_id"] = f"subset-{i}"
    return atoms_list
