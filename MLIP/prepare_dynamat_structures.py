#!/usr/bin/env python3
"""Extract the 17 Matbench Discovery dynamat starting frames to CIF files.

Run this once with an environment that has h5py installed.  The benchmark
itself can then read the CIF directory and does not need h5py.
"""

from pathlib import Path

import h5py
import numpy as np
from ase import Atoms
from ase.io import write


ROOT = Path(__file__).resolve().parent.parent
H5_PATH = ROOT / "MLIP/dynamat_trajectory/md_2026-06-29-dynamat-v1.0-reference-trajectories.h5"
OUTPUT_DIR = ROOT / "MLIP/dynamat_initial_structures"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with h5py.File(H5_PATH, "r") as handle:
        for name in handle:
            group = handle[name]
            atoms = Atoms(
                numbers=np.asarray(group["atomic_numbers"][:], dtype=int),
                positions=np.asarray(group["positions"][0], dtype=float),
                cell=np.asarray(group["cell"][0], dtype=float),
                pbc=np.asarray(group["pbc"][:], dtype=bool),
            )
            path = OUTPUT_DIR / f"{name}.cif"
            write(str(path), atoms, format="cif")
            print(f"wrote {path.name} ({len(atoms)} atoms)")


if __name__ == "__main__":
    main()
