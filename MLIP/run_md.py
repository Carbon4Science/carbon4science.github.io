"""
Shared MD simulation helper for MLIP benchmarking.

Runs NVT molecular dynamics using a custom Nose-Hoover thermostat on a given
structure with a provided ASE calculator. Returns speed metrics, structure info,
and MD parameters for the results JSON.
"""

import os
import time

import numpy as np
from ase import units
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from MLIP.nvtnosehoover import NVTNoseHoover


def run_md(
    structure_path,
    calculator,
    steps=1000,
    temperature_K=300,
    timestep_fs=1.0,
    nose_frequency=None,
    seed=42,
):
    """
    Run NVT MD simulation and return speed metrics.

    Args:
        structure_path: Path to CIF structure file.
        calculator: ASE calculator instance.
        steps: Number of MD steps to run.
        temperature_K: Target temperature in Kelvin.
        timestep_fs: Timestep in femtoseconds.
        nose_frequency: Nose-Hoover coupling frequency in fs.
            If None, defaults to 40 * timestep_fs.
        seed: Random seed for velocity initialization.

    Returns:
        dict with keys:
            - steps_per_second: MD throughput
            - ns_per_day: Simulation speed in ns/day
            - total_steps: Number of steps completed
            - elapsed_seconds: Wall-clock time
            - structure: {formula, num_atoms, source_file}
            - md_params: {ensemble, thermostat, temperature_K, timestep_fs,
                          nose_frequency, seed}
    """
    atoms = read(structure_path)
    atoms.calc = calculator

    # Initialize velocities
    rng = np.random.RandomState(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng)

    # Set up custom Nose-Hoover NVT dynamics
    # NVTNoseHoover handles Stationary() and ZeroRotation() internally
    timestep = timestep_fs * units.fs
    md = NVTNoseHoover(
        atoms,
        timestep=timestep,
        temperature_K=temperature_K,
        nose_frequency=nose_frequency,
    )

    # Run and time the simulation
    start = time.perf_counter()
    md.run(steps)
    elapsed = time.perf_counter() - start

    # Compute speed metrics
    steps_per_second = steps / elapsed
    # ns_per_day = (total_simulated_time_in_ns) / (wall_time_in_days)
    ns_per_day = (steps * timestep_fs * 1e-6) / (elapsed / 86400)

    # Actual nose_frequency used
    actual_nose_freq = nose_frequency if nose_frequency is not None else 40 * timestep_fs

    return {
        "steps_per_second": round(steps_per_second, 2),
        "ns_per_day": round(ns_per_day, 4),
        "steps": steps,
        "elapsed_seconds": round(elapsed, 2),
        "structure": {
            "formula": atoms.get_chemical_formula(),
            "num_atoms": len(atoms),
            "source_file": os.path.basename(structure_path),
        },
        "md_params": {
            "ensemble": "NVT",
            "thermostat": "NoseHoover",
            "temperature_K": temperature_K,
            "timestep_fs": timestep_fs,
            "nose_frequency": actual_nose_freq,
            "seed": seed,
        },
    }
