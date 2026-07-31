#!/usr/bin/env python3
"""Run the new 8-model dynamat MD/carbon benchmark.

This is deliberately separate from ``production/run_production_md.py``.  The
older runner remains available for the LGPS RDF/MSD experiments, while this
runner uses the 17 structures in the Matbench Discovery dynamat reference H5
file and does not compute RDF/MSD accuracy.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MLIP.dynamat_metrics import MODEL_NAMES, get_metrics
from MLIP.nvtnosehoover import NVTNoseHoover


MODEL_MODULES = {
    "eSEN": "MLIP.eSEN.Inference",
    "ORB": "MLIP.ORB.Inference",
    "DPA4": "MLIP.DPA4.Inference",
    "NequIP": "MLIP.NequIP.Inference",
    "MACE": "MLIP.MACE.Inference",
    "SevenNet": "MLIP.SevenNet.Inference",
    "Nequix": "MLIP.Nequix.Inference",
    "CHGNet": "MLIP.CHGNet.Inference",
}

DEFAULT_H5 = ROOT / "MLIP/dynamat_trajectory/md_2026-06-29-dynamat-v1.0-reference-trajectories.h5"
DEFAULT_OUTPUT = ROOT / "MLIP/dynamat_results"
DEFAULT_INITIAL_STRUCTURES = ROOT / "MLIP/dynamat_initial_structures"


def _temperature_from_name(name):
    # Some dynamat names use a hyphen after the temperature, e.g.
    # ``bulkCuAu_500K-Artrith_VASP``.
    match = re.search(r"_(\d+)K(?:_|-|$)", name)
    if not match:
        raise ValueError(f"Could not infer temperature from dynamat structure name: {name}")
    return int(match.group(1))


def list_structures(h5_path):
    import h5py

    with h5py.File(h5_path, "r") as handle:
        return list(handle.keys())


def load_first_frame(handle, name):
    group = handle[name]
    atomic_numbers = np.asarray(group["atomic_numbers"][:], dtype=int)
    return Atoms(
        numbers=atomic_numbers,
        positions=np.asarray(group["positions"][0], dtype=float),
        cell=np.asarray(group["cell"][0], dtype=float),
        pbc=np.asarray(group["pbc"][:], dtype=bool),
    )


def list_cif_structures(structures_dir):
    return sorted(Path(structures_dir).glob("*.cif"))


def save_initial_structure(atoms, name, output_dir):
    """Save the exact HDF5 first frame separately from MD trajectories."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.extxyz"
    if not path.exists():
        write(str(path), atoms, format="extxyz")
    return path


def _carbon_tracker(model, structure, seed_index):
    from MLIP.benchmarks.carbon_tracker import CarbonTracker

    return CarbonTracker(
        project_name=f"{model}_dynamat_{structure}_seed{seed_index}",
        model_name=model,
        task="inference",
        save_results=False,
    )


def run_structure(atoms_template, model, calculator, structure, output_dir,
                  temperature, timestep_fs, equilibration_ps, production_ps,
                  traj_interval, seeds, track_carbon):
    structure_dir = output_dir / model / structure
    structure_dir.mkdir(parents=True, exist_ok=True)
    equil_steps = int(equilibration_ps * 1000 / timestep_fs)
    production_steps = int(production_ps * 1000 / timestep_fs)
    seed_results = []

    for seed_index, seed in enumerate(seeds, start=1):
        atoms = atoms_template.copy()
        atoms.calc = calculator
        rng = np.random.RandomState(seed)
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=rng)
        md = NVTNoseHoover(
            atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=temperature,
            nose_frequency=None,
        )

        start_equil = time.perf_counter()
        md.run(equil_steps)
        equil_seconds = time.perf_counter() - start_equil

        traj_path = structure_dir / f"seed-{seed}.traj"
        trajectory = Trajectory(str(traj_path), "w", atoms)
        md.attach(trajectory.write, interval=traj_interval)
        trajectory.write(atoms)

        tracker = _carbon_tracker(model, structure, seed_index) if track_carbon else None
        if tracker:
            tracker.start()
        start_production = time.perf_counter()
        md.run(production_steps)
        production_seconds = time.perf_counter() - start_production
        if tracker:
            tracker.stop()
            carbon = tracker.get_metrics()
        else:
            carbon = {}
        trajectory.close()

        seed_results.append({
            "seed": seed,
            "equilibration_seconds": round(equil_seconds, 6),
            "production_seconds": round(production_seconds, 6),
            "production_steps": production_steps,
            "trajectory": os.path.relpath(traj_path, ROOT),
            "carbon": carbon,
        })

    numeric_carbon = {}
    if seed_results and seed_results[0]["carbon"]:
        keys = [k for k, v in seed_results[0]["carbon"].items()
                if isinstance(v, (int, float))]
        for key in keys:
            values = [row["carbon"].get(key, 0.0) for row in seed_results]
            numeric_carbon[key] = round(float(np.mean(values)), 6)
            numeric_carbon[f"{key}_std"] = round(float(np.std(values)), 6)

    result = {
        "model": model,
        "structure": structure,
        "temperature_K": temperature,
        "timestep_fs": timestep_fs,
        "equilibration_ps": equilibration_ps,
        "production_ps": production_ps,
        "seeds": list(seeds),
        "seed_results": seed_results,
        "carbon_mean_over_seeds": numeric_carbon,
    }
    with (structure_dir / "result.json").open("w") as handle:
        json.dump(result, handle, indent=2, default=str)
    return result


def run_model(model, h5_path, output_dir, timestep_fs=2.0,
              equilibration_ps=10.0, production_ps=50.0, traj_interval=50,
              seeds=(42, 43, 44), track_carbon=False, dpa4_checkpoint=None,
              structures=None, initial_structures_dir=DEFAULT_INITIAL_STRUCTURES,
              device=None):
    import importlib

    module = importlib.import_module(MODEL_MODULES[model])
    calculator = module._get_calculator(device=device, checkpoint_path=dpa4_checkpoint)
    model_dir = output_dir / model
    model_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    cif_paths = list_cif_structures(initial_structures_dir)
    if cif_paths:
        selected = [path for path in cif_paths
                    if structures is None or path.stem in structures]
        structure_items = [(path.stem, read(str(path), format="cif"))
                           for path in selected]
    else:
        import h5py

        with h5py.File(h5_path, "r") as handle:
            names = structures or list(handle.keys())
            structure_items = [(name, load_first_frame(handle, name))
                               for name in names]

    for name, atoms in structure_items:
        temperature = _temperature_from_name(name)
        print(f"[{model}] {name}: {len(atoms)} atoms, {temperature} K")
        all_results.append(run_structure(
            atoms, model, calculator, name, output_dir, temperature,
            timestep_fs, equilibration_ps, production_ps, traj_interval,
            seeds, track_carbon,
        ))

    structure_carbon = [
        item["carbon_mean_over_seeds"] for item in all_results
        if item["carbon_mean_over_seeds"]
    ]
    carbon_mean_over_structures = {}
    if structure_carbon:
        keys = [key for key, value in structure_carbon[0].items()
                if not key.endswith("_std") and isinstance(value, (int, float))]
        for key in keys:
            values = [item[key] for item in structure_carbon if key in item]
            if values:
                carbon_mean_over_structures[key] = round(float(np.mean(values)), 6)

    summary = {
        "benchmark": "Matbench Discovery dynamat",
        "model": model,
        "num_structures": len(all_results),
        "metric": get_metrics(model),
        "initial_structures_dir": os.path.relpath(initial_structures_dir, ROOT),
        "carbon_mean_over_structures": carbon_mean_over_structures,
        "md": {"timestep_fs": timestep_fs, "equilibration_ps": equilibration_ps,
               "production_ps": production_ps, "traj_interval": traj_interval,
               "seeds": list(seeds)},
        "structures": all_results,
    }
    with (model_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, default=str)
    return summary


def dry_run_model(model, h5_path, dpa4_checkpoint=None, structures=None,
                  initial_structures_dir=DEFAULT_INITIAL_STRUCTURES,
                  device="cpu", strict=False):
    """Validate model loading and planned structures without running MD."""
    import importlib

    module = importlib.import_module(MODEL_MODULES[model])
    calculator_error = None
    try:
        calculator = module._get_calculator(
            device=device, checkpoint_path=dpa4_checkpoint
        )
        del calculator
    except (AssertionError, RuntimeError) as exc:
        message = str(exc)
        gpu_required = (
            "cuda" in message.lower()
            or "gpu" in message.lower()
            or (device == "cpu" and model in {"NequIP", "Nequix"})
        )
        if strict or not gpu_required:
            raise
        calculator_error = message.splitlines()[0]
        print(
            f"[dry-run] {model}: calculator load skipped because this terminal "
            f"has no usable GPU ({calculator_error})"
        )

    cif_paths = list_cif_structures(initial_structures_dir)
    if cif_paths:
        selected = [path for path in cif_paths
                    if structures is None or path.stem in structures]
        items = [(path.stem, read(str(path), format="cif")) for path in selected]
    else:
        import h5py

        with h5py.File(h5_path, "r") as handle:
            names = structures or list(handle.keys())
            items = [(name, load_first_frame(handle, name)) for name in names]

    if not items:
        raise RuntimeError(f"No dynamat structures found in {initial_structures_dir}")
    for name, atoms in items:
        temperature = _temperature_from_name(name)
        print(f"[dry-run] {model}: {name}, {len(atoms)} atoms, {temperature} K")
    if calculator_error is None:
        status = "calculator loaded"
    else:
        status = "structures validated; calculator requires GPU and was not loaded"
    print(f"[dry-run] {model}: {status}; {len(items)} structures checked; "
          "no MD, carbon tracking, or trajectory writing performed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["all", *MODEL_NAMES])
    parser.add_argument("--h5", default=str(DEFAULT_H5))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--initial-structures-dir", default=str(DEFAULT_INITIAL_STRUCTURES))
    parser.add_argument("--track-carbon", action="store_true")
    parser.add_argument("--dpa4-checkpoint", default=None)
    parser.add_argument("--structures", nargs="*", default=None)
    parser.add_argument("--timestep-fs", type=float, default=2.0)
    parser.add_argument("--equilibration-ps", type=float, default=0.0)
    parser.add_argument("--production-ps", type=float, default=10.0)
    parser.add_argument("--traj-interval", type=int, default=50)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Load the calculator and validate structures without running MD",
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default=None,
        help="Calculator device; dry-run defaults to CPU",
    )
    parser.add_argument(
        "--strict-dry-run", action="store_true",
        help="Fail dry-run if a calculator cannot be loaded (including GPU-only models)",
    )
    args = parser.parse_args()
    models = MODEL_NAMES if args.model == "all" else [args.model]
    for model in models:
        if args.dry_run:
            dry_run_model(
                model, Path(args.h5),
                dpa4_checkpoint=args.dpa4_checkpoint,
                structures=args.structures,
                initial_structures_dir=args.initial_structures_dir,
                device=args.device or "cpu",
                strict=args.strict_dry_run,
            )
        else:
            run_model(model, Path(args.h5), Path(args.output_dir),
                      timestep_fs=args.timestep_fs,
                      equilibration_ps=args.equilibration_ps,
                      production_ps=args.production_ps,
                      traj_interval=args.traj_interval, seeds=args.seeds,
                      track_carbon=args.track_carbon,
                      dpa4_checkpoint=args.dpa4_checkpoint,
                      structures=args.structures,
                      initial_structures_dir=args.initial_structures_dir,
                      device=args.device)


if __name__ == "__main__":
    main()
