#!/usr/bin/env python3
"""Production MD simulation runner for MLIP models.

Runs long NVT MD simulations (equilibration + production) and computes
RDF and MSD from the production trajectory.

Usage:
    python MLIP/production/run_production_md.py \
        --model CHGNet \
        --config MLIP/production/configs/LGPS_600K.json \
        [--structure_index 0] \
        [--skip_md] \
        [--skip_analysis]
"""

import argparse
import importlib
import json
import os
import sys
import time

import numpy as np
from ase import units
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

# Ensure project root is on path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from MLIP.nvtnosehoover import NVTNoseHoover
from MLIP.production.analysis import (
    compute_rdf,
    compute_msd,
    compute_rdf_mae,
    compute_msd_mae,
    compute_diffusivity,
)


def load_config(config_path):
    """Load JSON configuration file."""
    with open(config_path) as f:
        return json.load(f)


def get_calculator(model_name):
    """Import and return ASE calculator from model's Inference.py."""
    mod = importlib.import_module(f"MLIP.{model_name}.Inference")
    return mod._get_calculator()


def resolve_rdf_pairs(struct_cfg, atoms):
    """Resolve RDF pairs from config.

    'auto' generates all pairs involving diffusing_species.
    Otherwise expects a list of [species_i, species_j] pairs.
    """
    rdf_cfg = struct_cfg.get("rdf", {})
    pairs = rdf_cfg.get("pairs", "auto")

    if pairs == "auto":
        diff = struct_cfg["diffusing_species"]
        all_species = sorted(set(atoms.get_chemical_symbols()))
        return [(diff, s) for s in all_species]
    else:
        return [tuple(p) for p in pairs]


def run_md_simulation(struct_cfg, calculator=None, model_name=None, carbon_tracker=None):
    """Run equilibration + production MD, saving production trajectory.

    Uses a single NVTNoseHoover instance to avoid thermostat bath reset
    between equilibration and production phases.

    Args:
        struct_cfg: Structure configuration dict.
        calculator: Pre-built ASE calculator. If None, falls back to get_calculator(model_name).
        model_name: Model name (required if calculator is None, used for output paths).
        carbon_tracker: Optional CarbonTracker instance. If provided, starts before
                       production MD and stops after (excludes equilibration from tracking).
    """
    if calculator is None:
        if model_name is None:
            raise ValueError("Either calculator or model_name must be provided")
        calculator = get_calculator(model_name)

    label = struct_cfg["label"]
    traj_dir = os.path.join(ROOT, "MLIP", "production", "trajectories", model_name or "unknown")
    os.makedirs(traj_dir, exist_ok=True)

    prod_traj_path = os.path.join(traj_dir, f"{label}_prod.traj")

    # Load structure
    cif_path = struct_cfg["cif"]
    if not os.path.isabs(cif_path):
        cif_path = os.path.join(ROOT, cif_path)
    atoms = read(cif_path)

    # Set calculator
    atoms.calc = calculator

    # Initialize velocities
    seed = struct_cfg.get("seed", 42)
    rng = np.random.RandomState(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=struct_cfg["temperature_K"], rng=rng)

    # MD parameters
    timestep = struct_cfg["timestep_fs"] * units.fs
    nose_freq = struct_cfg.get("nose_frequency")
    traj_interval = struct_cfg.get("traj_interval", 50)

    equil_steps = int(struct_cfg["equilibration_ps"] * 1000 / struct_cfg["timestep_fs"])
    prod_steps = int(struct_cfg["production_ps"] * 1000 / struct_cfg["timestep_fs"])

    # Single MD instance (avoids thermostat bath reset at equil/prod boundary)
    md = NVTNoseHoover(
        atoms,
        timestep=timestep,
        temperature_K=struct_cfg["temperature_K"],
        nose_frequency=nose_freq,
    )

    # --- Equilibration ---
    print(f"Equilibration: {equil_steps} steps ({struct_cfg['equilibration_ps']} ps)")
    t0 = time.perf_counter()
    md.run(equil_steps)
    equil_time = time.perf_counter() - t0
    print(f"  Done in {equil_time:.1f}s")

    # --- Production ---
    # Attach trajectory writer for production phase
    traj = Trajectory(prod_traj_path, "w", atoms)
    md.attach(traj.write, interval=traj_interval)

    # Write the first production frame (t=0 of production)
    traj.write(atoms)

    # Start carbon tracking for production phase only (excludes equilibration)
    if carbon_tracker is not None:
        carbon_tracker.start()

    print(f"Production: {prod_steps} steps ({struct_cfg['production_ps']} ps)")
    t0 = time.perf_counter()
    md.run(prod_steps)
    prod_time = time.perf_counter() - t0
    print(f"  Done in {prod_time:.1f}s")

    # Stop carbon tracking after production
    if carbon_tracker is not None:
        carbon_tracker.stop()

    traj.close()
    print(f"  Trajectory: {prod_traj_path}")

    # Expected frame count: 1 (initial) + prod_steps // traj_interval
    expected_frames = 1 + prod_steps // traj_interval
    print(f"  Expected frames: {expected_frames}")

    return {
        "equil_steps": equil_steps,
        "equil_seconds": round(equil_time, 2),
        "prod_steps": prod_steps,
        "prod_seconds": round(prod_time, 2),
        "traj_path": prod_traj_path,
    }


def run_analysis(model_name, struct_cfg, timing=None):
    """Load production trajectory and compute RDF + MSD."""
    label = struct_cfg["label"]
    traj_dir = os.path.join(ROOT, "MLIP", "production", "trajectories", model_name)
    result_dir = os.path.join(ROOT, "MLIP", "production", "results", model_name)
    os.makedirs(result_dir, exist_ok=True)

    prod_traj_path = os.path.join(traj_dir, f"{label}_prod.traj")

    if not os.path.exists(prod_traj_path):
        print(f"ERROR: Trajectory not found: {prod_traj_path}")
        sys.exit(1)

    # Load trajectory
    print(f"Loading trajectory: {prod_traj_path}")
    traj = Trajectory(prod_traj_path)
    images = list(traj)
    print(f"  Loaded {len(images)} frames")

    traj_interval = struct_cfg.get("traj_interval", 50)
    dt_per_frame_ps = struct_cfg["timestep_fs"] * traj_interval / 1000.0

    # --- RDF ---
    rdf_cfg = struct_cfg.get("rdf", {})
    rdf_window_ps = rdf_cfg.get("analysis_window_ps")
    rdf_images = images
    if rdf_window_ps is not None:
        n_window = int(rdf_window_ps / dt_per_frame_ps)
        rdf_images = images[-n_window:]
        print(f"  RDF analysis window: last {rdf_window_ps} ps ({len(rdf_images)} frames)")

    pairs = resolve_rdf_pairs(struct_cfg, images[0])
    rmax = rdf_cfg.get("rmax", 4.0)
    binwidth = rdf_cfg.get("binwidth", 0.05)

    for sp_i, sp_j in pairs:
        print(f"  Computing RDF: {sp_i}-{sp_j} ...")
        r, g_r, n_r = compute_rdf(rdf_images, sp_i, sp_j, rmax=rmax, binwidth=binwidth)

        csv_path = os.path.join(result_dir, f"{label}_rdf_{sp_i}-{sp_j}.csv")
        np.savetxt(
            csv_path,
            np.column_stack([r, g_r, n_r]),
            delimiter=",",
            header="r_angstrom,g_r,n_r",
            comments="",
        )
        print(f"    Saved: {csv_path}")

    # --- MSD ---
    msd_cfg = struct_cfg.get("msd", {})
    msd_window_ps = msd_cfg.get("analysis_window_ps")
    msd_images = images
    if msd_window_ps is not None:
        n_window = int(msd_window_ps / dt_per_frame_ps)
        msd_images = images[-n_window:]
        print(f"  MSD analysis window: last {msd_window_ps} ps ({len(msd_images)} frames)")

    diff_species = struct_cfg["diffusing_species"]
    print(f"  Computing MSD for {diff_species} ...")
    dt_fs, msd, msd_xyz = compute_msd(
        msd_images,
        diff_species,
        struct_cfg["timestep_fs"],
        traj_interval,
    )

    csv_path = os.path.join(result_dir, f"{label}_msd.csv")
    np.savetxt(
        csv_path,
        np.column_stack([dt_fs, msd, msd_xyz]),
        delimiter=",",
        header="t_fs,msd_angstrom2,msd_x,msd_y,msd_z",
        comments="",
    )
    print(f"    Saved: {csv_path}")

    # --- Accuracy metrics ---
    accuracy = compute_accuracy_metrics(model_name, struct_cfg)

    # --- Save results JSON ---
    results = {
        "model": model_name,
        "structure": struct_cfg["label"],
        "temperature_K": struct_cfg["temperature_K"],
        "equilibration_ps": struct_cfg["equilibration_ps"],
        "production_ps": struct_cfg["production_ps"],
        "timestep_fs": struct_cfg["timestep_fs"],
        "traj_interval": traj_interval,
        "num_frames": len(images),
    }
    if accuracy:
        results["accuracy"] = accuracy
    if timing:
        results["timing"] = timing

    json_path = os.path.join(result_dir, f"{label}_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results JSON: {json_path}")

    return results


def compute_accuracy_metrics(model_name, struct_cfg):
    """Compare MLIP results against AIMD ground truth.

    Returns accuracy metrics dict, or empty dict if ground truth not found.
    """
    label = struct_cfg["label"]
    result_dir = os.path.join(ROOT, "MLIP", "production", "results", model_name)
    gt_dir = os.path.join(ROOT, "MLIP", "production", "ground_truth")

    if not os.path.isdir(gt_dir):
        print("  No ground_truth directory found, skipping accuracy metrics.")
        return {}

    accuracy = {}

    # --- RDF MAE per pair ---
    rdf_maes = {}
    rdf_scores = {}
    # Detect pairs from MLIP result files
    import glob as globmod
    rdf_pattern = os.path.join(result_dir, f"{label}_rdf_*.csv")
    rdf_files = sorted(globmod.glob(rdf_pattern))

    for rdf_file in rdf_files:
        fname = os.path.basename(rdf_file)
        pair_str = fname.replace(f"{label}_rdf_", "").replace(".csv", "")

        gt_path = os.path.join(gt_dir, fname)
        if not os.path.exists(gt_path):
            print(f"  No ground truth for {pair_str}, skipping RDF MAE.")
            continue

        pred = np.genfromtxt(rdf_file, delimiter=",", names=True)
        gt = np.genfromtxt(gt_path, delimiter=",", names=True)

        mae = compute_rdf_mae(
            pred["r_angstrom"], pred["g_r"],
            gt["r_angstrom"], gt["g_r"],
        )
        if mae is not None:
            rdf_maes[pair_str] = round(mae, 6)
            rdf_scores[pair_str] = round(1 / (1 + mae), 6)  # Simple score: higher is better
            print(f"    RDF MAE ({pair_str}): {mae:.6f}")

    if rdf_maes:
        rdf_maes["average"] = round(
            sum(rdf_maes.values()) / len(rdf_maes), 6
        )
        rdf_scores["average"] = round(
            sum(rdf_scores.values()) / len(rdf_scores), 6
        )
        accuracy["rdf_mae"] = rdf_maes
        accuracy["rdf_score"] = rdf_scores

    # --- MSD MAE ---
    msd_pred_path = os.path.join(result_dir, f"{label}_msd.csv")
    msd_gt_path = os.path.join(gt_dir, f"{label}_msd.csv")

    if os.path.exists(msd_pred_path) and os.path.exists(msd_gt_path):
        pred_msd = np.genfromtxt(msd_pred_path, delimiter=",", names=True)
        gt_msd = np.genfromtxt(msd_gt_path, delimiter=",", names=True)

        mae = compute_msd_mae(
            pred_msd["t_fs"], pred_msd["msd_angstrom2"],
            gt_msd["t_fs"], gt_msd["msd_angstrom2"],
        )
        if mae is not None:
            accuracy["msd_mae"] = round(mae, 6)
            accuracy["msd_score"] = round(1 / (1 + mae), 6)  # Simple score: higher is better
            print(f"    MSD MAE: {mae:.6f}")

        # Diffusivity comparison
        d_mlip = compute_diffusivity(pred_msd["t_fs"], pred_msd["msd_angstrom2"])
        d_aimd = compute_diffusivity(gt_msd["t_fs"], gt_msd["msd_angstrom2"])

        if d_mlip is not None:
            accuracy["diffusivity_mlip_cm2s"] = d_mlip
            print(f"    Diffusivity (MLIP): {d_mlip:.4e} cm^2/s")
        if d_aimd is not None:
            accuracy["diffusivity_aimd_cm2s"] = d_aimd
            print(f"    Diffusivity (AIMD): {d_aimd:.4e} cm^2/s")
        if d_mlip is not None and d_aimd is not None and d_aimd != 0:
            rel_err = abs(d_mlip - d_aimd) / abs(d_aimd)
            accuracy["diffusivity_relative_error"] = round(rel_err, 6)
            print(f"    Diffusivity relative error: {rel_err:.4f}")
    else:
        if not os.path.exists(msd_gt_path):
            print("  No ground truth MSD found, skipping MSD accuracy metrics.")

    return accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Run production MD + analysis for MLIP models"
    )
    parser.add_argument("--model", required=True, help="Model name (e.g., CHGNet, MACE)")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    parser.add_argument(
        "--structure_index",
        type=int,
        default=0,
        help="Index of structure in config (default: 0)",
    )
    parser.add_argument("--skip_md", action="store_true", help="Skip MD, re-run analysis only")
    parser.add_argument(
        "--skip_analysis", action="store_true", help="Run MD only, skip analysis"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    struct_cfg = config["structures"][args.structure_index]

    print(f"Model: {args.model}")
    print(f"Structure: {struct_cfg['label']} ({struct_cfg['cif']})")
    print(f"Temperature: {struct_cfg['temperature_K']} K")
    print(f"Equilibration: {struct_cfg['equilibration_ps']} ps")
    print(f"Production: {struct_cfg['production_ps']} ps")
    print(f"Traj interval: {struct_cfg.get('traj_interval', 50)} steps")
    print()

    md_info = None
    if not args.skip_md:
        md_info = run_md_simulation(struct_cfg, model_name=args.model)
        print()

    timing = None
    if md_info:
        timing = {
            "equil_seconds": md_info["equil_seconds"],
            "prod_seconds": md_info["prod_seconds"],
        }

    if not args.skip_analysis:
        run_analysis(args.model, struct_cfg, timing=timing)


if __name__ == "__main__":
    main()
