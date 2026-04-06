#!/usr/bin/env python3
"""Production MD simulation runner for MLIP models.

Runs multi-seed NVT MD simulations (equilibration + production) and computes
ensemble-averaged RDF and MSD from the production trajectories.

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


def get_calculator(model_name, checkpoint_path=None):
    """Import and return ASE calculator from model's Inference.py.

    Args:
        model_name: Model name (e.g., CHGNet, MACE).
        checkpoint_path: If provided, loads fine-tuned model from this path.
    """
    mod = importlib.import_module(f"MLIP.{model_name}.Inference")
    return mod._get_calculator(checkpoint_path=checkpoint_path)


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


def _get_seeds(struct_cfg):
    """Get list of seeds from config. Supports both 'seeds' (list) and legacy 'seed' (int)."""
    seeds = struct_cfg.get("seeds")
    if seeds is not None:
        return list(seeds)
    seed = struct_cfg.get("seed", 42)
    return [seed]


def _import_carbon_tracker():
    """Import CarbonTracker from benchmarks directory."""
    benchmarks_dir = os.path.join(ROOT, "MLIP", "benchmarks")
    if benchmarks_dir not in sys.path:
        sys.path.insert(0, benchmarks_dir)
    from carbon_tracker import CarbonTracker
    return CarbonTracker


def run_md_simulation(struct_cfg, calculator=None, model_name=None, track_carbon=False,
                      variant="pretrained", checkpoint_path=None):
    """Run multi-seed equilibration + production MD, saving per-seed production trajectories.

    For each seed: equilibrate (no tracking) → start carbon tracker → produce → stop tracker.
    Per-seed carbon metrics are averaged and returned in the result dict.

    Per-seed output files use sequential index naming:
        {label}_prod-1.traj, {label}_prod-2.traj, ...

    Args:
        struct_cfg: Structure configuration dict.
        calculator: Pre-built ASE calculator. If None, falls back to get_calculator(model_name).
        model_name: Model name (required if calculator is None, used for output paths).
        track_carbon: If True, enables per-seed carbon tracking.
        variant: 'pretrained' or 'finetuned', determines output directory.
        checkpoint_path: Path to fine-tuned checkpoint (used when variant='finetuned').

    Returns:
        Dict with timing info, per-seed trajectory paths, and averaged carbon_metrics.
    """
    if calculator is None:
        if model_name is None:
            raise ValueError("Either calculator or model_name must be provided")
        calculator = get_calculator(model_name, checkpoint_path=checkpoint_path)

    label = struct_cfg["label"]
    traj_dir = os.path.join(ROOT, "MLIP", "production", "trajectories", variant, model_name or "unknown")
    os.makedirs(traj_dir, exist_ok=True)

    # Load structure template
    cif_path = struct_cfg["cif"]
    if not os.path.isabs(cif_path):
        cif_path = os.path.join(ROOT, cif_path)

    # MD parameters
    timestep_fs = struct_cfg["timestep_fs"]
    timestep = timestep_fs * units.fs
    nose_freq = struct_cfg.get("nose_frequency")
    traj_interval = struct_cfg.get("traj_interval", 50)
    equil_steps = int(struct_cfg["equilibration_ps"] * 1000 / timestep_fs)
    prod_steps = int(struct_cfg["production_ps"] * 1000 / timestep_fs)

    seeds = _get_seeds(struct_cfg)
    num_seeds = len(seeds)
    print(f"Seeds: {seeds} ({num_seeds} runs)")

    CarbonTracker = None
    if track_carbon:
        CarbonTracker = _import_carbon_tracker()

    traj_paths = []
    total_equil_time = 0.0
    total_prod_time = 0.0
    per_seed_carbon = []

    for idx, seed in enumerate(seeds, 1):
        # Load structure and initialize velocities
        atoms = read(cif_path)
        atoms.calc = calculator

        rng = np.random.RandomState(seed)
        MaxwellBoltzmannDistribution(atoms, temperature_K=struct_cfg["temperature_K"], rng=rng)

        md = NVTNoseHoover(
            atoms,
            timestep=timestep,
            temperature_K=struct_cfg["temperature_K"],
            nose_frequency=nose_freq,
        )

        # Equilibration (no carbon tracking)
        print(f"Equilibration [{idx}/{num_seeds}] (seed={seed}): {equil_steps} steps ({struct_cfg['equilibration_ps']} ps)")
        t0 = time.perf_counter()
        md.run(equil_steps)
        equil_time = time.perf_counter() - t0
        total_equil_time += equil_time
        print(f"  Done in {equil_time:.1f}s")

        # Attach trajectory writer for production
        prod_traj_path = os.path.join(traj_dir, f"{label}_prod-{idx}.traj")
        traj = Trajectory(prod_traj_path, "w", atoms)
        md.attach(traj.write, interval=traj_interval)
        traj.write(atoms)  # initial production frame (t=0)

        # Per-seed carbon tracking
        seed_tracker = None
        if track_carbon:
            seed_tracker = CarbonTracker(
                project_name=f"{model_name}_MLIP_benchmark_seed{idx}",
                model_name=model_name,
                task="inference",
                save_results=False,
            )
            seed_tracker.start()

        # Production
        print(f"Production [{idx}/{num_seeds}] (seed={seed}): {prod_steps} steps ({struct_cfg['production_ps']} ps)")
        t0 = time.perf_counter()
        md.run(prod_steps)
        prod_time = time.perf_counter() - t0
        total_prod_time += prod_time
        print(f"  Done in {prod_time:.1f}s")

        if seed_tracker is not None:
            seed_tracker.stop()
            per_seed_carbon.append(seed_tracker.get_metrics())

        traj.close()
        traj_paths.append(prod_traj_path)
        print(f"  Trajectory: {prod_traj_path}")

        expected_frames = 1 + prod_steps // traj_interval
        print(f"  Expected frames: {expected_frames}")

    # Average per-seed carbon metrics
    carbon_metrics = None
    if per_seed_carbon:
        carbon_metrics = {}
        numeric_keys = [k for k in per_seed_carbon[0] if isinstance(per_seed_carbon[0][k], (int, float))]
        for key in numeric_keys:
            values = [m[key] for m in per_seed_carbon]
            carbon_metrics[key] = round(sum(values) / len(values), 6)
        # Copy non-numeric fields from first seed
        for key in per_seed_carbon[0]:
            if key not in carbon_metrics:
                carbon_metrics[key] = per_seed_carbon[0][key]
        print(f"\n  Carbon metrics averaged over {len(per_seed_carbon)} seeds")

    result = {
        "equil_steps": equil_steps,
        "equil_seconds": round(total_equil_time / num_seeds, 2),
        "prod_steps": prod_steps,
        "prod_seconds": round(total_prod_time / num_seeds, 2),
        "num_seeds": num_seeds,
        "seeds": seeds,
        "timestep_fs": timestep_fs,
        "traj_paths": traj_paths,
    }
    if carbon_metrics is not None:
        result["carbon_metrics"] = carbon_metrics

    return result


def run_analysis(model_name, struct_cfg, timing=None, variant="pretrained"):
    """Load per-seed production trajectories, compute per-seed RDF/MSD,
    then ensemble-average and compare with ground truth.

    Stores per-trial accuracy metrics as a list ('trials') in results JSON,
    along with 'mean' and 'std' computed across trials.
    """
    label = struct_cfg["label"]
    traj_dir = os.path.join(ROOT, "MLIP", "production", "trajectories", variant, model_name)
    result_dir = os.path.join(ROOT, "MLIP", "production", "results", variant, model_name)
    os.makedirs(result_dir, exist_ok=True)

    seeds = _get_seeds(struct_cfg)
    num_seeds = len(seeds)
    traj_interval = struct_cfg.get("traj_interval", 50)
    dt_per_frame_ps = struct_cfg["timestep_fs"] * traj_interval / 1000.0

    # Discover available per-seed trajectories
    available_indices = []
    for idx in range(1, num_seeds + 1):
        traj_path = os.path.join(traj_dir, f"{label}_prod-{idx}.traj")
        if os.path.exists(traj_path):
            available_indices.append(idx)
        else:
            print(f"WARNING: Trajectory not found for seed index {idx}: {traj_path}")

    if not available_indices:
        print("ERROR: No production trajectories found.")
        sys.exit(1)

    print(f"Found {len(available_indices)}/{num_seeds} trajectories")

    # Analysis config
    rdf_cfg = struct_cfg.get("rdf", {})
    rdf_window_ps = rdf_cfg.get("analysis_window_ps")
    rmax = rdf_cfg.get("rmax", 4.0)
    binwidth = rdf_cfg.get("binwidth", 0.05)
    msd_cfg = struct_cfg.get("msd", {})
    msd_window_ps = msd_cfg.get("analysis_window_ps")
    diff_species = struct_cfg["diffusing_species"]

    # Load first trajectory to resolve RDF pairs
    first_traj_path = os.path.join(traj_dir, f"{label}_prod-{available_indices[0]}.traj")
    first_images = list(Trajectory(first_traj_path))
    pairs = resolve_rdf_pairs(struct_cfg, first_images[0])

    all_rdf = {pair: [] for pair in pairs}
    all_msd = []

    # Single pass: load each trajectory once, compute both RDF and MSD
    for idx in available_indices:
        traj_path = os.path.join(traj_dir, f"{label}_prod-{idx}.traj")
        print(f"  Loading trajectory: {traj_path}")
        if idx == available_indices[0]:
            images = first_images  # reuse already-loaded first trajectory
        else:
            images = list(Trajectory(traj_path))
        print(f"    {len(images)} frames")

        # --- Per-seed RDF ---
        rdf_images = images
        if rdf_window_ps is not None:
            n_window = int(rdf_window_ps / dt_per_frame_ps)
            rdf_images = images[-n_window:]
            print(f"    RDF analysis window: last {rdf_window_ps} ps ({len(rdf_images)} frames)")

        for sp_i, sp_j in pairs:
            print(f"    Computing RDF [{idx}]: {sp_i}-{sp_j} ...")
            r, g_r, n_r = compute_rdf(rdf_images, sp_i, sp_j, rmax=rmax, binwidth=binwidth)

            csv_path = os.path.join(result_dir, f"{label}_rdf-{idx}_{sp_i}-{sp_j}.csv")
            np.savetxt(csv_path, np.column_stack([r, g_r, n_r]),
                       delimiter=",", header="r_angstrom,g_r,n_r", comments="")
            print(f"      Saved: {csv_path}")
            all_rdf[(sp_i, sp_j)].append((r, g_r, n_r))

        # --- Per-seed MSD ---
        msd_images = images
        if msd_window_ps is not None:
            n_window = int(msd_window_ps / dt_per_frame_ps)
            msd_images = images[-n_window:]

        print(f"    Computing MSD [{idx}] for {diff_species} ...")
        dt_fs, msd, msd_xyz = compute_msd(
            msd_images, diff_species, struct_cfg["timestep_fs"], traj_interval,
        )

        csv_path = os.path.join(result_dir, f"{label}_msd-{idx}.csv")
        np.savetxt(csv_path, np.column_stack([dt_fs, msd, msd_xyz]),
                   delimiter=",", header="t_fs,msd_angstrom2,msd_x,msd_y,msd_z", comments="")
        print(f"      Saved: {csv_path}")
        all_msd.append((dt_fs, msd, msd_xyz))

    # --- Ensemble-average RDF ---
    for sp_i, sp_j in pairs:
        seed_data = all_rdf[(sp_i, sp_j)]
        if len(seed_data) > 1:
            avg_g_r = np.mean([d[1] for d in seed_data], axis=0)
            avg_n_r = np.mean([d[2] for d in seed_data], axis=0)
        else:
            avg_g_r = seed_data[0][1]
            avg_n_r = seed_data[0][2]
        r = seed_data[0][0]

        csv_path = os.path.join(result_dir, f"{label}_rdf_{sp_i}-{sp_j}.csv")
        np.savetxt(csv_path, np.column_stack([r, avg_g_r, avg_n_r]),
                   delimiter=",", header="r_angstrom,g_r,n_r", comments="")
        print(f"  Ensemble-average RDF ({sp_i}-{sp_j}): {csv_path}")

    # --- Ensemble-average MSD ---
    if len(all_msd) > 1:
        avg_msd = np.mean([d[1] for d in all_msd], axis=0)
        avg_msd_xyz = np.mean([d[2] for d in all_msd], axis=0)
    else:
        avg_msd = all_msd[0][1]
        avg_msd_xyz = all_msd[0][2]
    dt_fs = all_msd[0][0]

    csv_path = os.path.join(result_dir, f"{label}_msd.csv")
    np.savetxt(csv_path, np.column_stack([dt_fs, avg_msd, avg_msd_xyz]),
               delimiter=",", header="t_fs,msd_angstrom2,msd_x,msd_y,msd_z", comments="")
    print(f"  Ensemble-average MSD: {csv_path}")

    # --- Per-seed accuracy metrics (each seed vs GT ensemble average) ---
    gt_dir = os.path.join(ROOT, "MLIP", "production", "ground_truth",
                          f"{struct_cfg['temperature_K']}K")
    if not os.path.isdir(gt_dir):
        gt_dir = os.path.join(ROOT, "MLIP", "production", "ground_truth")

    trials = []
    for idx in available_indices:
        seed = _get_seeds(struct_cfg)[idx - 1]
        trial_accuracy = _compute_per_seed_accuracy(
            result_dir, gt_dir, label, idx, pairs, diff_species, struct_cfg,
        )
        trials.append({"seed": seed, "index": idx, "accuracy": trial_accuracy})

    # --- Mean/std across seeds → final accuracy ---
    mean_std = _compute_mean_std(trials)
    accuracy = {}
    if mean_std:
        accuracy["mean"] = mean_std["mean"]
        accuracy["std"] = mean_std["std"]

        # Print summary
        for key in sorted(mean_std["mean"].keys()):
            m = mean_std["mean"][key]
            s = mean_std["std"][key]
            print(f"    {key}: {m:.6f} ± {s:.6f}")

    # Return analysis results (JSON saving is done by run_benchmark.py)
    results = {
        "model": model_name,
        "variant": variant,
        "structure": struct_cfg["label"],
        "temperature_K": struct_cfg["temperature_K"],
        "equilibration_ps": struct_cfg["equilibration_ps"],
        "production_ps": struct_cfg["production_ps"],
        "timestep_fs": struct_cfg["timestep_fs"],
        "traj_interval": traj_interval,
        "num_seeds": len(available_indices),
        "seeds": [_get_seeds(struct_cfg)[i - 1] for i in available_indices],
        "trials": trials,
        "accuracy": accuracy,
    }
    if timing:
        results["timing"] = timing

    return results


def _load_ensemble_avg_rdf(directory, label, pair_str, write=False):
    """Load RDF from directory, ensemble-averaging per-seed files if needed.

    Looks for per-seed files first ({label}_rdf-{idx}_{pair}.csv).
    Falls back to a single file ({label}_rdf_{pair}.csv).
    If write=True and per-seed averaging is performed, saves the averaged file.
    Returns (r, g_r) or None if no files found.
    """
    import glob as globmod

    # Per-seed files
    pattern = os.path.join(directory, f"{label}_rdf-*_{pair_str}.csv")
    seed_files = sorted(globmod.glob(pattern))

    if seed_files:
        g_r_list = []
        n_r_list = []
        r = None
        for f in seed_files:
            data = np.genfromtxt(f, delimiter=",", names=True)
            if r is None:
                r = data["r_angstrom"]
            g_r_list.append(data["g_r"])
            if "n_r" in data.dtype.names:
                n_r_list.append(data["n_r"])
        avg_g_r = np.mean(g_r_list, axis=0) if len(g_r_list) > 1 else g_r_list[0]

        if write and len(seed_files) > 1:
            out_path = os.path.join(directory, f"{label}_rdf_{pair_str}.csv")
            if not os.path.exists(out_path):
                cols = [r, avg_g_r]
                header = "r_angstrom,g_r"
                if n_r_list:
                    avg_n_r = np.mean(n_r_list, axis=0)
                    cols.append(avg_n_r)
                    header += ",n_r"
                np.savetxt(out_path, np.column_stack(cols), delimiter=",",
                           header=header, comments="")
                print(f"    Wrote ensemble-average: {out_path} ({len(seed_files)} seeds)")

        return r, avg_g_r

    # Single file fallback
    single = os.path.join(directory, f"{label}_rdf_{pair_str}.csv")
    if os.path.exists(single):
        data = np.genfromtxt(single, delimiter=",", names=True)
        return data["r_angstrom"], data["g_r"]

    return None


def _load_ensemble_avg_msd(directory, label, write=False):
    """Load MSD from directory, ensemble-averaging per-seed files if needed.

    Looks for per-seed files first ({label}_msd-{idx}.csv).
    Falls back to a single file ({label}_msd.csv).
    If write=True and per-seed averaging is performed, saves the averaged file.
    Returns (t_fs, msd) or None if no files found.
    """
    import glob as globmod

    # Per-seed files
    pattern = os.path.join(directory, f"{label}_msd-*.csv")
    seed_files = sorted(globmod.glob(pattern))

    if seed_files:
        msd_list = []
        msd_xyz_list = []
        t_fs = None
        for f in seed_files:
            data = np.genfromtxt(f, delimiter=",", names=True)
            if t_fs is None:
                t_fs = data["t_fs"]
            msd_list.append(data["msd_angstrom2"])
            xyz_keys = [k for k in ("msd_x", "msd_y", "msd_z") if k in data.dtype.names]
            if len(xyz_keys) == 3:
                msd_xyz_list.append(np.column_stack([data[k] for k in xyz_keys]))
        avg_msd = np.mean(msd_list, axis=0) if len(msd_list) > 1 else msd_list[0]

        if write and len(seed_files) > 1:
            out_path = os.path.join(directory, f"{label}_msd.csv")
            if not os.path.exists(out_path):
                cols = [t_fs, avg_msd]
                header = "t_fs,msd_angstrom2"
                if msd_xyz_list:
                    avg_xyz = np.mean(msd_xyz_list, axis=0)
                    cols.append(avg_xyz)
                    header += ",msd_x,msd_y,msd_z"
                np.savetxt(out_path, np.column_stack(cols), delimiter=",",
                           header=header, comments="")
                print(f"    Wrote ensemble-average: {out_path} ({len(seed_files)} seeds)")

        return t_fs, avg_msd

    # Single file fallback
    single = os.path.join(directory, f"{label}_msd.csv")
    if os.path.exists(single):
        data = np.genfromtxt(single, delimiter=",", names=True)
        return data["t_fs"], data["msd_angstrom2"]

    return None


def _compute_per_seed_accuracy(result_dir, gt_dir, label, idx, pairs, diff_species, struct_cfg):
    """Compute accuracy metrics for a single seed by comparing per-seed CSV vs ground truth."""
    accuracy = {}

    if not os.path.isdir(gt_dir):
        return accuracy

    # Per-seed RDF MAE
    rdf_maes = {}
    for sp_i, sp_j in pairs:
        pair_str = f"{sp_i}-{sp_j}"
        pred_path = os.path.join(result_dir, f"{label}_rdf-{idx}_{pair_str}.csv")
        if not os.path.exists(pred_path):
            continue
        pred = np.genfromtxt(pred_path, delimiter=",", names=True)

        gt_data = _load_ensemble_avg_rdf(gt_dir, label, pair_str)
        if gt_data is None:
            continue
        gt_r, gt_g_r = gt_data

        mae = compute_rdf_mae(pred["r_angstrom"], pred["g_r"], gt_r, gt_g_r)
        if mae is not None:
            rdf_maes[pair_str] = round(mae, 6)

    if rdf_maes:
        rdf_maes["average"] = round(sum(rdf_maes.values()) / len(rdf_maes), 6)
        accuracy["rdf_mae"] = rdf_maes
        accuracy["rdf_score"] = {k: round(1 / (1 + v), 6) for k, v in rdf_maes.items()}

    # Per-seed MSD MAE
    msd_path = os.path.join(result_dir, f"{label}_msd-{idx}.csv")
    gt_msd_data = _load_ensemble_avg_msd(gt_dir, label)

    if os.path.exists(msd_path) and gt_msd_data is not None:
        pred_msd_raw = np.genfromtxt(msd_path, delimiter=",", names=True)
        mae = compute_msd_mae(pred_msd_raw["t_fs"], pred_msd_raw["msd_angstrom2"],
                              gt_msd_data[0], gt_msd_data[1])
        if mae is not None:
            accuracy["msd_mae"] = round(mae, 6)
            accuracy["msd_score"] = round(1 / (1 + mae), 6)

        d_mlip = compute_diffusivity(pred_msd_raw["t_fs"], pred_msd_raw["msd_angstrom2"])
        if d_mlip is not None:
            accuracy["diffusivity_mlip_cm2s"] = d_mlip

    return accuracy


def _compute_mean_std(trials):
    """Compute mean and std of accuracy metrics across trials."""
    if not trials or not any(t["accuracy"] for t in trials):
        return {}

    # Collect all numeric metric paths
    from collections import defaultdict
    values = defaultdict(list)

    for trial in trials:
        acc = trial.get("accuracy", {})
        for key, val in acc.items():
            if isinstance(val, dict):
                for subkey, subval in val.items():
                    if isinstance(subval, (int, float)):
                        values[f"{key}.{subkey}"].append(subval)
            elif isinstance(val, (int, float)):
                values[key].append(val)

    mean = {}
    std = {}
    for path, vals in values.items():
        if len(vals) >= 2:
            mean[path] = round(float(np.mean(vals)), 6)
            std[path] = round(float(np.std(vals, ddof=1)), 6)
        elif len(vals) == 1:
            mean[path] = round(vals[0], 6)
            std[path] = 0.0

    return {"mean": mean, "std": std} if mean else {}



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
    parser.add_argument("--variant", default="pretrained", choices=["pretrained", "finetuned"],
                        help="Model variant: pretrained or finetuned (default: pretrained)")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to fine-tuned checkpoint (required when variant=finetuned)")
    parser.add_argument("--skip_md", action="store_true", help="Skip MD, re-run analysis only")
    parser.add_argument(
        "--skip_analysis", action="store_true", help="Run MD only, skip analysis"
    )
    parser.add_argument("--track_carbon", action="store_true", help="Track carbon emissions")
    args = parser.parse_args()

    config = load_config(args.config)
    struct_cfg = config["structures"][args.structure_index]

    seeds = _get_seeds(struct_cfg)
    print(f"Model: {args.model}")
    print(f"Variant: {args.variant}")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")
    print(f"Structure: {struct_cfg['label']} ({struct_cfg['cif']})")
    print(f"Temperature: {struct_cfg['temperature_K']} K")
    print(f"Equilibration: {struct_cfg['equilibration_ps']} ps")
    print(f"Production: {struct_cfg['production_ps']} ps")
    print(f"Seeds: {seeds}")
    print(f"Traj interval: {struct_cfg.get('traj_interval', 50)} steps")
    print()

    md_info = None
    if not args.skip_md:
        md_info = run_md_simulation(
            struct_cfg, model_name=args.model, track_carbon=args.track_carbon,
            variant=args.variant, checkpoint_path=args.checkpoint,
        )
        print()

    timing = None
    if md_info:
        timing = {
            "equil_seconds": md_info["equil_seconds"],
            "prod_seconds": md_info["prod_seconds"],
            "num_seeds": md_info["num_seeds"],
        }

    if not args.skip_analysis:
        run_analysis(args.model, struct_cfg, timing=timing, variant=args.variant)


if __name__ == "__main__":
    main()
