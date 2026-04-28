#!/usr/bin/env python3
"""Unified-protocol relaxation runner.

Applies a single protocol (default: FIRE + FrechetCellFilter + fmax=0.05 +
max_steps=500) to every model in the subset so relax-cost is directly
comparable across models. Cost of the entire 100-structure loop is wrapped
by a single CarbonTracker.

Usage (inside a model's conda env):
    python MLIP/relaxation/run_relaxation.py --model CHGNet
    python MLIP/relaxation/run_relaxation.py --model MACE --config MLIP/relaxation/configs/wbm_100.json

Output:
    MLIP/relaxation/results/unified/<Model>/relax_results.json
    MLIP/relaxation/results/unified/<Model>/final/<material_id>.xyz
    MLIP/relaxation/results/unified/<Model>/traces/<material_id>.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# carbon_tracker lives under MLIP/benchmarks
BENCHMARKS_DIR = ROOT / "MLIP" / "benchmarks"
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

from MLIP.relaxation.relax_core import (
    import_calculator,
    load_subset,
    relax_one_structure,
)


def run(
    model: str,
    config_path: Path | None = None,
    config: dict | None = None,
    variant: str = "pretrained",
    track_carbon: bool = True,
    checkpoint_path: str | None = None,
    limit: int | None = None,
    results_root: Path | None = None,
):
    if config is None:
        if config_path is None:
            raise ValueError("Either config or config_path must be provided")
        with open(config_path) as f:
            config = json.load(f)

    subset = config["subset"]
    protocol = config["protocol"]

    xyz_path = ROOT / subset["xyz"]
    ids_path = ROOT / subset["ids_json"]
    with open(ids_path) as f:
        subset_meta = json.load(f)

    print(f"Model: {model}")
    print(f"Subset: {subset['name']} ({xyz_path})")
    print(f"Protocol: {protocol}")

    atoms_list = load_subset(xyz_path)
    if limit is not None:
        atoms_list = atoms_list[:limit]
    print(f"Loaded {len(atoms_list)} structures")

    bucket = "unified" if protocol.get("name") == "unified" else "specific"
    results_root = results_root or (HERE / "results" / bucket)
    out_dir = results_root / model
    final_dir = out_dir / "final"
    traces_dir = out_dir / "traces"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the model's calculator once
    print(f"Building {model} calculator…")
    calc = import_calculator(model, checkpoint_path=checkpoint_path)

    tracker = None
    if track_carbon:
        from carbon_tracker import CarbonTracker

        tracker = CarbonTracker(
            project_name=f"{model}_MLIP_relaxation_{bucket}",
            output_dir=str(out_dir),
            model_name=model,
            task="inference",
            save_results=False,
        )
        tracker.start()

    per_structure = []
    t_loop = time.perf_counter()
    for i, atoms in enumerate(atoms_list):
        mid = atoms.info.get("material_id", f"subset-{i}")
        print(f"[{i+1}/{len(atoms_list)}] {mid} n_atoms={len(atoms)}", flush=True)
        res = relax_one_structure(
            atoms,
            calc,
            optimizer=protocol["optimizer"],
            cell_filter=protocol["cell_filter"],
            fmax=float(protocol["fmax"]),
            max_steps=int(protocol["max_steps"]),
            traces_dir=traces_dir,
            final_dir=final_dir,
            material_id=mid,
        )
        status = "OK " if res["error"] is None else "ERR"
        conv = "conv" if res["converged"] else "unc "
        print(
            f"   {status} {conv} steps={res['steps']:3d} "
            f"fmax={res['final_fmax']} E={res['final_energy']} "
            f"wall={res['wall_seconds']:.2f}s",
            flush=True,
        )
        per_structure.append(res)
    total_wall = time.perf_counter() - t_loop

    carbon_metrics = None
    if tracker is not None:
        tracker.stop()
        carbon_metrics = tracker.get_metrics()

    n_conv = sum(1 for r in per_structure if r["converged"])
    n_err = sum(1 for r in per_structure if r["error"] is not None)
    total_steps = sum(int(r["steps"]) for r in per_structure)
    n_ok = len(per_structure) - n_err
    mean_steps = (total_steps / n_ok) if n_ok else 0.0

    summary = {
        "model": model,
        "variant": variant,
        "protocol": protocol,
        "subset": {"name": subset["name"], "n": len(atoms_list), **{
            k: subset_meta[k] for k in ("seed", "unique_prototypes_only") if k in subset_meta
        }},
        "aggregate": {
            "num_structures": len(atoms_list),
            "num_converged": n_conv,
            "num_errors": n_err,
            "convergence_rate": (n_conv / len(atoms_list)) if atoms_list else 0.0,
            "total_loop_seconds": round(total_wall, 3),
            "total_steps": total_steps,
            "mean_steps_per_structure": round(mean_steps, 2),
        },
        "per_structure": per_structure,
    }
    if carbon_metrics is not None:
        summary["carbon"] = carbon_metrics

    out_json = out_dir / "relax_results.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote {out_json}")
    print(
        f"Summary: converged {n_conv}/{len(atoms_list)}, "
        f"errors={n_err}, total_steps={total_steps}, "
        f"loop_wall={total_wall:.1f}s"
    )


def main():
    parser = argparse.ArgumentParser(description="Unified-protocol relaxation benchmark")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--config",
        default=str(HERE / "configs" / "wbm_100.json"),
        help="Path to relaxation config JSON",
    )
    parser.add_argument("--variant", default="pretrained", choices=["pretrained", "finetuned"])
    parser.add_argument("--checkpoint", default=None, help="Finetuned checkpoint path")
    parser.add_argument("--track_carbon", action="store_true", default=True)
    parser.add_argument("--no_carbon", dest="track_carbon", action="store_false")
    parser.add_argument("--limit", type=int, default=None, help="Truncate subset for debugging")
    args = parser.parse_args()

    run(
        model=args.model,
        config_path=Path(args.config),
        variant=args.variant,
        track_carbon=args.track_carbon,
        checkpoint_path=args.checkpoint,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
