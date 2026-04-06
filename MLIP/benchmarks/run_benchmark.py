#!/usr/bin/env python3
"""
MLIP Benchmark Runner

Runs production MD benchmarks for MLIP models (pretrained or finetuned).

Flow:
  1. Get calculator from model's Inference.py
  2. Run MD simulation via run_production_md.run_md_simulation()
  3. Run analysis via run_production_md.run_analysis()
  4. Save single results JSON to production/results/{variant}/{model}/

Usage:
    python MLIP/benchmarks/run_benchmark.py --model CHGNet
    python MLIP/benchmarks/run_benchmark.py --model CHGNet --variant finetuned \\
        --checkpoint MLIP/finetuned/CHGNet/checkpoints/best.pth.tar
    python MLIP/benchmarks/run_benchmark.py --model CHGNet --skip_md  # re-run analysis only
    python MLIP/benchmarks/run_benchmark.py --list_models
"""

import argparse
import importlib
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Model name → Inference module mapping
MODELS = {
    "eSEN": "MLIP.eSEN.Inference",
    "eSEN_OAM": "MLIP.eSEN_OAM.Inference",
    "NequIP": "MLIP.NequIP.Inference",
    "NequIP_OAM": "MLIP.NequIP_OAM.Inference",
    "Allegro": "MLIP.Allegro.Inference",
    "Nequix": "MLIP.Nequix.Inference",
    "DPA3": "MLIP.DPA3.Inference",
    "SevenNet": "MLIP.SevenNet.Inference",
    "MACE": "MLIP.MACE.Inference",
    "MACE_pruned": "MLIP.MACE.Inference",
    "ORB": "MLIP.ORB.Inference",
    "CHGNet": "MLIP.CHGNet.Inference",
    "PET": "MLIP.PET.Inference",
    "EquFlash": "MLIP.EquFlash.Inference",
}

DEFAULT_CONFIG = "MLIP/production/configs/LGPS_300K.json"


def count_model_parameters(model_name: str) -> Optional[int]:
    """Count trainable parameters from cached model in module globals."""
    try:
        import torch
        module = sys.modules.get(MODELS[model_name])
        if module is None:
            return None
        for attr_name in dir(module):
            obj = getattr(module, attr_name, None)
            if isinstance(obj, torch.nn.Module):
                return sum(p.numel() for p in obj.parameters())
        for var_name in ['_model', '_calculator']:
            obj = getattr(module, var_name, None)
            if obj is None:
                continue
            if isinstance(obj, torch.nn.Module):
                return sum(p.numel() for p in obj.parameters())
            inner = getattr(obj, 'model', None)
            if isinstance(inner, torch.nn.Module):
                return sum(p.numel() for p in inner.parameters())
    except Exception:
        pass
    return None


def run_benchmark(
    model_name: str,
    production_config: str = DEFAULT_CONFIG,
    structure_index: int = 0,
    track_carbon: bool = False,
    variant: str = "pretrained",
    checkpoint: Optional[str] = None,
    skip_md: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run MLIP production MD benchmark and save results."""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")

    from MLIP.production.run_production_md import load_config, run_md_simulation, run_analysis

    config = load_config(production_config)
    struct_cfg = config["structures"][structure_index]
    label = struct_cfg["label"]

    if verbose:
        print("=" * 60)
        print(f"Model:   {model_name}")
        print(f"Variant: {variant}")
        print(f"Config:  {production_config}")
        if checkpoint:
            print(f"Ckpt:    {checkpoint}")
        print(f"Skip MD: {'Yes' if skip_md else 'No'}")
        print(f"Carbon:  {'Yes' if track_carbon else 'No'}")
        print("=" * 60)
        print()

    # --- Results JSON path ---
    result_dir = Path(ROOT_DIR) / "MLIP" / "production" / "results" / variant / model_name
    result_dir.mkdir(parents=True, exist_ok=True)
    json_path = result_dir / f"{label}_results.json"

    # Load existing results to preserve timing/carbon on skip_md
    existing = {}
    if json_path.exists():
        with open(json_path) as f:
            existing = json.load(f)

    # --- Step 1: MD simulation (or skip) ---
    timing = {}
    carbon_metrics = {}

    if skip_md:
        # Preserve timing/carbon from existing results
        timing = existing.get("timing", {})
        carbon_metrics = existing.get("carbon", {})
    else:
        # Get calculator from Inference.py
        module = importlib.import_module(MODELS[model_name])
        calc = module._get_calculator(checkpoint_path=checkpoint)

        # Run MD
        md_info = run_md_simulation(
            struct_cfg,
            calculator=calc,
            model_name=model_name,
            track_carbon=track_carbon,
            variant=variant,
        )

        timing = {
            "equil_seconds": md_info["equil_seconds"],
            "prod_seconds": md_info["prod_seconds"],
            "num_seeds": md_info["num_seeds"],
        }
        carbon_metrics = md_info.get("carbon_metrics", {})

    # --- Step 2: Analysis (always runs) ---
    analysis = run_analysis(model_name, struct_cfg, timing=timing, variant=variant)

    # --- Step 3: Compile results ---
    accuracy = analysis.get("accuracy", {})
    trials = analysis.get("trials", [])

    # Add CPS from evaluate.py (static metric, not from MD)
    from MLIP.evaluate import CPS_VALUES
    accuracy["CPS"] = CPS_VALUES.get(model_name, 0.0)

    # Speed from timing (prod_seconds is already per-seed average)
    speed = {}
    if timing.get("prod_seconds", 0) > 0:
        prod_steps = int(struct_cfg["production_ps"] * 1000 / struct_cfg["timestep_fs"])
        sps = prod_steps / timing["prod_seconds"]
        speed = {
            "steps_per_second": round(sps, 2),
            "ns_per_day": round(sps * struct_cfg["timestep_fs"] * 1e-6 * 86400, 4),
            "steps": prod_steps,
        }

    # Model parameters
    num_params = count_model_parameters(model_name)

    results = {
        "task": "MLIP",
        "model": model_name,
        "variant": variant,
        "model_params": num_params,
        "accuracy": accuracy,
        "trials": trials,
        "timing": timing,
        "speed": speed,
        "carbon": carbon_metrics,
    }
    if checkpoint:
        results["checkpoint"] = checkpoint

    # --- Step 4: Print ---
    if verbose:
        print()
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        if num_params is not None:
            if num_params >= 1_000_000:
                print(f"Params:  {num_params:,} ({num_params/1e6:.2f}M)")
            else:
                print(f"Params:  {num_params:,} ({num_params/1e3:.1f}K)")

        mean = accuracy.get("mean", {})
        std = accuracy.get("std", {})
        if mean:
            print()
            print("Accuracy (mean ± std across seeds):")
            for key in sorted(mean.keys()):
                print(f"  {key}: {mean[key]:.6f} ± {std.get(key, 0):.6f}")

        if speed:
            print()
            print("Speed:")
            print(f"  Steps/sec:  {speed['steps_per_second']:.2f}")
            print(f"  ns/day:     {speed['ns_per_day']:.4f}")

        if carbon_metrics:
            print()
            print("Carbon:")
            for key in ["duration_seconds", "energy_wh", "emissions_g_co2"]:
                val = carbon_metrics.get(key)
                if val is not None:
                    print(f"  {key}: {val}")

        print("=" * 60)

    # --- Step 5: Save ---
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    if verbose:
        print(f"\nResults saved to: {json_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="MLIP Benchmark Runner — production MD with RDF/MSD analysis",
    )
    parser.add_argument("--model", type=str, required=True,
                        help=f"Model name: {', '.join(MODELS.keys())}")
    parser.add_argument("--variant", type=str, default="pretrained",
                        choices=["pretrained", "finetuned"],
                        help="Model variant (default: pretrained)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to finetuned checkpoint (required if variant=finetuned)")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG,
                        help=f"Production MD config JSON (default: {DEFAULT_CONFIG})")
    parser.add_argument("--structure_index", type=int, default=0,
                        help="Structure index in config (default: 0)")
    parser.add_argument("--track_carbon", action="store_true",
                        help="Track carbon emissions")
    parser.add_argument("--skip_md", action="store_true",
                        help="Skip MD, re-run analysis only (trajectories must exist)")
    parser.add_argument("--list_models", action="store_true",
                        help="List available models")

    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for model in MODELS:
            print(f"  {model}")
        return

    if args.variant == "finetuned" and not args.checkpoint:
        parser.error("--checkpoint is required when --variant=finetuned")

    run_benchmark(
        model_name=args.model,
        production_config=args.config,
        structure_index=args.structure_index,
        track_carbon=args.track_carbon,
        variant=args.variant,
        checkpoint=args.checkpoint,
        skip_md=args.skip_md,
    )


if __name__ == "__main__":
    main()
