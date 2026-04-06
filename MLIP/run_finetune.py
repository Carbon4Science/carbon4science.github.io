#!/usr/bin/env python3
"""
Common fine-tuning orchestrator for MLIP models.

Usage:
    python MLIP/run_finetune.py --model CHGNet --dataset MLIP/dataset_LGPS_450K_n100.xyz
    python MLIP/run_finetune.py --model MACE --dataset MLIP/dataset_LGPS_450K_n100.xyz --epochs 50 --track_carbon
"""

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from ase.io import read, write

# Add repository root to path (up from MLIP/ to Carbon4Science/)
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

MODEL_MODULES = {
    "CHGNet": "MLIP.CHGNet.Finetune",
    "MACE": "MLIP.MACE.Finetune",
    "SevenNet": "MLIP.SevenNet.Finetune",
    "DPA3": "MLIP.DPA3.Finetune",
    "NequIP": "MLIP.NequIP.Finetune",
    "ORB": "MLIP.ORB.Finetune",
    "PET": "MLIP.PET.Finetune",
    "Allegro": "MLIP.Allegro.Finetune",
    "EquFlash": "MLIP.EquFlash.Finetune",
}

# Model-to-conda-env mapping (synced with benchmarks/run.sh)
MODEL_ENVS = {
    "CHGNet": "chgnet",
    "MACE": "mace",
    "SevenNet": "sevennet",
    "DPA3": "deepmd",
    "NequIP": "nequip",
    "ORB": "orb",
    "PET": "pet-oam",
    "Allegro": "nequip",
    "EquFlash": "equflash",
}


def split_dataset(dataset_path, output_dir, val_ratio=0.1, seed=42):
    """Split extxyz dataset into train/val. Returns (train_path, val_path, n_train, n_val)."""
    frames = read(dataset_path, index=":")
    n = len(frames)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_val = max(1, int(n * val_ratio))
    val_idx = sorted(indices[:n_val])
    train_idx = sorted(indices[n_val:])

    train_path = os.path.join(output_dir, "train.extxyz")
    val_path = os.path.join(output_dir, "val.extxyz")

    write(train_path, [frames[i] for i in train_idx])
    write(val_path, [frames[i] for i in val_idx])

    return train_path, val_path, len(train_idx), len(val_idx)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MLIP models")
    parser.add_argument("--model", required=True, choices=list(MODEL_MODULES.keys()),
                        help="Model to fine-tune")
    parser.add_argument("--dataset", required=True, help="Path to extxyz dataset")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: MLIP/finetuned/<model>)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--track_carbon", action="store_true")
    parser.add_argument("--output_json", default=None, help="Path to save results JSON")
    parser.add_argument("--clean", action="store_true",
                        help="Delete output_dir before training (removes previous checkpoints, logs, etc.)")
    args = parser.parse_args()

    # Setup output directory
    if args.output_dir is None:
        args.output_dir = str(ROOT_DIR / "MLIP" / "finetuned" / args.model)

    if args.clean and os.path.isdir(args.output_dir):
        import shutil
        print(f"Cleaning output directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"Model:     {args.model}")
    print(f"Dataset:   {args.dataset}")
    print(f"Output:    {args.output_dir}")
    print(f"Epochs:    {args.epochs}")
    print(f"Val ratio: {args.val_ratio}")
    print(f"Device:    {args.device}")
    print(f"Carbon:    {'Yes' if args.track_carbon else 'No'}")
    print("=" * 60)
    print()

    # Split dataset
    print("Splitting dataset...")
    train_path, val_path, n_train, n_val = split_dataset(
        args.dataset, args.output_dir, val_ratio=args.val_ratio, seed=args.seed
    )
    print(f"  Train: {n_train} structures -> {train_path}")
    print(f"  Val:   {n_val} structures -> {val_path}")
    print()

    # Import model's Finetune module
    module = importlib.import_module(MODEL_MODULES[args.model])

    # Carbon tracking
    start_time = time.time()
    carbon_metrics = None
    if args.track_carbon:
        sys.path.insert(0, str(ROOT_DIR / "MLIP" / "benchmarks"))
        from carbon_tracker import CarbonTracker
        tracker = CarbonTracker(
            project_name=f"{args.model}_finetune",
            model_name=args.model,
            task="finetune",
            save_results=False,
        )
        tracker.start()

    # Run fine-tuning
    result = module.finetune(
        train_file=train_path,
        val_file=val_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        device=args.device,
    )

    duration = time.time() - start_time

    if args.track_carbon:
        tracker.stop()
        carbon_metrics = tracker.get_metrics()
    else:
        carbon_metrics = {"duration_seconds": round(duration, 2)}

    # Compile results
    results = {
        "task": "MLIP_finetune",
        "model": args.model,
        "dataset": str(args.dataset),
        "n_train": n_train,
        "n_val": n_val,
        "epochs": args.epochs,
        "checkpoint_path": result.get("checkpoint_path", ""),
        "val_errors": result.get("val_errors", {}),
        "carbon": carbon_metrics,
    }

    # Print results
    print()
    print("=" * 60)
    print("FINE-TUNING RESULTS")
    print("=" * 60)
    print(f"Checkpoint: {result.get('checkpoint_path', 'N/A')}")
    print()
    val_errors = result.get("val_errors", {})
    if val_errors:
        print("Validation Errors:")
        for k, v in val_errors.items():
            print(f"  {k}: {v:.6f}")
    print()
    print(f"Duration: {carbon_metrics.get('duration_seconds', duration):.1f}s")
    if carbon_metrics.get("energy_wh"):
        print(f"Energy:   {carbon_metrics['energy_wh']:.4f} Wh")
    if carbon_metrics.get("emissions_g_co2"):
        print(f"CO2:      {carbon_metrics['emissions_g_co2']:.4f} g")
    print("=" * 60)

    # Save results
    output_json = args.output_json or os.path.join(args.output_dir, "finetune_results.json")
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_json}")

    return results


if __name__ == "__main__":
    main()
