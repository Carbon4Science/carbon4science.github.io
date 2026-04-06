#!/usr/bin/env python3
"""
Evaluate fine-tuned MLIP models on the shared validation set.

Uses each model's Inference.py _get_calculator(checkpoint_path=...) to load
the fine-tuned model, ensuring consistency with inference pipelines.

Usage:
    python MLIP/evaluate_finetuned.py                    # all models
    python MLIP/evaluate_finetuned.py --model CHGNet     # single model
    python MLIP/evaluate_finetuned.py --device cuda       # specify device
"""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from ase.io import read

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

MODELS = ["CHGNet", "MACE", "SevenNet", "ORB", "PET", "Allegro", "EquFlash", "NequIP"]

# Model -> Inference module path
MODEL_MODULES = {
    "CHGNet": "MLIP.CHGNet.Inference",
    "MACE": "MLIP.MACE.Inference",
    "SevenNet": "MLIP.SevenNet.Inference",
    "ORB": "MLIP.ORB.Inference",
    "PET": "MLIP.PET.Inference",
    "Allegro": "MLIP.Allegro.Inference",
    "EquFlash": "MLIP.EquFlash.Inference",
    "NequIP": "MLIP.NequIP.Inference",
}

# Model -> default fine-tuned checkpoint path (relative to MLIP/finetuned/<model>/)
CHECKPOINT_PATHS = {
    "CHGNet": lambda d: _find_chgnet_best(d),
    "MACE": lambda d: str(d / "MACE_finetuned.model"),
    "SevenNet": lambda d: str(d / "checkpoint_best.pth"),
    "ORB": lambda d: str(d / "best_checkpoint.pt"),
    "PET": lambda d: str(d / "model.pt"),
    "NequIP": lambda d: str(d / "NequIP_finetuned.nequip.pt2") if (d / "NequIP_finetuned.nequip.pt2").exists() else _find_nequip_ckpt(d),
    "Allegro": lambda d: str(d / "Allegro_finetuned.nequip.pt2") if (d / "Allegro_finetuned.nequip.pt2").exists() else _find_nequip_ckpt(d),
    "EquFlash": lambda d: str(d / "checkpoint_best.pth"),
}


def _find_chgnet_best(ft_dir):
    """Find best CHGNet checkpoint."""
    ckpt_dir = ft_dir / "checkpoints"
    for f in sorted(os.listdir(ckpt_dir)):
        if f.startswith("bestE_"):
            return str(ckpt_dir / f)
    raise FileNotFoundError(f"No CHGNet checkpoint in {ckpt_dir}")


def _find_nequip_ckpt(ft_dir):
    """Find best NequIP/Allegro checkpoint."""
    import glob
    ckpts = sorted(glob.glob(str(ft_dir / "outputs" / "**" / "best.ckpt"), recursive=True))
    if ckpts:
        return ckpts[-1]
    raise FileNotFoundError(f"No checkpoint found in {ft_dir}")


def evaluate_model(model_name, val_file, device="cuda"):
    """Evaluate a fine-tuned model on validation data. Returns dict of MAE metrics."""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")

    ft_dir = ROOT_DIR / "MLIP" / "finetuned" / model_name

    # Find checkpoint path
    try:
        checkpoint_path = CHECKPOINT_PATHS[model_name](ft_dir)
        if not os.path.exists(checkpoint_path):
            print(f"  ERROR: checkpoint not found: {checkpoint_path}")
            return None
        print(f"  Checkpoint: {checkpoint_path}")
    except Exception as e:
        print(f"  ERROR finding checkpoint: {e}")
        return None

    # Load calculator via Inference.py _get_calculator
    try:
        module = importlib.import_module(MODEL_MODULES[model_name])
        calc = module._get_calculator(device=device, checkpoint_path=checkpoint_path)
    except Exception as e:
        print(f"  ERROR loading calculator: {e}")
        return None

    frames = read(val_file, index=":")
    n_atoms = len(frames[0])

    energy_errors = []
    force_errors = []
    stress_errors = []

    for i, atoms in enumerate(frames):
        # Ground truth
        e_true = atoms.get_potential_energy()
        f_true = atoms.get_forces()
        try:
            s_true = atoms.get_stress(voigt=False)  # 3x3, eV/A^3
        except Exception:
            s_true = None

        # Predict
        atoms_copy = atoms.copy()
        atoms_copy.calc = calc
        try:
            e_pred = atoms_copy.get_potential_energy()
            f_pred = atoms_copy.get_forces()

            energy_errors.append(abs(e_pred - e_true) / n_atoms)  # eV/atom
            force_errors.append(np.mean(np.abs(f_pred - f_true)))  # eV/A

            if s_true is not None:
                try:
                    s_pred = atoms_copy.get_stress(voigt=False)
                    stress_errors.append(np.mean(np.abs(s_pred - s_true)))
                except Exception:
                    pass

        except Exception as e:
            print(f"  WARNING: prediction failed for frame {i}: {e}")
            continue

    if not energy_errors:
        print(f"  ERROR: no successful predictions")
        return None

    results = {
        "energy_mae_eV_atom": float(np.mean(energy_errors)),
        "force_mae_eV_A": float(np.mean(force_errors)),
        "n_val": len(energy_errors),
    }
    if stress_errors:
        results["stress_mae_eV_A3"] = float(np.mean(stress_errors))

    print(f"  Energy MAE: {results['energy_mae_eV_atom']*1000:.3f} meV/atom")
    print(f"  Force  MAE: {results['force_mae_eV_A']*1000:.3f} meV/A")
    if stress_errors:
        print(f"  Stress MAE: {results['stress_mae_eV_A3']:.6f} eV/A^3")
    print(f"  ({results['n_val']} structures evaluated)")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Single model to evaluate (default: all)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--val_file", default=None, help="Validation file (default: auto-detect)")
    parser.add_argument("--output", default="MLIP/finetuned/val_results.json")
    args = parser.parse_args()

    # Use shared validation file (all models have identical val.extxyz)
    if args.val_file is None:
        args.val_file = str(ROOT_DIR / "MLIP" / "finetuned" / "CHGNet" / "val.extxyz")

    models = [args.model] if args.model else MODELS
    all_results = {}

    for model_name in models:
        result = evaluate_model(model_name, args.val_file, device=args.device)
        if result:
            all_results[model_name] = result
            # Also update the model's finetune_results.json
            ft_json = ROOT_DIR / "MLIP" / "finetuned" / model_name / "finetune_results.json"
            if ft_json.exists():
                with open(ft_json) as f:
                    ft_data = json.load(f)
                ft_data["val_errors"] = result
                with open(ft_json, "w") as f:
                    json.dump(ft_data, f, indent=2, default=str)

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Model':<12} {'Energy MAE (meV/atom)':>22} {'Force MAE (meV/A)':>20} {'Stress MAE (eV/A³)':>20}")
    print(f"{'-'*80}")
    for model, r in sorted(all_results.items(), key=lambda x: x[1]["force_mae_eV_A"]):
        s = f"{r.get('stress_mae_eV_A3', 0):.6f}" if "stress_mae_eV_A3" in r else "N/A"
        print(f"{model:<12} {r['energy_mae_eV_atom']*1000:>22.3f} {r['force_mae_eV_A']*1000:>20.3f} {s:>20}")
    print(f"{'='*80}")

    # Save combined results
    output_path = ROOT_DIR / args.output
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
