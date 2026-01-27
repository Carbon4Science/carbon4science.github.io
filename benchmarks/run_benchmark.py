#!/usr/bin/env python3
"""
Unified Retrosynthesis Benchmark Runner

Usage:
    python run_benchmark.py --model neuralsym --task inference --device cuda:0 --metric top_1 top_10
    python run_benchmark.py --model LocalRetro --task inference --input test.csv --track_carbon
"""

import argparse
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Optional

# Add parent directory to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

MODELS = {
    "neuralsym": {
        "module": "Retrosynthesis.neuralsym.Inference",
        "requires_init": False,
    },
    "LocalRetro": {
        "module": "Retrosynthesis.LocalRetro.Inference",
        "requires_init": False,
    },
    "RetroBridge": {
        "module": "Retrosynthesis.RetroBridge.Inference",
        "requires_init": False,
    },
    "Chemformer": {
        "module": "Retrosynthesis.Chemformer.Inference",
        "requires_init": True,
        "init_func": "load_model",
    },
    "RSGPT": {
        "module": "Retrosynthesis.RSGPT.inference",
        "requires_init": False,
    },
}

METRICS = ["top_1", "top_3", "top_5", "top_10", "top_50"]


def get_model(model_name: str, device: str = "cuda:0"):
    """Dynamically import and return the run function for a model."""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")

    config = MODELS[model_name]
    module = __import__(config["module"], fromlist=["run"])

    # Some models accept device parameter
    run_func = module.run

    # Chemformer requires initialization
    if config.get("requires_init"):
        init_func = getattr(module, config["init_func"])
        return run_func, init_func

    return run_func, None


def calculate_metrics(
    results: List[Dict],
    ground_truth: List[str],
    metrics: List[str] = ["top_1", "top_10"]
) -> Dict[str, float]:
    """Calculate accuracy metrics."""
    try:
        from rdkit import Chem
    except ImportError:
        print("Warning: RDKit not available, skipping accuracy calculation")
        return {}

    def canonicalize(smiles: str) -> Optional[str]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            return Chem.MolToSmiles(mol) if mol else None
        except:
            return None

    n_samples = len(ground_truth)
    metric_results = {}

    for metric in metrics:
        k = int(metric.split("_")[1])
        correct = 0

        for result, gt in zip(results, ground_truth):
            gt_canon = canonicalize(gt)
            if gt_canon is None:
                continue

            predictions = result.get("predictions", [])[:k]
            for pred in predictions:
                pred_smiles = pred.get("smiles", "") if isinstance(pred, dict) else pred
                if canonicalize(pred_smiles) == gt_canon:
                    correct += 1
                    break

        metric_results[metric] = correct / n_samples if n_samples > 0 else 0.0

    return metric_results


def run_inference(
    model_name: str,
    smiles: List[str],
    device: str = "cuda:0",
    top_k: int = 10,
    track_carbon: bool = False,
    ground_truth: List[str] = None,
    metrics: List[str] = ["top_1", "top_10"],
) -> Dict:
    """Run inference benchmark."""
    run_func, init_func = get_model(model_name, device)

    # Initialize model if needed (Chemformer)
    if init_func is not None:
        print(f"Note: {model_name} requires initialization.")
        print("Please call load_model() with appropriate paths first.")

    # Run with or without carbon tracking
    if track_carbon:
        from carbon_tracker import CarbonTracker
        tracker = CarbonTracker(
            project_name=f"{model_name}_inference",
            model_name=model_name,
            task="inference"
        )
        with tracker:
            results = run_func(smiles, top_k=top_k)
        carbon_metrics = tracker.get_metrics()
    else:
        start_time = time.time()
        results = run_func(smiles, top_k=top_k)
        duration = time.time() - start_time
        carbon_metrics = {"duration_seconds": duration}

    # Calculate accuracy if ground truth provided
    accuracy = {}
    if ground_truth:
        accuracy = calculate_metrics(results, ground_truth, metrics)

    return {
        "model": model_name,
        "task": "inference",
        "device": device,
        "n_samples": len(smiles),
        "top_k": top_k,
        "accuracy": accuracy,
        "carbon": carbon_metrics,
        "predictions": results,
    }


def run_training(
    model_name: str,
    device: str = "cuda:0",
    track_carbon: bool = False,
) -> Dict:
    """Run training benchmark (placeholder - requires model-specific implementation)."""
    print(f"Training benchmark for {model_name}")
    print("Please use model-specific training scripts with CarbonTracker:")
    print(f"""
    from benchmarks.carbon_tracker import CarbonTracker

    tracker = CarbonTracker(
        project_name="{model_name}_training",
        model_name="{model_name}",
        task="training"
    )

    with tracker:
        # Your training code here
        train_model()

    tracker.print_summary()
    """)
    return {"model": model_name, "task": "training", "status": "use_model_specific_script"}


def main():
    parser = argparse.ArgumentParser(
        description="Unified Retrosynthesis Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Core arguments
    parser.add_argument("--model", type=str, required=False,
                        help=f"Model: {', '.join(MODELS.keys())}")
    parser.add_argument("--task", type=str, default="inference",
                        choices=["inference", "training"],
                        help="Task type (default: inference)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device: cuda:0, cuda:1, cpu (default: cuda:0)")
    parser.add_argument("--metric", type=str, nargs="+", default=["top_1", "top_10"],
                        help=f"Metrics: {', '.join(METRICS)} (default: top_1 top_10)")

    # Input/output
    parser.add_argument("--smiles", type=str, nargs="+",
                        help="SMILES string(s) to predict")
    parser.add_argument("--input", type=str,
                        help="Input CSV/TXT file with SMILES")
    parser.add_argument("--ground_truth", type=str,
                        help="Ground truth file for accuracy calculation")
    parser.add_argument("--output", type=str,
                        help="Output JSON file for results")

    # Options
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of predictions (default: 10)")
    parser.add_argument("--track_carbon", action="store_true",
                        help="Track carbon emissions")
    parser.add_argument("--list_models", action="store_true",
                        help="List available models")
    parser.add_argument("--list_metrics", action="store_true",
                        help="List available metrics")

    args = parser.parse_args()

    # List options
    if args.list_models:
        print("Available models:")
        for name in MODELS:
            print(f"  {name}")
        return

    if args.list_metrics:
        print("Available metrics:")
        for m in METRICS:
            print(f"  {m}")
        return

    # Validate args
    if not args.model:
        parser.error("--model is required")

    if args.model not in MODELS:
        parser.error(f"Unknown model: {args.model}. Use --list_models to see options.")

    # Handle training task
    if args.task == "training":
        result = run_training(args.model, args.device, args.track_carbon)
        return

    # Get SMILES for inference
    smiles_list = []
    if args.smiles:
        smiles_list = args.smiles
    elif args.input:
        input_path = Path(args.input)
        if input_path.suffix == ".csv":
            import pandas as pd
            df = pd.read_csv(input_path)
            # Try common column names
            for col in ["smiles", "SMILES", "product", "products", "input"]:
                if col in df.columns:
                    smiles_list = df[col].tolist()
                    break
            if not smiles_list:
                smiles_list = df.iloc[:, 0].tolist()
        else:
            with open(input_path) as f:
                smiles_list = [line.strip() for line in f if line.strip()]
    else:
        parser.error("--smiles or --input required for inference")

    # Load ground truth if provided
    ground_truth = None
    if args.ground_truth:
        gt_path = Path(args.ground_truth)
        if gt_path.suffix == ".csv":
            import pandas as pd
            df = pd.read_csv(gt_path)
            for col in ["reactants", "reactant", "ground_truth", "target"]:
                if col in df.columns:
                    ground_truth = df[col].tolist()
                    break
        else:
            with open(gt_path) as f:
                ground_truth = [line.strip() for line in f if line.strip()]

    # Print config
    print("=" * 50)
    print(f"Model:   {args.model}")
    print(f"Task:    {args.task}")
    print(f"Device:  {args.device}")
    print(f"Metrics: {args.metric}")
    print(f"Top-k:   {args.top_k}")
    print(f"Samples: {len(smiles_list)}")
    print(f"Carbon:  {'Yes' if args.track_carbon else 'No'}")
    print("=" * 50)

    # Run benchmark
    result = run_inference(
        model_name=args.model,
        smiles=smiles_list,
        device=args.device,
        top_k=args.top_k,
        track_carbon=args.track_carbon,
        ground_truth=ground_truth,
        metrics=args.metric,
    )

    # Print results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    if result["accuracy"]:
        print("\nAccuracy:")
        for metric, value in result["accuracy"].items():
            print(f"  {metric}: {value:.2%}")

    print(f"\nDuration: {result['carbon'].get('duration_seconds', 0):.2f}s")

    if args.track_carbon:
        print(f"Energy:   {result['carbon'].get('energy_kwh', 0):.6f} kWh")
        print(f"CO2:      {result['carbon'].get('emissions_kg_co2', 0):.6f} kg")

    # Show sample predictions
    print("\nSample predictions:")
    for r in result["predictions"][:3]:
        print(f"  Input: {r['input'][:50]}...")
        for p in r["predictions"][:2]:
            print(f"    -> {p['smiles'][:50]}...")

    # Save results
    if args.output:
        # Remove full predictions for smaller file size
        save_result = {k: v for k, v in result.items() if k != "predictions"}
        save_result["n_predictions"] = len(result["predictions"])
        with open(args.output, "w") as f:
            json.dump(save_result, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
