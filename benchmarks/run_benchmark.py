#!/usr/bin/env python3
"""
Benchmark Runner for Forward (Forward Reaction Prediction)

Usage:
    python benchmarks/run_benchmark.py --model LocalTransform --track_carbon
    python benchmarks/run_benchmark.py --model neuralsym --limit 500
"""

import argparse
import importlib
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

TASK_NAME = "Forward"

EVAL_MODULE = "evaluate"

MODELS = {
    "neuralsym": "neuralsym.Inference",
    "MolecularTransformer": "MolecularTransformer.Inference",
    "MEGAN": "MEGAN.Inference",
    "Graph2SMILES": "Graph2SMILES.Inference",
    "Chemformer": "Chemformer.Inference",
    "RSMILES_20x": "RSMILES.Inference",
    "LlaSMol": "LlaSMol.Inference",
    "WLDN": "WLDN.Inference",
}

MODEL_ENVS = {
    "neuralsym": "neuralsym",
    "MolecularTransformer": "mol_transformer",
    "MEGAN": "megan",
    "Graph2SMILES": "mol_transformer",
    "Chemformer": "chemformer",
    "RSMILES_20x": "rsmiles",
    "LlaSMol": "llasmol",
    "WLDN": "wldn",
}


def get_task_evaluator():
    return importlib.import_module(EVAL_MODULE)


def get_model_run_func(model_name: str):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
    module = importlib.import_module(MODELS[model_name])
    return module.run


def count_model_parameters(model_name: str) -> Optional[int]:
    try:
        import torch
        module_name = MODELS[model_name]
        module = sys.modules.get(module_name)
        if module is None:
            return None
        for attr_name in dir(module):
            obj = getattr(module, attr_name, None)
            if isinstance(obj, torch.nn.Module):
                return sum(p.numel() for p in obj.parameters())
        for var_name in ['_model', '_proposer', '_model_instance']:
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
    limit: Optional[int] = None,
    top_k: int = 5,
    metrics: Optional[List[str]] = None,
    track_carbon: bool = False,
    data_path: Optional[str] = None,
    output_path: Optional[str] = None,
    save_predictions: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    evaluator = get_task_evaluator()
    model_run = get_model_run_func(model_name)

    if metrics is None:
        metrics = evaluator.METRICS

    if verbose:
        print("=" * 60)
        print(f"Task:    {TASK_NAME}")
        print(f"Model:   {model_name}")
        print(f"Metrics: {metrics}")
        print(f"Top-k:   {top_k}")
        print(f"Limit:   {limit or 'Full dataset'}")
        print(f"Carbon:  {'Yes' if track_carbon else 'No'}")
        print("=" * 60)
        print()

    if verbose:
        print("Loading test data...")
    test_cases = evaluator.load_test_data(data_path=data_path, limit=limit)
    if verbose:
        print(f"Loaded {len(test_cases)} test cases\n")

    if track_carbon:
        sys.path.insert(0, str(ROOT_DIR / "benchmarks"))
        from carbon_tracker import CarbonTracker
        tracker = CarbonTracker(
            project_name=f"{model_name}_{TASK_NAME}_benchmark",
            model_name=model_name,
            task="inference",
            save_results=False,
        )
    else:
        tracker = None

    if verbose:
        print(f"Running inference (top_k={top_k})...")
        sys.stdout.flush()

    predictions = []
    start_time = time.time()

    if tracker:
        tracker.start()

    for i, tc in enumerate(test_cases):
        try:
            result = model_run(tc['reactants'], top_k=top_k)
            predictions.append(result[0] if result else {'input': tc['reactants'], 'predictions': []})
        except Exception:
            predictions.append({'input': tc['reactants'], 'predictions': []})

        if verbose and (i + 1) % 100 == 0:
            intermediate_results = evaluator.evaluate(predictions, test_cases[:i+1], metrics)
            acc_str = ", ".join([f"{m}: {intermediate_results[m]*100:.1f}%" for m in metrics[:2]])
            print(f"  {i+1}/{len(test_cases)} - {acc_str}")
            sys.stdout.flush()

    duration = time.time() - start_time

    if tracker:
        tracker.stop()
        carbon_metrics = tracker.get_metrics()
    else:
        carbon_metrics = {"duration_seconds": duration}

    if verbose:
        print("\nEvaluating predictions...")
    eval_results = evaluator.evaluate(predictions, test_cases, metrics)

    num_params = count_model_parameters(model_name)

    results = {
        "task": TASK_NAME,
        "model": model_name,
        "num_samples": len(test_cases),
        "top_k": top_k,
        "model_params": num_params,
        "metrics": metrics,
        "accuracy": {m: eval_results[m] for m in metrics},
        "correct": eval_results.get("correct", {}),
        "carbon": carbon_metrics,
    }

    if verbose:
        print()
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Samples: {len(test_cases)}")
        if num_params is not None:
            if num_params >= 1_000_000_000:
                print(f"Params:  {num_params:,} ({num_params/1e9:.2f}B)")
            elif num_params >= 1_000_000:
                print(f"Params:  {num_params:,} ({num_params/1e6:.2f}M)")
            else:
                print(f"Params:  {num_params:,} ({num_params/1e3:.1f}K)")
        print()
        print("Accuracy:")
        for m in metrics:
            acc = eval_results[m] * 100
            correct = eval_results.get("correct", {}).get(m, 0)
            print(f"  {m}: {acc:.2f}% ({correct}/{len(test_cases)})")
        print()
        print(f"Duration: {carbon_metrics.get('duration_seconds', duration):.1f}s")
        energy_wh = carbon_metrics.get('energy_wh', 0)
        co2_g = carbon_metrics.get('emissions_g_co2', 0)
        if energy_wh > 0:
            print(f"Energy:   {energy_wh:.4f} Wh")
        if co2_g > 0:
            print(f"CO2:      {co2_g:.4f} g")
        peak_gpu = carbon_metrics.get('peak_gpu_memory_mb', 0)
        if peak_gpu > 0:
            print(f"Peak GPU: {peak_gpu:.1f} MB")
        print("=" * 60)

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        if verbose:
            print(f"\nResults saved to: {output_path}")

    if save_predictions:
        pred_data = []
        for pred, tc in zip(predictions, test_cases):
            pred_data.append({
                "input": tc["reactants"],
                "ground_truth": tc["ground_truth"],
                "predictions": pred.get("predictions", []) if isinstance(pred, dict) else pred,
            })
        with open(save_predictions, 'w') as f:
            json.dump(pred_data, f, indent=2, default=str)
        if verbose:
            print(f"Predictions saved to: {save_predictions}")

    return results


def main():
    parser = argparse.ArgumentParser(description=f"Benchmark Runner for {TASK_NAME}")

    parser.add_argument("--model", type=str, required=True,
                        help=f"Model name: {', '.join(MODELS.keys())}")
    parser.add_argument("--metrics", type=str, nargs="+",
                        help="Metrics to compute (uses task defaults if not specified)")
    parser.add_argument("--limit", type=int, help="Limit number of test samples")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of predictions per sample (default: 5)")
    parser.add_argument("--data", type=str, help="Custom test data path")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    parser.add_argument("--track_carbon", action="store_true", help="Track carbon emissions")
    parser.add_argument("--save_predictions", type=str, help="Path to save raw predictions JSON")

    args = parser.parse_args()

    run_benchmark(
        model_name=args.model,
        limit=args.limit,
        top_k=args.top_k,
        metrics=args.metrics,
        track_carbon=args.track_carbon,
        data_path=args.data,
        output_path=args.output,
        save_predictions=args.save_predictions,
        verbose=True,
    )


if __name__ == "__main__":
    main()
