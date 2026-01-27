#!/usr/bin/env python3
"""
Unified Benchmark Runner for Carbon4Science

Dynamically loads task-specific evaluation modules from each task directory.
Each task (Retrosynthesis, MolGen, MatGen) defines its own:
  - METRICS: Available metrics for the task
  - load_test_data(): Load test dataset
  - evaluate(): Compute metrics from predictions

Usage:
    python run_benchmark.py --task Retrosynthesis --model LocalRetro --track_carbon
    python run_benchmark.py --task Retrosynthesis --model neuralsym --limit 500
"""

import argparse
import importlib
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Task to evaluation module mapping
TASKS = {
    "Retrosynthesis": {
        "eval_module": "Retrosynthesis.evaluate",
        "models": {
            "neuralsym": "Retrosynthesis.neuralsym.Inference",
            "LocalRetro": "Retrosynthesis.LocalRetro.Inference",
            "RetroBridge": "Retrosynthesis.RetroBridge.Inference",
            "Chemformer": "Retrosynthesis.Chemformer.Inference",
            "RSGPT": "Retrosynthesis.RSGPT.inference",
        }
    },
    "MolGen": {
        "eval_module": "MolGen.evaluate",
        "models": {}  # To be added
    },
    "MatGen": {
        "eval_module": "MatGen.evaluate",
        "models": {}  # To be added
    },
}


def get_task_evaluator(task_name: str):
    """Load the task-specific evaluation module."""
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASKS.keys())}")

    eval_module_name = TASKS[task_name]["eval_module"]
    try:
        eval_module = importlib.import_module(eval_module_name)
        return eval_module
    except ImportError as e:
        raise ImportError(f"Could not load evaluation module for {task_name}: {e}")


def get_model_run_func(task_name: str, model_name: str):
    """Load the model's run function."""
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name}")

    models = TASKS[task_name]["models"]
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available for {task_name}: {list(models.keys())}")

    module_name = models[model_name]
    module = importlib.import_module(module_name)
    return module.run


def run_benchmark(
    task_name: str,
    model_name: str,
    limit: Optional[int] = None,
    top_k: int = 50,
    metrics: Optional[List[str]] = None,
    track_carbon: bool = False,
    data_path: Optional[str] = None,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run benchmark for a model on a task.

    Args:
        task_name: Task name (Retrosynthesis, MolGen, MatGen)
        model_name: Model name within the task
        limit: Limit number of test samples (optional)
        top_k: Number of predictions per sample
        metrics: Metrics to compute (uses task defaults if None)
        track_carbon: Whether to track carbon emissions
        data_path: Custom test data path (optional)
        output_path: Path to save results JSON (optional)
        verbose: Print progress updates

    Returns:
        Results dictionary with accuracy and carbon metrics
    """
    # Load task-specific evaluator
    evaluator = get_task_evaluator(task_name)
    model_run = get_model_run_func(task_name, model_name)

    # Use task's default metrics if not specified
    if metrics is None:
        metrics = evaluator.METRICS

    # Print config
    if verbose:
        print("=" * 60)
        print(f"Task:    {task_name}")
        print(f"Model:   {model_name}")
        print(f"Metrics: {metrics}")
        print(f"Top-k:   {top_k}")
        print(f"Limit:   {limit or 'Full dataset'}")
        print(f"Carbon:  {'Yes' if track_carbon else 'No'}")
        print("=" * 60)
        print()

    # Load test data
    if verbose:
        print("Loading test data...")
    test_cases = evaluator.load_test_data(data_path=data_path, limit=limit)
    if verbose:
        print(f"Loaded {len(test_cases)} test cases")
        print()

    # Setup carbon tracking
    if track_carbon:
        from carbon_tracker import CarbonTracker
        tracker = CarbonTracker(
            project_name=f"{model_name}_{task_name}_benchmark",
            model_name=model_name,
            task="inference",
            save_results=False  # We'll save manually with full results
        )
    else:
        tracker = None

    # Run inference
    if verbose:
        print(f"Running inference (top_k={top_k})...")
        sys.stdout.flush()

    predictions = []
    start_time = time.time()

    if tracker:
        tracker.start()

    for i, tc in enumerate(test_cases):
        try:
            result = model_run(tc['product'], top_k=top_k)
            predictions.append(result[0] if result else {'input': tc['product'], 'predictions': []})
        except Exception as e:
            predictions.append({'input': tc['product'], 'predictions': []})

        if verbose and (i + 1) % 100 == 0:
            # Compute intermediate accuracy
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

    # Evaluate predictions
    if verbose:
        print()
        print("Evaluating predictions...")
    eval_results = evaluator.evaluate(predictions, test_cases, metrics)

    # Compile final results
    results = {
        "task": task_name,
        "model": model_name,
        "num_samples": len(test_cases),
        "top_k": top_k,
        "metrics": metrics,
        "accuracy": {m: eval_results[m] for m in metrics},
        "correct": eval_results.get("correct", {}),
        "carbon": carbon_metrics,
    }

    # Print results
    if verbose:
        print()
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Samples: {len(test_cases)}")
        print()
        print("Accuracy:")
        for m in metrics:
            acc = eval_results[m] * 100
            correct = eval_results.get("correct", {}).get(m, 0)
            print(f"  {m}: {acc:.2f}% ({correct}/{len(test_cases)})")
        print()
        print(f"Duration: {carbon_metrics.get('duration_seconds', duration):.1f}s")
        if track_carbon:
            energy = carbon_metrics.get('energy_kwh', 0) * 1000
            co2 = carbon_metrics.get('emissions_kg_co2', 0) * 1000
            print(f"Energy:   {energy:.4f} Wh")
            print(f"CO2:      {co2:.4f} g")
        print("=" * 60)

    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        if verbose:
            print(f"\nResults saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Unified Benchmark Runner for Carbon4Science",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Core arguments
    parser.add_argument("--task", type=str, default="Retrosynthesis",
                        help=f"Task: {', '.join(TASKS.keys())} (default: Retrosynthesis)")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name within the task")
    parser.add_argument("--metrics", type=str, nargs="+",
                        help="Metrics to compute (uses task defaults if not specified)")

    # Data options
    parser.add_argument("--limit", type=int,
                        help="Limit number of test samples")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Number of predictions per sample (default: 50)")
    parser.add_argument("--data", type=str,
                        help="Custom test data path")

    # Output options
    parser.add_argument("--output", type=str,
                        help="Output JSON file for results")
    parser.add_argument("--track_carbon", action="store_true",
                        help="Track carbon emissions")

    # Info options
    parser.add_argument("--list_tasks", action="store_true",
                        help="List available tasks")
    parser.add_argument("--list_models", action="store_true",
                        help="List available models for a task")
    parser.add_argument("--list_metrics", action="store_true",
                        help="List available metrics for a task")

    args = parser.parse_args()

    # List options
    if args.list_tasks:
        print("Available tasks:")
        for task in TASKS:
            print(f"  {task}")
        return

    if args.list_models:
        if args.task not in TASKS:
            print(f"Unknown task: {args.task}")
            return
        print(f"Available models for {args.task}:")
        for model in TASKS[args.task]["models"]:
            print(f"  {model}")
        return

    if args.list_metrics:
        try:
            evaluator = get_task_evaluator(args.task)
            print(f"Available metrics for {args.task}:")
            for m in evaluator.METRICS:
                print(f"  {m}")
        except ImportError as e:
            print(f"Could not load evaluator for {args.task}: {e}")
        return

    # Run benchmark
    run_benchmark(
        task_name=args.task,
        model_name=args.model,
        limit=args.limit,
        top_k=args.top_k,
        metrics=args.metrics,
        track_carbon=args.track_carbon,
        data_path=args.data,
        output_path=args.output,
        verbose=True,
    )


if __name__ == "__main__":
    main()
