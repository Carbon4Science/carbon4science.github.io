#!/usr/bin/env python
"""
Standardized Retrosynthesis Benchmark Evaluation Script

This script provides a unified interface for evaluating retrosynthesis models
with carbon tracking. It supports all models in the Retro/ directory.

Usage:
    python evaluate_retrosynthesis.py --model neuralsym --task inference --runs 3
    python evaluate_retrosynthesis.py --model LocalRetro --task training
    python evaluate_retrosynthesis.py --model all --task inference

Requirements:
    - Model-specific conda environment must be activated
    - CodeCarbon: pip install codecarbon
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.carbon_tracker import CarbonTracker, aggregate_results, create_comparison_table
from Retro.evaluate import evaluate as retro_evaluate


SUPPORTED_MODELS = {
    "neuralsym": {
        "path": "Retro/neuralsym",
        "inference_module": "Inference",
        "inference_func": "run",
        "env": "neuralsym",
        "description": "Template-based Highway-ELU network",
        "requires_init": False,
    },
    "LocalRetro": {
        "path": "Retro/LocalRetro",
        "inference_module": "Inference",
        "inference_func": "run",
        "env": "rdenv",
        "description": "MPNN with global attention",
        "requires_init": False,
    },
    "RetroBridge": {
        "path": "Retro/RetroBridge",
        "inference_module": "Inference",
        "inference_func": "run",
        "env": "retrobridge",
        "description": "Markov bridge generative model",
        "requires_init": False,
    },
    "Chemformer": {
        "path": "Retro/Chemformer",
        "inference_module": "Inference",
        "inference_func": "run",
        "env": "chemformer",
        "description": "BART transformer for SMILES",
        "requires_init": True,  # Needs load_model() first
        "init_func": "load_model",
    },
    "RSGPT": {
        "path": "Retro/RSGPT",
        "inference_module": "inference",
        "inference_func": "run",
        "env": "gpt",
        "description": "GPT-based causal language model",
        "requires_init": False,
    }
}

# Uniform interface: All models now return List[Dict] with format:
# [{'input': str, 'predictions': [{'smiles': str, 'score': float}, ...]}, ...]


def load_test_data(model_name: str) -> Tuple[List[str], List[str]]:
    """
    Load USPTO-50K test data.

    Returns:
        Tuple of (product_smiles_list, ground_truth_reactants_list)
    """
    # Try to find test data in model directory
    model_path = Path(SUPPORTED_MODELS[model_name]["path"])

    # Common test data locations
    possible_paths = [
        model_path / "data" / "USPTO_50K" / "test.csv",
        model_path / "data" / "test.csv",
        model_path / "data" / "50k_test.csv",
        Path("data") / "USPTO_50K" / "test.csv",
    ]

    for path in possible_paths:
        if path.exists():
            print(f"Loading test data from: {path}")
            import pandas as pd
            df = pd.read_csv(path)

            # Handle different column naming conventions
            product_col = None
            reactant_col = None

            for col in ["product", "products", "prod", "product_smiles"]:
                if col in df.columns:
                    product_col = col
                    break

            for col in ["reactant", "reactants", "rxn", "reactant_smiles", "precursors"]:
                if col in df.columns:
                    reactant_col = col
                    break

            if product_col and reactant_col:
                return df[product_col].tolist(), df[reactant_col].tolist()

    raise FileNotFoundError(
        f"Could not find test data for {model_name}. "
        f"Please ensure USPTO-50K test data is available."
    )


def run_inference_benchmark(
    model_name: str,
    run_id: int = 1,
    batch_size: int = 32,
    max_samples: Optional[int] = None
) -> Dict:
    """
    Run inference benchmark for a single model.

    Args:
        model_name: Name of the model to benchmark
        run_id: Run identifier for multiple trials
        batch_size: Batch size for inference
        max_samples: Maximum samples to evaluate (None = all)

    Returns:
        Benchmark metrics dictionary
    """
    print(f"\n{'='*60}")
    print(f"Running inference benchmark: {model_name} (Run {run_id})")
    print(f"{'='*60}")

    model_config = SUPPORTED_MODELS[model_name]

    # Import model's inference function
    model_path = Path(model_config["path"])
    sys.path.insert(0, str(model_path))

    try:
        inference_module = __import__(model_config["inference_module"])
        run_func = getattr(inference_module, model_config["inference_func"])

        # Handle models that require initialization (e.g., Chemformer)
        if model_config.get("requires_init", False):
            init_func = getattr(inference_module, model_config["init_func"])
            print(f"Note: {model_name} requires initialization. Call {model_config['init_func']}() first.")
    except ImportError as e:
        print(f"Error: Could not import {model_name} inference module.")
        print(f"Make sure you're in the correct conda environment: {model_config['env']}")
        print(f"Error details: {e}")
        return {}

    # Load test data
    try:
        products, ground_truth = load_test_data(model_name)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return {}

    if max_samples:
        products = products[:max_samples]
        ground_truth = ground_truth[:max_samples]

    print(f"Test samples: {len(products)}")
    print(f"Batch size: {batch_size}")

    # Initialize tracker
    tracker = CarbonTracker(
        project_name=f"{model_name}_inference_run{run_id}",
        model_name=model_name,
        task="inference"
    )

    # Run inference with tracking
    all_results = []
    with tracker:
        for i in range(0, len(products), batch_size):
            batch = products[i:i+batch_size]
            try:
                # All models now use uniform interface: run(smiles, top_k=N)
                batch_results = run_func(batch, top_k=10)
                all_results.extend(batch_results)
            except Exception as e:
                print(f"Warning: Batch {i} failed: {e}")
                # Create empty results for failed batch
                for smi in batch:
                    all_results.append({'input': smi, 'predictions': []})

            # Progress update
            if (i + batch_size) % 500 == 0:
                print(f"  Processed {min(i + batch_size, len(products))}/{len(products)}")

    # Calculate accuracy using canonical evaluator from Retro.evaluate
    test_cases = [{'ground_truth': gt} for gt in ground_truth]
    accuracy = retro_evaluate(all_results, test_cases)
    print(f"\nAccuracy: Top-1={accuracy['top_1']:.2%}, Top-10={accuracy['top_10']:.2%}")

    # Add metrics
    tracker.add_accuracy(
        top1=accuracy.get("top_1"),
        top5=accuracy.get("top_5"),
        top10=accuracy.get("top_10"),
        num_samples=len(products),
        batch_size=batch_size
    )

    tracker.print_summary()
    return tracker.get_metrics()


def run_all_benchmarks(
    models: List[str],
    task: str = "inference",
    runs: int = 3
) -> None:
    """
    Run benchmarks for multiple models.

    Args:
        models: List of model names to benchmark
        task: "inference" or "training"
        runs: Number of runs per model (for inference)
    """
    all_results = []

    for model_name in models:
        if model_name not in SUPPORTED_MODELS:
            print(f"Warning: Unknown model '{model_name}', skipping.")
            continue

        if task == "inference":
            for run_id in range(1, runs + 1):
                try:
                    result = run_inference_benchmark(model_name, run_id)
                    if result:
                        all_results.append(result)
                except Exception as e:
                    print(f"Error benchmarking {model_name} run {run_id}: {e}")
        else:
            print(f"Training benchmark for {model_name} - use model-specific scripts")

    # Print comparison
    if all_results:
        print("\n" + "="*60)
        print("BENCHMARK COMPARISON")
        print("="*60)
        print(create_comparison_table(all_results))


def main():
    parser = argparse.ArgumentParser(
        description="Standardized Retrosynthesis Benchmark Evaluation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help=f"Model to benchmark. Options: {', '.join(SUPPORTED_MODELS.keys())}, all"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["inference", "training"],
        default="inference",
        help="Benchmark type"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of inference runs (for computing mean/std)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List supported models and exit"
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate and display existing results"
    )

    args = parser.parse_args()

    if args.list_models:
        print("\nSupported Models:")
        print("-" * 60)
        for name, config in SUPPORTED_MODELS.items():
            print(f"  {name:15} - {config['description']}")
            print(f"                  Environment: {config['env']}")
        return

    if args.aggregate:
        results = aggregate_results()
        if results:
            print(create_comparison_table(results))
        else:
            print("No results found in benchmarks/results/")
        return

    # Determine which models to run
    if args.model.lower() == "all":
        models = list(SUPPORTED_MODELS.keys())
    else:
        models = [args.model]

    # Run benchmarks
    if args.task == "inference":
        for model in models:
            for run_id in range(1, args.runs + 1):
                try:
                    run_inference_benchmark(
                        model,
                        run_id=run_id,
                        batch_size=args.batch_size,
                        max_samples=args.max_samples
                    )
                except Exception as e:
                    print(f"Error: {e}")
    else:
        print("For training benchmarks, use the model-specific training scripts")
        print("with CarbonTracker wrapped around the training loop.")


if __name__ == "__main__":
    main()
