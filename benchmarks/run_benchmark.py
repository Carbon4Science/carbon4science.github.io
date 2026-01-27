#!/usr/bin/env python3
"""
Unified Retrosynthesis Benchmark Runner

Usage:
    python run_benchmark.py --model neuralsym --smiles "CCO"
    python run_benchmark.py --model LocalRetro --input test.csv --top_k 10
    python run_benchmark.py --model RetroBridge --smiles "CCO" --track_carbon

All models use the same interface, just change --model argument.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict

# Add parent directory to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

MODELS = {
    "neuralsym": "Retrosynthesis.neuralsym.Inference",
    "LocalRetro": "Retrosynthesis.LocalRetro.Inference",
    "RetroBridge": "Retrosynthesis.RetroBridge.Inference",
    "Chemformer": "Retrosynthesis.Chemformer.Inference",
    "RSGPT": "Retrosynthesis.RSGPT.inference",
}


def get_model(model_name: str):
    """Dynamically import and return the run function for a model."""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")

    module_path = MODELS[model_name]
    module = __import__(module_path, fromlist=["run"])

    # Chemformer requires initialization
    if model_name == "Chemformer":
        print("Note: Chemformer requires load_model() first.")
        print("Call: load_model(model_path, vocabulary_path)")
        return module.run, module.load_model

    return module.run, None


def run_benchmark(
    model_name: str,
    smiles: List[str],
    top_k: int = 10,
    track_carbon: bool = False,
    output_file: str = None,
) -> List[Dict]:
    """
    Run benchmark for specified model.

    Args:
        model_name: One of neuralsym, LocalRetro, RetroBridge, Chemformer, RSGPT
        smiles: List of product SMILES to predict
        top_k: Number of predictions per molecule
        track_carbon: Whether to track carbon emissions
        output_file: Optional file to save results

    Returns:
        List of prediction results in uniform format
    """
    run_func, init_func = get_model(model_name)

    if track_carbon:
        from carbon_tracker import CarbonTracker
        tracker = CarbonTracker(
            project_name=f"{model_name}_benchmark",
            model_name=model_name,
            task="inference"
        )
        with tracker:
            results = run_func(smiles, top_k=top_k)
        tracker.print_summary()
    else:
        results = run_func(smiles, top_k=top_k)

    # Save results if output file specified
    if output_file:
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Unified Retrosynthesis Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmark.py --model neuralsym --smiles "CCO"
  python run_benchmark.py --model LocalRetro --smiles "CCO" "CCCO" --top_k 5
  python run_benchmark.py --model RetroBridge --input molecules.txt --track_carbon
  python run_benchmark.py --list_models
        """
    )
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--smiles", type=str, nargs="+", help="SMILES string(s) to predict")
    parser.add_argument("--input", type=str, help="File with SMILES (one per line)")
    parser.add_argument("--top_k", type=int, default=10, help="Number of predictions")
    parser.add_argument("--track_carbon", action="store_true", help="Track carbon emissions")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--list_models", action="store_true", help="List available models")

    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for name in MODELS:
            print(f"  - {name}")
        return

    if not args.model:
        parser.error("--model is required (or use --list_models)")

    # Get SMILES from args or file
    if args.smiles:
        smiles_list = args.smiles
    elif args.input:
        with open(args.input) as f:
            smiles_list = [line.strip() for line in f if line.strip()]
    else:
        parser.error("Either --smiles or --input is required")

    print(f"Model: {args.model}")
    print(f"Input: {len(smiles_list)} molecules")
    print(f"Top-k: {args.top_k}")
    print("-" * 50)

    results = run_benchmark(
        model_name=args.model,
        smiles=smiles_list,
        top_k=args.top_k,
        track_carbon=args.track_carbon,
        output_file=args.output,
    )

    # Print results
    for r in results:
        print(f"\nInput: {r['input']}")
        print("Predictions:")
        for i, p in enumerate(r['predictions'][:5], 1):
            score = p.get('score', 'N/A')
            score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
            print(f"  {i}. {p['smiles']} (score: {score_str})")


if __name__ == "__main__":
    main()
