"""
ExampleModel inference interface.

Every model MUST expose a run() function with this exact signature.
The benchmark runner depends on this contract.

Usage:
    from ExampleTask.ExampleModel.Inference import run
    results = run("input_string", top_k=10)
"""

import os
import sys
from typing import Dict, List, Union

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_model = None


def _get_model():
    """Lazy-load the model on first use."""
    global _model
    if _model is None:
        # Load your model here. Example:
        #   import torch
        #   _model = torch.load(os.path.join(_ROOT_DIR, "checkpoint.pt"))
        _model = "placeholder"
    return _model


def run(input_data: Union[str, List[str]], top_k: int = 10) -> List[Dict]:
    """
    Run inference on input data.

    Args:
        input_data: A single input string or list of input strings.
        top_k: Number of top predictions to return (default: 10).

    Returns:
        List of result dicts, one per input. Each dict contains:
            - 'input': the input string
            - 'predictions': list of dicts with 'smiles' and 'score'

        Example:
            [{'input': 'CCO',
              'predictions': [{'smiles': 'C=C.O', 'score': 0.95},
                              {'smiles': 'CC.O', 'score': 0.80}]}]
    """
    model = _get_model()

    if isinstance(input_data, str):
        input_list = [input_data]
    else:
        input_list = list(input_data)

    results = []
    for inp in input_list:
        # Replace this with your actual inference logic
        predictions = []
        # predictions = model.predict(inp, top_k=top_k)

        results.append({
            "input": inp,
            "predictions": predictions[:top_k],
        })

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python Inference.py <input> [top_k]")
        sys.exit(1)

    input_data = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    results = run(input_data, top_k=top_k)
    for r in results:
        print(f"Input: {r['input']}")
        for i, p in enumerate(r["predictions"], 1):
            print(f"  {i}. {p['smiles']} (score: {p['score']:.4f})")
