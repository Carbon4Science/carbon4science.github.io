"""
ExampleTask evaluation module.

Each task defines its own evaluate.py with:
- METRICS: list of available metrics
- load_test_data(data_path, limit): load test dataset
- evaluate(predictions, test_cases, metrics): compute metrics

Copy this file and adapt for your task.
"""

import csv
import os
from typing import Dict, List, Optional

# Available metrics for this task
METRICS = ["top_1", "top_5"]

# Default test data location (relative to this file)
DEFAULT_TEST_DATA = "data/test.csv"


def load_test_data(data_path: Optional[str] = None, limit: Optional[int] = None) -> List[Dict]:
    """
    Load test data.

    Args:
        data_path: Path to test data file. If None, uses default location.
        limit: Maximum number of test cases to load.

    Returns:
        List of dicts with task-specific keys (e.g. 'input' and 'ground_truth').
    """
    if data_path is None:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(this_dir, DEFAULT_TEST_DATA)

    test_cases = []

    with open(data_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_cases.append({
                "input": row["input"],
                "ground_truth": row["ground_truth"],
            })
            if limit and len(test_cases) >= limit:
                break

    return test_cases


def evaluate(
    predictions: List,
    test_cases: List[Dict],
    metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Evaluate predictions against ground truth.

    Args:
        predictions: List of prediction results from model.run().
        test_cases: List of test cases with 'ground_truth' key.
        metrics: List of metrics to compute. If None, computes all.

    Returns:
        Dict mapping metric names to scores (0.0 to 1.0).
    """
    if metrics is None:
        metrics = METRICS

    k_values = {"top_1": 1, "top_5": 5}
    correct_counts = {m: 0 for m in metrics}

    for pred, test_case in zip(predictions, test_cases):
        gt = test_case["ground_truth"]

        pred_list = pred
        if isinstance(pred, dict):
            pred_list = pred.get("predictions", [])

        pred_values = []
        for p in pred_list:
            if isinstance(p, dict):
                pred_values.append(p.get("smiles", ""))
            elif isinstance(p, str):
                pred_values.append(p)

        for metric in metrics:
            k = k_values[metric]
            if gt in pred_values[:k]:
                correct_counts[metric] += 1

    n = len(test_cases)
    results = {m: correct_counts[m] / n if n > 0 else 0.0 for m in metrics}
    results["correct"] = {m: correct_counts[m] for m in metrics}
    return results
