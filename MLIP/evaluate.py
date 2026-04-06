"""
MLIP (Machine Learning Interatomic Potentials) evaluation module.

Currently benchmarks computational cost only (speed, carbon).
Accuracy metrics (energy_mae, force_mae, etc.) will be added later.
"""

import os
from typing import Dict, List, Optional

METRICS = ["CPS", "rdf_score", "msd_score"]

# Default test structure
DEFAULT_STRUCTURE = "LGPS_mp-696128.cif"

# Matbench Discovery metrics
CPS_VALUES = {
    "CHGNet": 0.343,
    "DPA3": 0.718,
    "MACE": 0.637,
    "MACE_pruned": 0.637,
    "NequIP": 0.733,
    "Nequix": 0.729,
    "ORB": 0.47,
    "SevenNet": 0.714,
    "eSEN": 0.797,
    "PET": 0.898,
    "eSEN_OAM": 0.888,
    "EquFlash": 0.888,

    "NequIP_OAM": 0.87,
    "Allegro": 0.84,
}
    
F1_VALUES = {
    "CHGNet": 0.613,
    "DPA3": 0.803,
    "MACE": 0.669,
    "MACE_pruned": 0.669,
    "NequIP": 0.761,
    "Nequix": 0.751,
    "ORB": 0.765,
    "SevenNet": 0.76,
    "eSEN": 0.831,
    "PET": 0.924,
    "eSEN_OAM": 0.925,
    "EquFlash": 0.919,

    "NequIP_OAM": 0.893,
    "Allegro": 0.895,
}

# By definition, the original K_SRME is lower better, with maximum being 2.
# To fit between 0 and 1 and make higher better, we use 1 - K_SRME / 2.
K_VALUES = {
    "CHGNet": 0,
    "DPA3": 0.675,
    "MACE": 0.659,
    "NequIP": 0.774,
    "Nequix": 0.777,
    "ORB": 0.137,
    "SevenNet": 0.725,
    "eSEN": 0.83,
    "PET": 0.9405,
    "eSEN_OAM": 0.915,
    "EquFlash": 0.921,

    "NequIP_OAM": 0.917,
    "Allegro": 0.8405,
}

# By definition, the original RMSD is lower better.
# To fit between 0 and 1 and make higher better, we use 1 / (1 + RMSD).
RMSD_VALUES = {
    "CHGNet": 0.913,
    "DPA3": 0.926,
    "MACE": 0.916,
    "NequIP": 0.921,
    "Nequix": 0.922,
    "ORB": 0.908,
    "SevenNet": 0.922,
    "eSEN": 0.93,
    "PET": 0.943,
    "eSEN_OAM": 0.943,
    "EquFlash": 0.943,

    "NequIP_OAM": 0.939,
    "Allegro": 0.939,
}

def load_test_data(data_path: Optional[str] = None, limit: Optional[int] = None) -> List[Dict]:
    """
    Load test structures for MLIP MD benchmarking.

    Each "test case" is a structure to run MD on. Currently uses the single
    LGPS (Li10GeP2S12) structure. The 'product' key is used for pipeline
    compatibility with run_benchmark.py.

    Args:
        data_path: Path to a CIF file. If None, uses the default LGPS structure.
        limit: Maximum number of structures (currently only 1 available).

    Returns:
        List of dicts with 'product' (CIF path) and 'ground_truth' fields.
    """
    if data_path is None:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(this_dir, DEFAULT_STRUCTURE)

    structures = [{"product": data_path, "ground_truth": {}}]

    if limit is not None:
        structures = structures[:limit]

    return structures


def evaluate(
    predictions: List,
    test_cases: List[Dict],
    metrics: Optional[List[str]] = None,
    model_name: Optional[str] = None,
) -> Dict:
    """
    Evaluate MLIP predictions.

    Args:
        predictions: List of prediction results from model.run().
        test_cases: List of test cases with ground truth.
        metrics: List of accuracy metrics to compute.
        model_name: Model name for CPS lookup.

    Returns:
        Dict with accuracy metrics at top level (CPS, rdf_score, msd_score)
        plus 'speed', 'structure', and 'md_params' sub-dicts.
    """
    if metrics is None:
        metrics = METRICS

    results = {}

    # Extract speed, structure, and md_params from predictions
    speed_data = {}
    structure_data = {}
    md_params_data = {}

    # Collect rdf_score and msd_score from predictions (if present)
    pred_rdf_score = 0.0
    pred_msd_score = 0.0

    for pred in predictions:
        for p in pred.get("predictions", []):
            for key in ["steps_per_second", "ns_per_day", "total_steps", "elapsed_seconds"]:
                if key in p:
                    speed_data[key] = p[key]
            if "structure" in p:
                structure_data = p["structure"]
            if "md_params" in p:
                md_params_data = p["md_params"]
            # Extract accuracy scores from production results
            if "rdf_score" in p:
                pred_rdf_score = p["rdf_score"]
            if "msd_score" in p:
                pred_msd_score = p["msd_score"]

    # Build top-level accuracy metrics
    results["CPS"] = CPS_VALUES.get(model_name, 0.0)
    results["rdf_score"] = pred_rdf_score
    results["msd_score"] = pred_msd_score

    results["speed"] = speed_data
    results["structure"] = structure_data
    results["md_params"] = md_params_data

    return results
