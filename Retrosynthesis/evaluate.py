"""
Retrosynthesis evaluation module.

Metrics:
- top_1: Exact match at rank 1
- top_5: Correct answer in top 5 predictions
- top_10: Correct answer in top 10 predictions
- top_50: Correct answer in top 50 predictions
"""

import os
import pickle
from typing import Dict, List, Optional

from rdkit import Chem

# Available metrics for this task
METRICS = ["top_1", "top_5", "top_10", "top_50"]

# Default test data location (relative to this file)
DEFAULT_TEST_DATA = "data/USPTO_50K_test.pickle"


def remove_atom_mapping(smiles: str) -> str:
    """Remove atom mapping from SMILES and return canonical form."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True)


def canonicalize_smiles(smiles: str) -> str:
    """Canonicalize SMILES string, removing atom mapping."""
    return remove_atom_mapping(smiles)


def load_test_data(data_path: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, str]]:
    """
    Load test data for retrosynthesis evaluation.

    Args:
        data_path: Path to test data file. If None, uses default location.
        limit: Maximum number of test cases to load.

    Returns:
        List of dicts with 'product' and 'ground_truth' keys.
    """
    if data_path is None:
        # Use default path relative to this file
        this_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(this_dir, DEFAULT_TEST_DATA)

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    test_cases = []
    for idx, row in data.iterrows():
        # Expected columns: product_smiles, reactant_smiles (or similar)
        product = row.get('product_smiles', row.get('product', ''))
        ground_truth = row.get('reactant_smiles', row.get('reactants', ''))

        if product and ground_truth:
            test_cases.append({
                'product': product,
                'ground_truth': ground_truth
            })

        if limit and len(test_cases) >= limit:
            break

    return test_cases


def evaluate(
    predictions: List,
    test_cases: List[Dict[str, str]],
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Evaluate retrosynthesis predictions.

    Args:
        predictions: List of prediction results from model.run().
                    Each element should be a list of dicts with 'smiles' key,
                    or a list of SMILES strings.
        test_cases: List of test cases with 'ground_truth' key.
        metrics: List of metrics to compute. If None, computes all.

    Returns:
        Dict mapping metric names to scores (0.0 to 1.0).
    """
    if metrics is None:
        metrics = METRICS

    # Validate metrics
    for m in metrics:
        if m not in METRICS:
            raise ValueError(f"Unknown metric: {m}. Available: {METRICS}")

    results = {m: 0.0 for m in metrics}
    correct_counts = {m: 0 for m in metrics}

    k_values = {
        'top_1': 1,
        'top_5': 5,
        'top_10': 10,
        'top_50': 50
    }

    for pred, test_case in zip(predictions, test_cases):
        # Canonicalize ground truth
        gt = canonicalize_smiles(test_case['ground_truth'])

        # Extract SMILES from predictions
        pred_smiles = []
        if isinstance(pred, list):
            for p in pred:
                if isinstance(p, dict):
                    smiles = p.get('smiles', p.get('precursors', ''))
                    if isinstance(smiles, list):
                        smiles = '.'.join(smiles)
                    pred_smiles.append(smiles)
                elif isinstance(p, str):
                    pred_smiles.append(p)

        # Canonicalize predictions
        pred_canonical = [canonicalize_smiles(s) for s in pred_smiles]

        # Check each metric
        for metric in metrics:
            k = k_values[metric]
            top_k_preds = pred_canonical[:k]
            if gt in top_k_preds:
                correct_counts[metric] += 1

    # Calculate accuracy
    n = len(test_cases)
    for metric in metrics:
        results[metric] = correct_counts[metric] / n if n > 0 else 0.0

    return results
