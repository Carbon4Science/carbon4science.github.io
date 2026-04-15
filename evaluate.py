"""
Retrosynthesis evaluation module.

Metrics:
- top_1: Exact match at rank 1
- top_5: Correct answer in top 5 predictions
- top_10: Correct answer in top 10 predictions
- top_50: Correct answer in top 50 predictions
"""

import csv
import os
import pickle
from typing import Dict, List, Optional

from rdkit import Chem

# Available metrics for this task
METRICS = ["top_1", "top_5", "top_10", "top_50"]

# Default test data location (relative to this file)
DEFAULT_TEST_DATA = "data/test_demapped.csv"


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

    test_cases = []

    if data_path.endswith('.csv'):
        # CSV format: "id,class,reactants>reagents>production"
        with open(data_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rxn_col = row.get('reactants>reagents>production', '')
                if rxn_col:
                    # Split by '>' to get [reactants, reagents, product]
                    parts = rxn_col.split('>')
                    if len(parts) == 3:
                        reactants, reagents, product = parts
                    elif '>>' in rxn_col:
                        reactants, product = rxn_col.split('>>', 1)
                    else:
                        continue
                    if product and reactants:
                        test_cases.append({
                            'product': product.strip(),
                            'ground_truth': reactants.strip()
                        })
                if limit and len(test_cases) >= limit:
                    break
    elif data_path.endswith('.pickle') or data_path.endswith('.pkl'):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, list):
            # Data is a list of reaction SMILES strings: "reactants>>product"
            for rxn_smiles in data:
                if '>>' not in rxn_smiles:
                    continue
                reactants, product = rxn_smiles.split('>>', 1)
                if product and reactants:
                    test_cases.append({
                        'product': product,
                        'ground_truth': reactants
                    })
                if limit and len(test_cases) >= limit:
                    break
        else:
            # Data is a DataFrame
            # Check for Mol-object columns (e.g. Chemformer pickle with reactants_mol/products_mol)
            if 'products_mol' in data.columns and 'reactants_mol' in data.columns:
                # Filter to test set if 'set' column exists
                if 'set' in data.columns:
                    data = data[data['set'] == 'test']

                for idx, row in data.iterrows():
                    prod_mol = row['products_mol']
                    react_mol = row['reactants_mol']
                    if prod_mol is None or react_mol is None:
                        continue
                    product = Chem.MolToSmiles(prod_mol, canonical=True)
                    ground_truth = Chem.MolToSmiles(react_mol, canonical=True)
                    if product and ground_truth:
                        test_cases.append({
                            'product': product,
                            'ground_truth': canonicalize_smiles(ground_truth),
                        })
                    if limit and len(test_cases) >= limit:
                        break
            else:
                # Standard DataFrame with SMILES string columns
                for idx, row in data.iterrows():
                    product = row.get('product_smiles', row.get('product', ''))
                    ground_truth = row.get('reactant_smiles', row.get('reactants', ''))
                    if product and ground_truth:
                        test_cases.append({
                            'product': product,
                            'ground_truth': ground_truth
                        })
                    if limit and len(test_cases) >= limit:
                        break
    else:
        raise ValueError(f"Unsupported file format: {data_path}")

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
        gt = test_case['ground_truth']

        # Extract SMILES from predictions
        pred_smiles = []
        # Handle dict format from run_benchmark.py: {'input': ..., 'predictions': [...]}
        pred_list = pred
        if isinstance(pred, dict):
            pred_list = pred.get('predictions', [])
        if isinstance(pred_list, list):
            for p in pred_list:
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

    # Include raw correct counts for reporting
    results["correct"] = {m: correct_counts[m] for m in metrics}

    return results
