"""
Molecule Generation (MolGen) evaluation module.

Metrics:
- validity: Fraction of valid SMILES strings
- uniqueness: Fraction of unique molecules among valid ones
- novelty: Fraction of molecules not in training set
- diversity: Internal Tanimoto diversity (1 - avg similarity)
"""

from typing import Dict, List, Optional, Set

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

# Available metrics for this task
METRICS = ["validity", "uniqueness", "novelty", "diversity"]


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Canonicalize SMILES string. Returns None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def compute_fingerprint(smiles: str):
    """Compute Morgan fingerprint for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def evaluate(
    generated_smiles: List[str],
    reference_smiles: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Evaluate generated molecules.

    Args:
        generated_smiles: List of generated SMILES strings.
        reference_smiles: Training set SMILES for novelty calculation.
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

    results = {}
    n_total = len(generated_smiles)

    if n_total == 0:
        return {m: 0.0 for m in metrics}

    # Canonicalize and filter valid molecules
    valid_smiles = []
    for smiles in generated_smiles:
        canonical = canonicalize_smiles(smiles)
        if canonical is not None:
            valid_smiles.append(canonical)

    n_valid = len(valid_smiles)

    # Validity
    if "validity" in metrics:
        results["validity"] = n_valid / n_total

    # Uniqueness
    unique_smiles = list(set(valid_smiles))
    n_unique = len(unique_smiles)

    if "uniqueness" in metrics:
        results["uniqueness"] = n_unique / n_valid if n_valid > 0 else 0.0

    # Novelty
    if "novelty" in metrics:
        if reference_smiles is None:
            results["novelty"] = 1.0  # Assume all novel if no reference
        else:
            ref_set: Set[str] = set()
            for smiles in reference_smiles:
                canonical = canonicalize_smiles(smiles)
                if canonical:
                    ref_set.add(canonical)

            novel_count = sum(1 for s in unique_smiles if s not in ref_set)
            results["novelty"] = novel_count / n_unique if n_unique > 0 else 0.0

    # Diversity (internal Tanimoto diversity)
    if "diversity" in metrics:
        if n_unique < 2:
            results["diversity"] = 0.0
        else:
            fps = []
            for smiles in unique_smiles:
                fp = compute_fingerprint(smiles)
                if fp is not None:
                    fps.append(fp)

            if len(fps) < 2:
                results["diversity"] = 0.0
            else:
                # Compute pairwise similarities
                similarities = []
                for i in range(len(fps)):
                    for j in range(i + 1, len(fps)):
                        sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                        similarities.append(sim)

                avg_similarity = sum(similarities) / len(similarities)
                results["diversity"] = 1.0 - avg_similarity

    return results
