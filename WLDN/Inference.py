"""
WLDN (rexgen_direct) forward reaction prediction inference.

Reference: Coley, Jin, Rogers & Jamison, "A graph-convolutional neural network
model for the prediction of chemical reactivity", Chem. Sci., 2019.

GitHub: https://github.com/connorcoley/rexgen_direct

Two-stage approach:
  Stage 1 (CoreFinder): Identify candidate bond changes in reactants
  Stage 2 (CandRanker): Enumerate and rank product candidates

Implements the uniform inference interface for the benchmark runner.

Usage:
    from Forward.WLDN.Inference import run
    results = run("CCO.CC(=O)Cl", top_k=5)
    results = run(["CCO.CC(=O)Cl", "c1ccccc1.Cl"], top_k=5)
"""

import os
import sys
import warnings
from typing import List, Dict, Union

# Suppress TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.join(_ROOT_DIR, "repo")
_REXGEN_DIR = os.path.join(_REPO_DIR, "rexgen_direct")

# Global model holders (lazy init)
_corefinder = None
_candranker = None
_edit_mol_func = None


def _setup_paths():
    """Add necessary paths for rexgen_direct imports."""
    paths_to_add = [
        _REPO_DIR,
        _REXGEN_DIR,
        os.path.join(_REXGEN_DIR, "core_wln_global"),
        os.path.join(_REXGEN_DIR, "rank_diff_wln"),
    ]
    for p in paths_to_add:
        if p not in sys.path:
            sys.path.insert(0, p)


def _get_models():
    """Lazy initialization of both CoreFinder and CandRanker models."""
    global _corefinder, _candranker, _edit_mol_func

    if _corefinder is not None and _candranker is not None:
        return _corefinder, _candranker, _edit_mol_func

    _setup_paths()

    # Suppress TF1 deprecation warnings
    import tensorflow as tf
    if hasattr(tf, "logging"):
        tf.logging.set_verbosity(tf.logging.ERROR)
    elif hasattr(tf, "compat"):
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    from rexgen_direct.core_wln_global.directcorefinder import DirectCoreFinder
    from rexgen_direct.scripts.eval_by_smiles import edit_mol

    # IMPORTANT: directcandranker.py uses edit_mol() inside its predict() method
    # but only imports it in its __main__ block. We must inject edit_mol into
    # the directcandranker module's namespace before using it.
    import rexgen_direct.rank_diff_wln.directcandranker as _dcr_module
    _dcr_module.edit_mol = edit_mol
    from rexgen_direct.rank_diff_wln.directcandranker import DirectCandRanker

    print("[WLDN] Loading CoreFinder model (Stage 1)...")
    _corefinder = DirectCoreFinder()
    _corefinder.load_model()
    print("[WLDN] CoreFinder loaded.")

    print("[WLDN] Loading CandRanker model (Stage 2)...")
    _candranker = DirectCandRanker()
    _candranker.load_model()
    print("[WLDN] CandRanker loaded.")

    _edit_mol_func = edit_mol

    return _corefinder, _candranker, _edit_mol_func


def _count_parameters():
    """Count total trainable parameters across both TF graphs."""
    import tensorflow as tf

    total = 0
    corefinder, candranker, _ = _get_models()

    with corefinder.graph.as_default():
        total += sum(
            int(v.shape.num_elements())
            for v in tf.trainable_variables()
        )
    with candranker.graph.as_default():
        total += sum(
            int(v.shape.num_elements())
            for v in tf.trainable_variables()
        )
    return total


def _predict_single(reactant_smi: str, top_k: int = 5) -> Dict:
    """
    Run two-stage WLDN prediction for a single reactant SMILES.

    Args:
        reactant_smi: Reactant SMILES (dot-separated for multiple reactants).
                       Atom mapping is optional; if absent, sequential mapping
                       is assigned automatically by the CoreFinder.
        top_k: Number of top product predictions to return.

    Returns:
        Dict with 'input' and 'predictions' keys.
    """
    corefinder, candranker, edit_mol = _get_models()

    try:
        # Stage 1: Identify candidate bond changes
        # DirectCoreFinder.predict() handles unmapped SMILES by assigning
        # sequential atom map numbers if molAtomMapNumber is missing.
        react_mapped, bond_preds, bond_scores, _ = corefinder.predict(reactant_smi)

        # Stage 2: Rank candidate products
        # Returns list of dicts with 'rank', 'smiles', 'score', 'prob'
        outcomes = candranker.predict(
            react_mapped, bond_preds, bond_scores, top_n=top_k
        )

        predictions = []
        for outcome in outcomes[:top_k]:
            smi = outcome.get("smiles", "")
            # edit_mol returns a list of fragment SMILES; join them
            if isinstance(smi, list):
                smi = ".".join(smi)
            score = float(outcome.get("prob", outcome.get("score", 0.0)))
            predictions.append({"smiles": smi, "score": score})

        return {"input": reactant_smi, "predictions": predictions}

    except Exception as e:
        warnings.warn(f"[WLDN] Prediction failed for '{reactant_smi}': {e}")
        return {"input": reactant_smi, "predictions": []}


def run(input_data: Union[str, List[str]], top_k: int = 10) -> List[Dict]:
    """
    Uniform inference interface for WLDN forward reaction prediction.

    Args:
        input_data: A single SMILES string or list of SMILES strings.
                    Each SMILES represents the reactant(s) for a reaction
                    (multiple reactants separated by '.').
                    Atom mapping is optional.
        top_k: Number of top predictions to return per input.

    Returns:
        List of dicts, each with:
            'input': the input SMILES string
            'predictions': list of dicts with 'smiles' and 'score' keys,
                           sorted by descending score
    """
    if isinstance(input_data, str):
        input_data = [input_data]

    # Ensure models are loaded
    _get_models()

    results = []
    for i, smi in enumerate(input_data):
        smi = smi.strip()
        if not smi:
            results.append({"input": smi, "predictions": []})
            continue

        result = _predict_single(smi, top_k=top_k)
        results.append(result)

        if (i + 1) % 100 == 0:
            print(f"[WLDN] Processed {i + 1}/{len(input_data)} molecules")

    return results


if __name__ == "__main__":
    # Quick test with the example from the repo
    test_smi = (
        "[CH3:26][c:27]1[cH:28][cH:29][cH:30][cH:31][cH:32]1."
        "[Cl:18][C:19](=[O:20])[O:21][C:22]([Cl:23])([Cl:24])[Cl:25]."
        "[NH2:1][c:2]1[cH:3][cH:4][c:5]([Br:17])[c:6]2[c:10]1"
        "[O:9][C:8]([CH3:11])([C:12](=[O:13])[O:14][CH2:15][CH3:16])[CH2:7]2"
    )
    print(f"Input: {test_smi[:80]}...")
    results = run(test_smi, top_k=5)
    for r in results:
        print(f"\nInput: {r['input'][:80]}...")
        for pred in r["predictions"]:
            print(f"  {pred['smiles']} (score: {pred['score']:.4f})")
