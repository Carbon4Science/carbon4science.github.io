"""
R-SMILES inference interface for retrosynthesis prediction.

Uses root-aligned SMILES augmentation with an OpenNMT-py Transformer model.
Supports test-time augmentation (TTA) by enumerating root atoms for the
product SMILES, running each through the model, and aggregating predictions
via frequency-weighted scoring.

Reference:
    Zhong et al. "Root-aligned SMILES: A Tight Representation for Chemical
    Reaction Prediction" (Chemical Science, 2022)
"""

import argparse
import io
import os
import re
import random
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

_RSMILES_DIR = Path(__file__).resolve().parent

# SMILES tokenizer pattern (from R-SMILES repo)
_SMI_PATTERN = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)


def _smi_tokenize(smi: str) -> str:
    """Tokenize SMILES into space-separated tokens."""
    tokens = _SMI_PATTERN.findall(smi)
    return " ".join(tokens)


def _smi_detokenize(tokens: str) -> str:
    """Remove spaces from tokenized SMILES."""
    return "".join(tokens.strip().split())


def _canonicalize(smi: str) -> str:
    """Canonicalize SMILES, removing atom mapping."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    for atom in mol.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _enumerate_roots(smi: str, n_augmentations: int) -> List[str]:
    """
    Generate augmented SMILES by enumerating root atoms.

    For test-time augmentation without atom mapping, we enumerate different
    root atoms and generate SMILES starting from each root.

    Args:
        smi: Canonical product SMILES.
        n_augmentations: Number of augmented variants to generate.

    Returns:
        List of SMILES strings (length == n_augmentations).
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return [smi] * n_augmentations

    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        return [smi] * n_augmentations

    # First variant is always canonical (root=-1)
    variants = [Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)]

    if n_augmentations == 1:
        return variants

    # Enumerate root atoms
    available_roots = list(range(n_atoms))
    random.shuffle(available_roots)

    used_smiles = {variants[0]}
    for root in available_roots:
        if len(variants) >= n_augmentations:
            break
        rooted_smi = Chem.MolToSmiles(mol, isomericSmiles=True, rootedAtAtom=root)
        if rooted_smi not in used_smiles:
            variants.append(rooted_smi)
            used_smiles.add(rooted_smi)

    # Pad with repeats if not enough unique roots
    while len(variants) < n_augmentations:
        variants.append(random.choice(variants))

    return variants


def _compute_rank(predictions, augmentation, beam_size, score_alpha=1.0):
    """
    Aggregate predictions from augmented inputs using frequency-weighted scoring.

    Adapted from R-SMILES repo score.py compute_rank().

    Args:
        predictions: List of lists, shape [augmentation][beam_size],
                     each element is (canonical_smiles, max_frag_smiles) tuple.
        augmentation: Number of augmentations used.
        beam_size: Beam size used during translation.
        score_alpha: Weighting factor for rank position.

    Returns:
        List of (smiles, score) tuples sorted by score descending.
    """
    rank = {}
    highest = {}

    if augmentation == 1:
        # No augmentation: use raw ranking
        for k, pred in enumerate(predictions[0]):
            smi = pred[0]  # canonical SMILES
            if smi == "":
                continue
            rank[smi] = 1.0 / (score_alpha * k + 1)
    else:
        for j in range(augmentation):
            # Deduplicate within this augmentation's beam
            seen = set()
            deduped = []
            for pred in predictions[j]:
                if pred[0] != "" and pred[0] not in seen:
                    seen.add(pred[0])
                    deduped.append(pred[0])

            for k, smi in enumerate(deduped):
                if smi in rank:
                    rank[smi] += 1.0 / (score_alpha * k + 1)
                else:
                    rank[smi] = 1.0 / (score_alpha * k + 1)
                if smi in highest:
                    highest[smi] = min(k, highest[smi])
                else:
                    highest[smi] = k

        # Tie-break by best position (lower is better)
        for key in rank:
            rank[key] += highest.get(key, 0) * -1e8

    # Sort by score descending
    sorted_results = sorted(rank.items(), key=lambda x: x[1], reverse=True)
    return sorted_results


def _canonicalize_prediction(smi: str):
    """Canonicalize a predicted SMILES, returning (canonical, max_frag)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ("", "")
    for atom in mol.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")
    try:
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return ("", "")

    # Max fragment
    frags = canonical.split(".")
    if len(frags) > 1:
        frag_mols = [(f, Chem.MolFromSmiles(f)) for f in frags]
        frag_mols = [(f, m) for f, m in frag_mols if m is not None]
        if frag_mols:
            max_frag = max(frag_mols, key=lambda x: x[1].GetNumAtoms())[0]
        else:
            max_frag = canonical
    else:
        max_frag = canonical

    return (canonical, max_frag)


# ── Model state ──────────────────────────────────────────────────────────────

_translator = None
_augmentation_factor = 1
_beam_size = 10
_n_best = 10


def load_model(
    model_path: str,
    augmentation_factor: int = 1,
    beam_size: int = 10,
    **kwargs,
):
    """
    Load the R-SMILES model into memory for fast inference.

    Args:
        model_path: Path to the OpenNMT-py model checkpoint (.pt file).
        augmentation_factor: Number of root-aligned augmentations per input.
                             1 = no augmentation, 20 = full TTA.
        beam_size: Beam size for OpenNMT translation.
    """
    global _translator, _augmentation_factor, _beam_size, _n_best

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    _augmentation_factor = augmentation_factor
    _beam_size = beam_size
    _n_best = beam_size

    # Build OpenNMT-py Translator using Python API (loads model once)
    import onmt.opts as opts

    parser = argparse.ArgumentParser()
    opts.translate_opts(parser)
    opt = parser.parse_args([
        "-model", model_path,
        "-src", "/dev/null",
        "-beam_size", str(beam_size),
        "-n_best", str(beam_size),
        "-batch_size", "8192",
        "-batch_type", "tokens",
        "-max_length", "500",
        "-gpu", "0",
        "-seed", "0",
    ])

    from onmt.translate.translator import build_translator
    _translator = build_translator(opt, report_score=False, out_file=io.StringIO())


def _translate_batch(src_lines: List[str], n_best: int) -> List[List[str]]:
    """
    Translate a batch of tokenized SMILES using the in-memory translator.

    Args:
        src_lines: List of space-tokenized SMILES strings.
        n_best: Number of best translations per input.

    Returns:
        List of lists: for each input, a list of n_best predicted token strings.
    """
    # Write source to temp file (OpenNMT-py translate() reads from file)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, dir="/tmp"
    ) as src_f:
        src_f.write("\n".join(src_lines) + "\n")
        src_path = src_f.name

    try:
        scores, preds = _translator.translate(src=src_path, batch_size=8192)
        # preds is List[List[str]], each inner list is n_best predictions
        # Each prediction is still tokenized (space-separated)
        return preds
    finally:
        if os.path.exists(src_path):
            os.unlink(src_path)


def run(
    smiles: Union[str, List[str]], top_k: int = 10
) -> List[Dict]:
    """
    Run retrosynthesis prediction using R-SMILES.

    Args:
        smiles: Single SMILES string or list of SMILES strings.
        top_k: Number of top predictions to return.

    Returns:
        List of result dicts with uniform interface:
        [{'input': '...', 'predictions': [{'smiles': '...', 'score': 0.95}, ...]}]
    """
    if _translator is None:
        raise RuntimeError(
            "No model loaded. Call load_model(model_path) first."
        )

    if isinstance(smiles, str):
        smiles_list = [smiles]
    else:
        smiles_list = list(smiles)

    aug = _augmentation_factor
    n_best = min(top_k, _n_best)

    # Generate augmented inputs for all molecules
    all_src_lines = []
    for smi in smiles_list:
        cano = _canonicalize(smi)
        variants = _enumerate_roots(cano, aug)
        for v in variants:
            all_src_lines.append(_smi_tokenize(v))

    # Run translation in one batch (model already in memory)
    all_preds = _translate_batch(all_src_lines, n_best=n_best)

    # Parse and aggregate results
    results = []

    for mol_idx, smi in enumerate(smiles_list):
        start = mol_idx * aug
        mol_preds = all_preds[start: start + aug]

        # Organize into [augmentation][beam_size] of (canonical, max_frag) tuples
        predictions_by_aug = []
        for aug_preds in mol_preds:
            canon_preds = []
            for pred_tokens in aug_preds:
                pred_smi = _smi_detokenize(pred_tokens)
                canon = _canonicalize_prediction(pred_smi)
                canon_preds.append(canon)
            predictions_by_aug.append(canon_preds)

        # Aggregate via frequency-weighted scoring
        ranked = _compute_rank(
            predictions_by_aug, aug, n_best, score_alpha=1.0
        )

        # Format output
        formatted_preds = []
        for pred_smi, score in ranked[:top_k]:
            if pred_smi:
                formatted_preds.append({"smiles": pred_smi, "score": float(score)})

        results.append(
            {"input": _canonicalize(smi), "predictions": formatted_preds}
        )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="R-SMILES inference")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--smiles", required=True, help="Input product SMILES")
    parser.add_argument("--augmentation", type=int, default=1, help="Augmentation factor (1 or 20)")
    parser.add_argument("--top_k", type=int, default=10, help="Number of predictions")
    parser.add_argument("--beam_size", type=int, default=10, help="Beam size")
    args = parser.parse_args()

    load_model(args.model_path, augmentation_factor=args.augmentation, beam_size=args.beam_size)
    results = run(args.smiles, top_k=args.top_k)

    for r in results:
        print(f"\nInput: {r['input']}")
        print("Predictions:")
        for i, p in enumerate(r["predictions"], 1):
            print(f"  {i}. {p['smiles']} (score: {p['score']:.4f})")
