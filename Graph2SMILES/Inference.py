"""
Graph2SMILES Inference Module for Forward Reaction Prediction.

Uniform interface for Carbon4Science benchmarking.
Model: Graph2SMILES (DGCN variant) on USPTO_480k.
Paper: "Graph2SMILES: An End-to-End Approach for Reaction Prediction" (ICML 2021 WS / JCIM 2022)
"""

import os
import sys
import logging
import numpy as np
import torch
from typing import List, Dict, Union

# ---------------------------------------------------------------------------
# Path setup: add the cloned repo to sys.path so we can import its modules
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.join(_THIS_DIR, "repo")

if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

# ---------------------------------------------------------------------------
# Lazy-loaded globals (populated on first call to run())
# ---------------------------------------------------------------------------
_model = None
_vocab = None
_vocab_tokens = None
_device = None
_pretrain_args = None

# ---------------------------------------------------------------------------
# Default paths (relative to this file)
# ---------------------------------------------------------------------------
_CHECKPOINT_PATH = os.path.join(_REPO_DIR, "checkpoints", "pretrained", "USPTO_480k_dgcn.pt")
_VOCAB_PATH = os.path.join(_REPO_DIR, "preprocessed", "default_vocab_smiles.txt")


def _load_model(device: str = "cuda:0"):
    """Load the pretrained Graph2SMILES model (DGCN, USPTO_480k)."""
    global _model, _vocab, _vocab_tokens, _device, _pretrain_args

    if _model is not None:
        return

    from models.graph2seq_series_rel import Graph2SeqSeriesRel
    from utils.data_utils import load_vocab
    from utils.train_utils import param_count, set_seed
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    set_seed(42)

    _device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    assert os.path.exists(_CHECKPOINT_PATH), f"Checkpoint not found: {_CHECKPOINT_PATH}"
    state = torch.load(_CHECKPOINT_PATH, map_location="cpu")
    _pretrain_args = state["args"]
    pretrain_state_dict = state["state_dict"]

    # Ensure required attributes exist on pretrain_args
    defaults = {
        "mpn_type": "dgcn",
        "rel_pos": "emb_only",
        "predict_batch_size": 4096,
    }
    for attr, default_val in defaults.items():
        if not hasattr(_pretrain_args, attr):
            setattr(_pretrain_args, attr, default_val)

    # Load vocab
    _vocab = load_vocab(_VOCAB_PATH)
    _vocab_tokens = [k for k, v in sorted(_vocab.items(), key=lambda tup: tup[1])]

    # Build model
    _model = Graph2SeqSeriesRel(_pretrain_args, _vocab)
    _model.load_state_dict(pretrain_state_dict)
    _model.to(_device)
    _model.eval()

    logging.info(f"Graph2SMILES loaded. Parameters: {param_count(_model)}")


def _smiles_to_graph_batch(smiles_list: List[str]):
    """
    Convert a list of reactant SMILES into a G2SBatch ready for inference.

    This replicates the preprocessing pipeline:
      1. tokenize_smiles (for src_token_ids / src_lens -- needed for batching metadata)
      2. get_graph_features_from_smi (graph featurization)
      3. collate_graph_features + collate_graph_distances
    """
    from utils.data_utils import (
        tokenize_smiles, get_graph_features_from_smi,
        collate_graph_features, collate_graph_distances, G2SBatch
    )
    from rdkit import Chem
    import re

    graph_features_list = []
    tgt_token_ids_list = []
    tgt_lens_list = []
    a_lengths = []

    # We need dummy target tokens for the batch structure (they are not used during prediction)
    # Just use a minimal dummy: [_EOS]
    dummy_tgt = [_vocab["_EOS"]]

    max_tgt_len = 1

    for i, smi in enumerate(smiles_list):
        # Remove atom mapping numbers for input SMILES (the model expects clean SMILES)
        clean_smi = smi.strip()
        if not clean_smi:
            clean_smi = "CC"

        # Graph featurization
        gf = get_graph_features_from_smi((i, clean_smi, False))
        a_scopes, a_scopes_lens, b_scopes, b_scopes_lens, \
            a_features, a_features_lens, b_features, b_features_lens, \
            a_graphs, b_graphs = gf

        # Mask out chiral tag (as done in G2SDataset.__init__)
        a_features[:, 6] = 2

        graph_feature = (a_scopes, b_scopes, a_features, b_features, a_graphs, b_graphs)
        graph_features_list.append(graph_feature)

        # Compute atom length for distance computation
        a_length = a_scopes[-1][0] + a_scopes[-1][1] - a_scopes[0][0]
        a_lengths.append(a_length)

        # Dummy target
        tgt_ids = np.array(dummy_tgt, dtype=np.int64)
        tgt_token_ids_list.append(tgt_ids)
        tgt_lens_list.append(len(dummy_tgt))

    # Collate graph features
    fnode, fmess, agraph, bgraph, atom_scope, bond_scope = collate_graph_features(graph_features_list)

    # Target tensors
    max_tgt_len = max(tgt_lens_list)
    padded_tgt = np.zeros((len(smiles_list), max_tgt_len), dtype=np.int64)
    for i, tgt in enumerate(tgt_token_ids_list):
        padded_tgt[i, :len(tgt)] = tgt

    tgt_token_ids = torch.as_tensor(padded_tgt, dtype=torch.long)
    tgt_lengths = torch.tensor(tgt_lens_list, dtype=torch.long)

    # Compute graph distances (needed for attention encoder with rel_pos)
    # Create a minimal args-like object for collate_graph_distances
    class _Args:
        task = "reaction_prediction"
        compute_graph_distance = True
    _args = _Args()

    distances = collate_graph_distances(_args, graph_features_list, a_lengths)

    batch = G2SBatch(
        fnode=fnode,
        fmess=fmess,
        agraph=agraph,
        bgraph=bgraph,
        atom_scope=atom_scope,
        bond_scope=bond_scope,
        tgt_token_ids=tgt_token_ids,
        tgt_lengths=tgt_lengths,
        distances=distances
    )

    return batch


def _decode_predictions(predictions, scores, n_best: int) -> List[List[Dict]]:
    """
    Decode model predictions (token indices) into SMILES strings with scores.

    Args:
        predictions: list of lists of tensors (batch x n_best x token_indices)
        scores: list of lists of floats (batch x n_best)
        n_best: number of predictions per input

    Returns:
        List of lists of dicts: [{'smiles': str, 'score': float}, ...]
    """
    from utils.data_utils import canonicalize_smiles

    all_results = []
    for sample_preds, sample_scores in zip(predictions, scores):
        sample_results = []
        seen = set()
        for pred, score in zip(sample_preds, sample_scores):
            predicted_idx = pred.detach().cpu().numpy()
            # Decode tokens (exclude EOS token at the end)
            predicted_tokens = [_vocab_tokens[idx] for idx in predicted_idx[:-1]]
            smi = "".join(predicted_tokens)

            # Canonicalize
            canon_smi = canonicalize_smiles(smi, trim=False, suppress_warning=True)
            if not canon_smi:
                canon_smi = smi  # keep raw if canonicalization fails

            # Deduplicate
            if canon_smi in seen:
                continue
            seen.add(canon_smi)

            score_val = score.item() if hasattr(score, 'item') else float(score)
            sample_results.append({
                "smiles": canon_smi,
                "score": score_val,
            })

        all_results.append(sample_results)

    return all_results


def run(input_data: Union[str, List[str]], top_k: int = 5, device: str = "cuda:0",
        beam_size: int = None) -> List[Dict]:
    """
    Run forward reaction prediction using Graph2SMILES.

    Args:
        input_data: Reactant SMILES string or list of SMILES strings.
                    For multi-reactant reactions, use '.' separator (e.g., "CCO.CC(=O)Cl").
        top_k: Number of top predictions to return per input.
        device: Device to use (e.g., "cuda:0", "cpu").
        beam_size: Beam size for beam search. Defaults to max(top_k, 10).

    Returns:
        List of dicts with format:
        [{'input': '...', 'predictions': [{'smiles': '...', 'score': 0.95}, ...]}, ...]
    """
    # Handle single SMILES input
    if isinstance(input_data, str):
        smiles_list = [input_data]
    else:
        smiles_list = list(input_data)

    if not smiles_list:
        return []

    # Load model if not already loaded
    _load_model(device=device)

    if beam_size is None:
        beam_size = max(top_k, 10)

    n_best = beam_size  # request beam_size predictions, then trim to top_k

    results = []

    # Process in small batches to avoid OOM
    batch_size_limit = 32

    for batch_start in range(0, len(smiles_list), batch_size_limit):
        batch_smiles = smiles_list[batch_start:batch_start + batch_size_limit]

        # Build graph batch
        batch = _smiles_to_graph_batch(batch_smiles)
        batch.to(_device)

        with torch.no_grad():
            model_results = _model.predict_step(
                reaction_batch=batch,
                batch_size=batch.size,
                beam_size=beam_size,
                n_best=n_best,
                temperature=1.0,
                min_length=1,
                max_length=512
            )

        # Decode predictions
        decoded = _decode_predictions(
            model_results["predictions"],
            model_results["scores"],
            n_best=n_best
        )

        for smi, preds in zip(batch_smiles, decoded):
            # Trim to top_k
            top_preds = preds[:top_k]
            results.append({
                "input": smi,
                "predictions": top_preds,
            })

    return results


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    test_smiles = "CCO.CC(=O)Cl"
    print(f"Input: {test_smiles}")
    print("Running Graph2SMILES forward prediction...")

    output = run(test_smiles, top_k=5)
    for res in output:
        print(f"\nInput: {res['input']}")
        for i, pred in enumerate(res['predictions']):
            print(f"  Top-{i+1}: {pred['smiles']} (score: {pred['score']:.4f})")
