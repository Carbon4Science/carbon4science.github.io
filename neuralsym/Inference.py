"""
Simple inference API for NeuralSym retrosynthesis prediction.

Usage:
    from inference import run
    results = run("CCO")  # Single SMILES
    results = run(["CCO", "CCCO"])  # Multiple SMILES
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Union
from scipy import sparse
from rdkit import RDLogger
from rdchiral.main import rdchiralReaction, rdchiralReactants, rdchiralRun

# Add neuralsym directory to path so local modules are importable from any working directory
_NEURALSYM_DIR = os.path.dirname(os.path.abspath(__file__))
if _NEURALSYM_DIR not in sys.path:
    sys.path.insert(0, _NEURALSYM_DIR)

from model import TemplateNN_Highway
from prepare_data import mol_smi_to_count_fp
from infer_config import infer_config

RDLogger.DisableLog("rdApp.warning")

DATA_FOLDER = Path(__file__).resolve().parent / 'data'
CHECKPOINT_FOLDER = Path(__file__).resolve().parent / 'checkpoint'

_proposer = None


def _get_proposer():
    """Lazy initialization of the proposer model."""
    global _proposer
    if _proposer is None:
        _proposer = _Proposer()
    return _proposer


class _Proposer:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load templates
        with open(DATA_FOLDER / infer_config['templates_file'], 'r') as f:
            templates = f.readlines()
        self.templates = []
        for p in templates:
            pa, cnt = p.strip().split(': ')
            if int(cnt) >= infer_config['min_freq']:
                self.templates.append(pa)

        # Load model
        checkpoint = torch.load(
            CHECKPOINT_FOLDER / f"{infer_config['expt_name']}.pth.tar",
            map_location=self.device,
            weights_only=False,
        )
        self.model = TemplateNN_Highway(
            output_size=len(self.templates),
            size=infer_config['hidden_size'],
            num_layers_body=infer_config['depth'],
            input_size=infer_config['final_fp_size']
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Load variance indices for fingerprint filtering
        self.indices = np.loadtxt(DATA_FOLDER / 'variance_indices.txt').astype('int')

    def predict(self, smiles: str, topk: int = 10) -> List[Dict]:
        """Predict precursors for a single SMILES."""
        with torch.no_grad():
            # Generate fingerprint
            prod_fp = mol_smi_to_count_fp(smiles, infer_config['radius'], infer_config['orig_fp_size'])
            logged = sparse.csr_matrix(np.log(prod_fp.toarray() + 1))
            final_fp = logged[:, self.indices]
            final_fp = torch.as_tensor(final_fp.toarray()).float().to(self.device)

            # Model inference
            outputs = self.model(final_fp)
            probs = nn.Softmax(dim=1)(outputs)
            top_indices = torch.topk(probs, k=topk, dim=1)[1].squeeze(dim=0).cpu().numpy()

            # Apply templates to generate precursors
            results = []
            for idx in top_indices:
                score = probs[0, idx.item()].item()
                template = self.templates[idx.item()]
                try:
                    rxn = rdchiralReaction(template)
                    prod = rdchiralReactants(smiles)
                    precursors = rdchiralRun(rxn, prod)
                except:
                    precursors = []

                results.append({
                    'precursors': precursors,
                    'score': score,
                    'template': template
                })

        return results


def run(smiles: Union[str, List[str]], top_k: int = 10) -> List[Dict]:
    """
    Run retrosynthesis prediction on input SMILES.

    Args:
        smiles: A single SMILES string or a list of SMILES strings
        top_k: Number of top predictions to return (default: 10)

    Returns:
        List of result dicts, one per input SMILES. Each dict contains:
            - 'input': Input SMILES string
            - 'predictions': List of prediction dicts with 'smiles' and 'score'

    Example:
        >>> results = run("CCO")
        >>> results[0]['predictions'][0]
        {'smiles': 'C=C.O', 'score': 0.85}
    """
    proposer = _get_proposer()

    if isinstance(smiles, str):
        smiles_list = [smiles]
    else:
        smiles_list = list(smiles)

    results = []
    for smi in smiles_list:
        preds = proposer.predict(smi, topk=top_k)
        formatted_preds = []
        for p in preds:
            # Join multiple precursors with '.'
            precursor_smiles = '.'.join(p['precursors']) if p['precursors'] else ''
            if precursor_smiles:
                formatted_preds.append({
                    'smiles': precursor_smiles,
                    'score': p['score']
                })
        results.append({
            'input': smi,
            'predictions': formatted_preds
        })

    return results


if __name__ == '__main__':
    # Example usage
    test_smiles = "COC(=O)c1cccc2[nH]c(NCC3CCNCC3)nc12"
    print(f"Running inference on: {test_smiles}\n")

    results = run(test_smiles, topk=5)

    for smi, predictions in results.items():
        print(f"Product: {smi}")
        print("-" * 60)
        for i, pred in enumerate(predictions, 1):
            print(f"  {i}. Score: {pred['score']:.4f}")
            print(f"     Precursors: {pred['precursors']}")
        print()
