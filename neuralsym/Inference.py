"""
NeuralSym FORWARD reaction prediction inference.

Given reactants, predicts product(s) by classifying the most likely
forward reaction template and applying it to the reactants.

Key difference from retrosynthesis:
  - Input: reactant SMILES
  - Templates stored as r_temp >> p_temp (forward direction)
  - RDChiral applies template to reactants to generate products

Usage:
    from Forward.neuralsym.Inference import run
    results = run("CCO.CC(=O)Cl")  # Single SMILES
    results = run(["CCO.CC(=O)Cl", "c1ccccc1.Cl"])  # Multiple SMILES
"""
import os
import sys
import itertools
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Union
from scipy import sparse
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

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

        # Load forward templates (r_temp >> p_temp)
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

    def predict(self, smiles: str, topk: int = 5) -> List[Dict]:
        """Predict products for a single reactant SMILES."""
        with torch.no_grad():
            # Generate fingerprint from REACTANTS
            rct_fp = mol_smi_to_count_fp(smiles, infer_config['radius'], infer_config['orig_fp_size'])
            logged = sparse.csr_matrix(np.log(rct_fp.toarray() + 1))
            final_fp = logged[:, self.indices]
            final_fp = torch.as_tensor(final_fp.toarray()).float().to(self.device)

            # Model inference
            outputs = self.model(final_fp)
            probs = nn.Softmax(dim=1)(outputs)
            top_indices = torch.topk(probs, k=topk, dim=1)[1].squeeze(dim=0).cpu().numpy()

            # Apply forward templates to reactants to generate products
            rct_mols = [Chem.MolFromSmiles(s) for s in smiles.split('.')]
            rct_mols = [m for m in rct_mols if m is not None]

            results = []
            for idx in top_indices:
                score = probs[0, idx.item()].item()
                template = self.templates[idx.item()]
                products = []
                try:
                    rxn = AllChem.ReactionFromSmarts(template)
                    n_tpl = rxn.GetNumReactantTemplates()
                    # Try permutations of reactant molecules to match template slots
                    for combo in itertools.islice(
                        itertools.permutations(range(len(rct_mols)), n_tpl), 120
                    ):
                        mols = [rct_mols[i] for i in combo]
                        try:
                            product_sets = rxn.RunReactants(mols)
                            if product_sets:
                                for ps in product_sets:
                                    for p in ps:
                                        try:
                                            smi = Chem.MolToSmiles(p)
                                            if smi and smi not in products:
                                                products.append(smi)
                                        except:
                                            pass
                                break  # stop after first successful permutation
                        except:
                            pass
                except:
                    products = []

                results.append({
                    'products': products,
                    'score': score,
                    'template': template
                })

        return results


def run(smiles: Union[str, List[str]], top_k: int = 5) -> List[Dict]:
    """
    Run forward reaction prediction on input SMILES.

    Args:
        smiles: Reactant SMILES string or a list of reactant SMILES strings
        top_k: Number of top predictions to return (default: 5)

    Returns:
        List of result dicts, one per input SMILES. Each dict contains:
            - 'input': Input reactant SMILES string
            - 'predictions': List of prediction dicts with 'smiles' and 'score'

    Example:
        >>> results = run("CCO.CC(=O)Cl")
        >>> results[0]['predictions'][0]
        {'smiles': 'CCOC(C)=O', 'score': 0.85}
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
            # Each product prediction may have multiple product SMILES
            product_smiles = p['products'][0] if p['products'] else ''
            if product_smiles:
                formatted_preds.append({
                    'smiles': product_smiles,
                    'score': p['score']
                })
        results.append({
            'input': smi,
            'predictions': formatted_preds
        })

    return results


if __name__ == '__main__':
    test_smiles = "CCO.CC(=O)Cl"
    print(f"Running forward prediction on: {test_smiles}\n")

    results = run(test_smiles, top_k=5)
    for r in results:
        print(f"Reactants: {r['input']}")
        print("-" * 60)
        for i, pred in enumerate(r['predictions'], 1):
            print(f"  {i}. Score: {pred['score']:.4f}")
            print(f"     Product: {pred['smiles']}")
        print()
