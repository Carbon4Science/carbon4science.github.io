"""
MEGAN forward reaction prediction inference.

Reference: Sacha et al., "Molecule Edit Graph Attention Network:
Modeling Chemical Reactions as Sequences of Graph Edits", JCIM, 2021.

GitHub: https://github.com/molecule-one/megan

Implements the uniform inference interface for the benchmark runner.
"""

import os
import sys
from typing import List, Dict, Union

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.join(_ROOT_DIR, 'repo')

_DEFAULT_CONFIG = {
    'save_path': os.path.join(_REPO_DIR, 'models', 'uspto_mit_mix'),
    'beam_size': 10,
    'max_gen_steps': 16,
    'n_max_atoms': 200,
    'device': 'cuda:0',
}

_model = None
_action_vocab = None
_base_action_masks = None
_featurizer = None


def _load_model():
    """Lazy-load the MEGAN model with proper gin config and action vocabulary."""
    global _model, _action_vocab, _base_action_masks, _featurizer

    if _model is not None:
        return _model

    import torch

    if _REPO_DIR not in sys.path:
        sys.path.insert(0, _REPO_DIR)

    os.environ.setdefault('PROJECT_ROOT', _REPO_DIR)
    os.environ.setdefault('DATA_DIR', os.path.join(_REPO_DIR, 'data'))
    os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

    # Import train_megan to register gin configurables
    from bin.train import train_megan
    import gin

    save_path = _DEFAULT_CONFIG['save_path']
    config_path = os.path.join(save_path, 'config.gin')
    gin.parse_config_file(config_path)

    from src.config import get_featurizer
    featurizer_key = gin.query_parameter('train_megan.featurizer_key')
    _featurizer = get_featurizer(featurizer_key)

    _action_vocab = _featurizer.get_actions_vocabulary(save_path)

    from src.model.megan import Megan
    from src.model.megan_utils import get_base_action_masks

    n_max_atoms = _DEFAULT_CONFIG['n_max_atoms']
    _base_action_masks = get_base_action_masks(n_max_atoms + 1, action_vocab=_action_vocab)

    device = _DEFAULT_CONFIG['device']
    if not torch.cuda.is_available():
        device = 'cpu'
    _DEFAULT_CONFIG['device'] = device

    checkpoint = torch.load(
        os.path.join(save_path, 'model_best.pt'),
        map_location=device,
        weights_only=False,
    )
    _model = Megan(
        n_atom_actions=_action_vocab['n_atom_actions'],
        n_bond_actions=_action_vocab['n_bond_actions'],
        prop2oh=_action_vocab['prop2oh'],
    ).to(device)
    _model.load_state_dict(checkpoint['model'])
    _model.eval()

    return _model


def load_model(model_path=None, **kwargs):
    """Load model checkpoint. Called by benchmark runner."""
    if model_path:
        _DEFAULT_CONFIG['save_path'] = model_path


def run(smiles: Union[str, List[str]], top_k: int = 5) -> List[Dict]:
    """
    Predict products from reactants using MEGAN.

    Args:
        smiles: Reactant SMILES string or list of SMILES strings
        top_k: Number of top predictions to return

    Returns:
        [{'input': '...', 'predictions': [{'smiles': '...', 'score': 0.95}, ...]}]
    """
    model = _load_model()

    if isinstance(smiles, str):
        smiles_list = [smiles]
    else:
        smiles_list = list(smiles)

    import torch
    from rdkit import Chem
    from src.model.beam_search import beam_search
    from src.model.megan_utils import RdkitCache
    from src.utils import mol_to_unmapped_smiles
    from src.feat.utils import fix_explicit_hs, add_map_numbers

    device = _DEFAULT_CONFIG['device']
    beam_size = max(_DEFAULT_CONFIG['beam_size'], top_k)
    rdkit_cache = RdkitCache(props=_action_vocab['props'])

    results = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                results.append({'input': smi, 'predictions': []})
                continue

            mol = add_map_numbers(mol)
            mol = fix_explicit_hs(mol)

            with torch.no_grad():
                beam_results = beam_search(
                    [model], [mol], rdkit_cache=rdkit_cache,
                    max_steps=_DEFAULT_CONFIG['max_gen_steps'],
                    beam_size=beam_size, batch_size=1,
                    base_action_masks=_base_action_masks,
                    max_atoms=_DEFAULT_CONFIG['n_max_atoms'],
                    action_vocab=_action_vocab,
                )

            formatted_preds = []
            seen = set()
            for path in beam_results[0][:top_k]:
                raw_smi = path.get('final_smi_unmapped', '')
                if not raw_smi:
                    continue

                # For forward prediction, select the largest product fragment
                parts = sorted(raw_smi.split('.'), key=len, reverse=True)
                pred_smi = parts[0]

                # Canonicalize
                pred_mol = Chem.MolFromSmiles(pred_smi)
                if pred_mol is None:
                    continue
                canon = Chem.MolToSmiles(pred_mol, canonical=True)

                if canon in seen:
                    continue
                seen.add(canon)

                formatted_preds.append({
                    'smiles': canon,
                    'score': float(path.get('prob', 0.0)),
                })

        except Exception:
            formatted_preds = []

        results.append({
            'input': smi,
            'predictions': formatted_preds,
        })

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python Inference.py <SMILES> [top_k]")
        sys.exit(1)
    smiles = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    results = run(smiles, top_k=top_k)
    for r in results:
        print(f"Input: {r['input']}")
        for i, p in enumerate(r['predictions'], 1):
            print(f"  {i}. {p['smiles']} (score: {p['score']:.4f})")
