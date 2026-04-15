"""
Root-aligned SMILES (R-SMILES) forward reaction prediction inference.

Reference: Zhong et al., "Root-aligned SMILES: A Tight Representation
for Chemical Reaction Prediction", Chem. Sci., 2022.

GitHub: https://github.com/otori-bird/retrosynthesis

Uses OpenNMT-py 2.2 as the translation backend in RtoP (Reactant-to-Product)
mode with root-aligned SMILES augmentation.

Implements the uniform inference interface for the benchmark runner.
"""

import os
import sys
import re
import tempfile
from typing import List, Dict, Union

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

_DEFAULT_CONFIG = {
    'model_path': os.path.join(_ROOT_DIR, 'checkpoint', 'model.pt'),
    'beam_size': 10,
    'n_best': 10,
    'batch_size': 4096,
    'batch_type': 'tokens',
    'max_length': 1000,
    'gpu': 0,
    'augmentation_factor': 1,
}

_translator = None


def _smi_tokenize(smi: str) -> str:
    """Tokenize a SMILES string."""
    pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    tokens = re.findall(pattern, smi)
    return ' '.join(tokens)


def _detokenize(tokens: str) -> str:
    """Remove spaces from tokenized SMILES."""
    return tokens.replace(' ', '')


def _load_translator():
    """Lazy-load the OpenNMT-py 2.x translator."""
    global _translator
    if _translator is None:
        import io
        import torch

        if _ROOT_DIR not in sys.path:
            sys.path.insert(0, _ROOT_DIR)

        try:
            import onmt
            from onmt.translate.translator import build_translator
            import argparse

            parser = argparse.ArgumentParser()
            onmt.opts.translate_opts(parser)

            model_path = _DEFAULT_CONFIG['model_path']
            gpu = _DEFAULT_CONFIG['gpu'] if torch.cuda.is_available() else -1

            opt = parser.parse_args([
                '-model', model_path,
                '-src', '/dev/null',
                '-beam_size', str(_DEFAULT_CONFIG['beam_size']),
                '-n_best', str(_DEFAULT_CONFIG['n_best']),
                '-batch_size', str(_DEFAULT_CONFIG['batch_size']),
                '-batch_type', _DEFAULT_CONFIG['batch_type'],
                '-max_length', str(_DEFAULT_CONFIG['max_length']),
                '-replace_unk',
                '-gpu', str(gpu),
            ])

            _translator = build_translator(opt, report_score=False,
                                           out_file=io.StringIO())
        except ImportError:
            raise ImportError(
                "OpenNMT-py >= 2.2.0 is required. Install with: pip install OpenNMT-py==2.2.0"
            )
    return _translator


def load_model(model_path=None, augmentation_factor=None, **kwargs):
    """Load model checkpoint. Called by benchmark runner."""
    if model_path:
        _DEFAULT_CONFIG['model_path'] = model_path
    if augmentation_factor is not None:
        _DEFAULT_CONFIG['augmentation_factor'] = augmentation_factor


def _augment_single_mol(smi: str, n_augments: int) -> List[str]:
    """Generate root-aligned SMILES augmentations for a single molecule."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return [smi]

    augmented = set()
    augmented.add(smi)
    num_atoms = mol.GetNumAtoms()

    for root_idx in range(num_atoms):
        if len(augmented) >= n_augments:
            break
        try:
            new_smi = Chem.MolToSmiles(mol, rootedAtAtom=root_idx, canonical=False)
            augmented.add(new_smi)
        except Exception:
            continue

    return list(augmented)[:n_augments]


def _augment_smiles(smi: str, n_augments: int = 1) -> List[str]:
    """Generate root-aligned SMILES augmentations.

    For multi-component reactants (e.g. 'A.B'), augments each component
    independently and combines them.
    """
    if n_augments <= 1:
        return [smi]

    # Split multi-component SMILES
    components = smi.split('.')
    if len(components) == 1:
        return _augment_single_mol(smi, n_augments)

    # Augment each component independently
    from rdkit import Chem
    comp_augments = []
    for comp in components:
        comp_augments.append(_augment_single_mol(comp, n_augments))

    # Combine: for each augmentation index, join the components
    augmented = set()
    augmented.add(smi)

    # Zip augmentations across components
    max_len = max(len(a) for a in comp_augments)
    for i in range(max_len):
        if len(augmented) >= n_augments:
            break
        parts = []
        for ca in comp_augments:
            parts.append(ca[i % len(ca)])
        augmented.add('.'.join(parts))

    # If still need more, use random combinations
    import itertools
    if len(augmented) < n_augments:
        for combo in itertools.product(*comp_augments):
            if len(augmented) >= n_augments:
                break
            augmented.add('.'.join(combo))

    return list(augmented)[:n_augments]


def run(smiles: Union[str, List[str]], top_k: int = 5) -> List[Dict]:
    """
    Predict products from reactants using Root-aligned SMILES.

    Args:
        smiles: Reactant SMILES string or list of SMILES strings
        top_k: Number of top predictions to return

    Returns:
        [{'input': '...', 'predictions': [{'smiles': '...', 'score': 0.95}, ...]}]
    """
    translator = _load_translator()
    aug_factor = _DEFAULT_CONFIG['augmentation_factor']

    if isinstance(smiles, str):
        smiles_list = [smiles]
    else:
        smiles_list = list(smiles)

    results = []
    for smi in smiles_list:
        try:
            # Generate augmented SMILES
            augmented = _augment_smiles(smi, aug_factor)
            tokenized_lines = [_smi_tokenize(a) for a in augmented]

            # Write to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                for line in tokenized_lines:
                    f.write(line + '\n')
                src_path = f.name

            # Translate
            scores_all, preds_all = translator.translate(
                src=src_path,
                batch_size=len(tokenized_lines),
            )

            # Aggregate predictions across augmentations
            from rdkit import Chem
            seen = {}
            for aug_preds, aug_scores in zip(preds_all, scores_all):
                for pred_tokens, score in zip(aug_preds, aug_scores):
                    pred_smi = _detokenize(pred_tokens)
                    # Canonicalize for deduplication
                    mol = Chem.MolFromSmiles(pred_smi)
                    if mol is None:
                        continue
                    canon = Chem.MolToSmiles(mol)
                    import math
                    prob = math.exp(float(score))
                    if canon not in seen or prob > seen[canon]:
                        seen[canon] = prob

            # Sort by score
            sorted_preds = sorted(seen.items(), key=lambda x: x[1], reverse=True)
            formatted_preds = [
                {'smiles': s, 'score': sc}
                for s, sc in sorted_preds[:top_k]
            ]

            os.unlink(src_path)
        except Exception:
            formatted_preds = []

        results.append({
            'input': smi,
            'predictions': formatted_preds
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
