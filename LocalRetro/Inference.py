"""
Simple inference interface for LocalRetro.

Usage:
    from Inference import run
    results = run("CCO")  # Returns DataFrame with predicted reactants
"""

import os
import sys

# Set default paths relative to this file's location
_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add scripts directory to path for imports
_SCRIPTS_DIR = os.path.join(_ROOT_DIR, 'scripts')
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

_DEFAULT_ARGS = {
    'dataset': 'USPTO_50K',
    'data_dir': os.path.join(_ROOT_DIR, 'data', 'USPTO_50K'),
    'config_path': os.path.join(_ROOT_DIR, 'data', 'configs', 'default_config.json'),
    'model_path': os.path.join(_ROOT_DIR, 'models', 'LocalRetro_USPTO_50K.pth'),
    'device': 'cuda:0',
}

_model = None


def _get_model():
    """Lazy-load the model on first use."""
    global _model
    if _model is None:
        import torch

        # Add LocalRetro directory to path for internal imports (scripts, LocalTemplate)
        _localretro_dir = os.path.dirname(os.path.abspath(__file__))
        if _localretro_dir not in sys.path:
            sys.path.insert(0, _localretro_dir)

        # Now import LocalRetro class from Retrosynthesis.py
        from Retro.LocalRetro.Retrosynthesis import LocalRetro

        args = _DEFAULT_ARGS.copy()
        if not torch.cuda.is_available():
            args['device'] = 'cpu'

        _model = LocalRetro(args)
    return _model


def run(smiles, top_k=10, verbose=False):
    """
    Predict retrosynthesis for a given product SMILES.

    Args:
        smiles: Product SMILES string or list of SMILES strings
        top_k: Number of top predictions to return (default: 10)
        verbose: Print detailed prediction info (default: False)

    Returns:
        List of result dicts, one per input SMILES. Each dict contains:
            - 'input': Input SMILES string
            - 'predictions': List of prediction dicts with 'smiles' and 'score'

    Example:
        >>> results = run("CCO")
        >>> results[0]['predictions'][0]
        {'smiles': 'C=C.O', 'score': 0.85}
    """
    model = _get_model()

    if isinstance(smiles, str):
        smiles_list = [smiles]
    else:
        smiles_list = list(smiles)

    results = []
    for smi in smiles_list:
        # Request extra predictions to compensate for filtered self-predictions
        df = model.retrosnythesis(smi, top_k=top_k + 5, verbose=verbose)
        formatted_preds = []
        if df is not None and len(df) > 0:
            # Canonicalize input for comparison
            from rdkit import Chem
            input_mol = Chem.MolFromSmiles(smi)
            input_canonical = Chem.MolToSmiles(input_mol) if input_mol else smi

            for _, row in df.iterrows():
                score = float(row['Score'])
                pred_smiles = row['SMILES']
                # Skip NaN scores (no-reaction template) and self-predictions
                if score != score:  # NaN check
                    continue
                pred_mol = Chem.MolFromSmiles(pred_smiles)
                if pred_mol and Chem.MolToSmiles(pred_mol) == input_canonical:
                    continue
                formatted_preds.append({
                    'smiles': pred_smiles,
                    'score': score
                })
                if len(formatted_preds) >= top_k:
                    break
        results.append({
            'input': smi,
            'predictions': formatted_preds
        })

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python Inference.py <SMILES> [top_k]")
        print("Example: python Inference.py 'CCO' 10")
        sys.exit(1)

    smiles = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    results = run(smiles, top_k=top_k, verbose=True)
    print("\nResults:")
    for r in results:
        print(f"Input: {r['input']}")
        for i, p in enumerate(r['predictions'], 1):
            print(f"  {i}. {p['smiles']} (score: {p['score']:.4f})")
