"""
Molecular Transformer forward reaction prediction inference.

Reference: Schwaller et al., "Molecular Transformer: A Model for
Uncertainty-Calibrated Chemical Reaction Prediction", ACS Cent. Sci., 2019.

GitHub: https://github.com/pschwllr/MolecularTransformer

Uses OpenNMT-py as the translation backend. Expects a trained checkpoint
and vocabulary files in the checkpoint directory.

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
    'beam_size': 5,
    'batch_size': 64,
    'max_length': 200,
    'gpu': 0,
}

_translator = None


def _smi_tokenize(smi: str) -> str:
    """Tokenize a SMILES string by inserting spaces between characters."""
    pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    tokens = re.findall(pattern, smi)
    return ' '.join(tokens)


def _detokenize(tokens: str) -> str:
    """Remove spaces from tokenized SMILES."""
    return tokens.replace(' ', '')


def _load_translator():
    """Lazy-load the OpenNMT translator."""
    global _translator
    if _translator is None:
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
                '-n_best', str(_DEFAULT_CONFIG['beam_size']),
                '-batch_size', str(_DEFAULT_CONFIG['batch_size']),
                '-beam_size', str(_DEFAULT_CONFIG['beam_size']),
                '-max_length', str(_DEFAULT_CONFIG['max_length']),
                '-replace_unk',
                '-gpu', str(gpu),
            ])

            import io
            _translator = build_translator(opt, report_score=False,
                                           out_file=io.StringIO())
        except ImportError:
            raise ImportError(
                "OpenNMT-py is required. Install with: pip install OpenNMT-py==1.2.0"
            )
    return _translator


def load_model(model_path=None, **kwargs):
    """Load model checkpoint. Called by benchmark runner."""
    if model_path:
        _DEFAULT_CONFIG['model_path'] = model_path


def run(smiles: Union[str, List[str]], top_k: int = 5) -> List[Dict]:
    """
    Predict products from reactants using Molecular Transformer.

    Args:
        smiles: Reactant SMILES string or list of SMILES strings
        top_k: Number of top predictions to return

    Returns:
        [{'input': '...', 'predictions': [{'smiles': '...', 'score': 0.95}, ...]}]
    """
    translator = _load_translator()

    if isinstance(smiles, str):
        smiles_list = [smiles]
    else:
        smiles_list = list(smiles)

    results = []
    for smi in smiles_list:
        try:
            tokenized = _smi_tokenize(smi)

            # Write tokenized input to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(tokenized + '\n')
                src_path = f.name

            # Translate
            scores, predictions = translator.translate(
                src=src_path,
                batch_size=1,
            )

            formatted_preds = []
            if predictions and len(predictions) > 0:
                for pred_tokens, score_tensor in zip(predictions[0][:top_k], scores[0][:top_k]):
                    pred_smi = _detokenize(pred_tokens)
                    score = float(score_tensor) if hasattr(score_tensor, '__float__') else 0.0
                    import math
                    score = math.exp(score)  # Convert log-prob to probability
                    if pred_smi:
                        formatted_preds.append({
                            'smiles': pred_smi,
                            'score': score
                        })

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
