#!/usr/bin/env python3
"""
RSGPT Inference Module

Usage:
    from inference import RSGPTPredictor

    predictor = RSGPTPredictor()  # or RSGPTPredictor(device='cuda:0')
    results = predictor.run('CC(=O)Oc1ccccc1C(=O)O')  # Aspirin
    print(results)
"""

import os
import re
import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Any
from transformers import LlamaForCausalLM, LlamaConfig
from rdkit import Chem


class RSGPTPredictor:
    """RSGPT model for retrosynthesis prediction."""

    def __init__(
        self,
        weights_path: str = "weights/finetune_50k.pth",
        config_path: str = "configs/rxngpt_llama1B.json",
        tokenizer_path: str = "vocab.json",
        device: str = "cuda:0",
    ):
        """
        Initialize the RSGPT predictor.

        Args:
            weights_path: Path to model weights (.pth file)
            config_path: Path to model config (JSON file)
            tokenizer_path: Path to tokenizer (vocab.json)
            device: Device to run inference on ('cuda:0', 'cuda:1', 'cpu', etc.)
        """
        self.device = device

        # Resolve paths relative to this file's directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(base_dir, weights_path) if not os.path.isabs(weights_path) else weights_path
        config_path = os.path.join(base_dir, config_path) if not os.path.isabs(config_path) else config_path
        tokenizer_path = os.path.join(base_dir, tokenizer_path) if not os.path.isabs(tokenizer_path) else tokenizer_path

        print(f"Loading tokenizer from {tokenizer_path}...")
        self._load_tokenizer(tokenizer_path)

        print(f"Loading model from {config_path}...")
        self._load_model(config_path, weights_path)

        print(f"Model ready on {device}")

    def _load_tokenizer(self, tokenizer_path: str):
        """Load the SMILES BPE tokenizer."""
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.eos_token_id = self.tokenizer.encode('</s>').ids[0]
        self.bos_token_id = self.tokenizer.encode('<s>').ids[0]

    def _load_model(self, config_path: str, weights_path: str):
        """Load the LlamaForCausalLM model with pretrained weights."""
        config = LlamaConfig.from_pretrained(config_path)
        self.model = LlamaForCausalLM(config)

        # Load weights
        state_dict = torch.load(weights_path, map_location='cpu')

        # Map checkpoint keys to LlamaForCausalLM structure
        # RxnGPT checkpoint has 'module.' prefix (DDP) and nested structure
        new_state_dict = {}
        for k, v in state_dict.items():
            # Remove 'module.' DDP prefix
            if k.startswith('module.'):
                k = k[7:]

            # Skip nested duplicate keys
            if k.startswith('model.model.') or k.startswith('model.lm_head'):
                continue
            if k.startswith('model.'):
                continue

            # Map: embed_tokens -> model.embed_tokens, layers -> model.layers, etc.
            new_state_dict['model.' + k] = v

        # Load lm_head separately if exists
        for k, v in state_dict.items():
            if 'lm_head' in k:
                clean_k = k.replace('module.', '').replace('model.', '')
                new_state_dict[clean_k] = v
                break

        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.model.half()  # Use FP16 for faster inference

    @staticmethod
    def canonicalize_smiles(smiles: str) -> Optional[str]:
        """Canonicalize a SMILES string."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol)
        except:
            return None

    def _beam_search(
        self,
        input_ids: List[int],
        beam_size: int = 10,
        max_length: int = 100,
    ) -> List[str]:
        """Perform beam search decoding."""
        input_tensor = torch.tensor(input_ids).unsqueeze(0).to(self.device)

        sequences = [(input_tensor, 0.0)]  # (sequence, cumulative score)
        completed_sequences = []

        with torch.no_grad():
            for _ in range(max_length):
                all_candidates = []

                for seq, score in sequences:
                    if seq[0, -1].item() == self.eos_token_id:
                        completed_sequences.append((seq, score))
                        continue

                    outputs = self.model(input_ids=seq)
                    logits = outputs.logits[:, -1, :]
                    logits = F.log_softmax(logits, dim=-1)

                    topk_probs, topk_indices = torch.topk(logits, beam_size, dim=-1)

                    for i in range(beam_size):
                        candidate_seq = torch.cat([seq, topk_indices[:, i].unsqueeze(0)], dim=1)
                        candidate = (candidate_seq, score - topk_probs[0, i].item())
                        all_candidates.append(candidate)

                if not all_candidates:
                    break

                ordered = sorted(all_candidates, key=lambda x: x[1])
                sequences = ordered[:beam_size]

        completed_sequences.extend(sequences)
        completed_sequences = sorted(completed_sequences, key=lambda x: x[1])

        # Decode sequences
        decoded = []
        for seq, score in completed_sequences[:beam_size]:
            tokens = seq[0].tolist()
            text = self.tokenizer.decode(tokens)
            decoded.append(text)

        return decoded

    @staticmethod
    def _extract_reactants(prediction: str) -> Optional[str]:
        """Extract reactants from model prediction."""
        if '<F1>' not in prediction:
            return None

        try:
            # Get everything after <F1>
            after_f1 = prediction.split('<F1>')[-1]
            # Remove </s> if present
            after_f1 = after_f1.replace('</s>', '').strip()

            # Split by <F2>, <F3>, etc.
            reactants = re.split(r'<F\d+>', after_f1)
            reactants = [r.strip() for r in reactants if r.strip()]

            return '.'.join(reactants) if reactants else None
        except:
            return None

    def run(
        self,
        smiles: str,
        beam_size: int = 10,
        max_length: int = 100,
        return_raw: bool = False,
    ) -> Dict[str, Any]:
        """
        Run retrosynthesis prediction on a product SMILES.

        Args:
            smiles: Product SMILES string
            beam_size: Number of beams for beam search
            max_length: Maximum generation length
            return_raw: If True, include raw model outputs

        Returns:
            Dictionary with:
                - 'product': Input product SMILES (canonicalized)
                - 'predictions': List of predicted reactant SMILES
                - 'valid_predictions': List of valid (parseable) reactant SMILES
                - 'raw_outputs': Raw model outputs (if return_raw=True)
        """
        # Canonicalize input
        canonical = self.canonicalize_smiles(smiles)
        if canonical is None:
            return {
                'product': smiles,
                'error': 'Invalid SMILES',
                'predictions': [],
                'valid_predictions': [],
            }

        # Format input: <s><Isyn><O>PRODUCT<F1>
        input_text = f'<s><Isyn><O>{canonical}<F1>'
        input_ids = self.tokenizer.encode(input_text).ids

        # Run beam search
        raw_outputs = self._beam_search(input_ids, beam_size, max_length)

        # Extract reactants
        predictions = []
        valid_predictions = []

        for output in raw_outputs:
            reactants = self._extract_reactants(output)
            if reactants:
                predictions.append(reactants)
                # Validate
                canonical_reactants = self.canonicalize_smiles(reactants)
                if canonical_reactants:
                    valid_predictions.append(canonical_reactants)

        # Deduplicate while preserving order
        seen = set()
        unique_predictions = []
        for p in predictions:
            if p not in seen:
                seen.add(p)
                unique_predictions.append(p)

        seen = set()
        unique_valid = []
        for p in valid_predictions:
            if p not in seen:
                seen.add(p)
                unique_valid.append(p)

        result = {
            'product': canonical,
            'predictions': unique_predictions,
            'valid_predictions': unique_valid,
        }

        if return_raw:
            result['raw_outputs'] = raw_outputs

        return result

    def batch_run(
        self,
        smiles_list: List[str],
        beam_size: int = 10,
        max_length: int = 100,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Run retrosynthesis prediction on multiple SMILES.

        Args:
            smiles_list: List of product SMILES strings
            beam_size: Number of beams for beam search
            max_length: Maximum generation length
            verbose: Print progress

        Returns:
            List of result dictionaries
        """
        results = []

        for i, smiles in enumerate(smiles_list):
            if verbose:
                print(f"Processing {i+1}/{len(smiles_list)}: {smiles[:50]}...")

            result = self.run(smiles, beam_size, max_length)
            results.append(result)

        return results


# Convenience function
_predictor = None

def run(smiles, top_k: int = 10, device: str = "cuda:0") -> List[Dict[str, Any]]:
    """
    Run retrosynthesis prediction.

    Args:
        smiles: Product SMILES string or list of SMILES strings
        top_k: Number of predictions to return
        device: Device to run on

    Returns:
        List of result dicts, one per input SMILES. Each dict contains:
            - 'input': Input SMILES string
            - 'predictions': List of prediction dicts with 'smiles' and 'score'

    Example:
        >>> results = run("CCO")
        >>> results[0]['predictions'][0]
        {'smiles': 'C=C.O', 'score': 1.0}
    """
    global _predictor

    if _predictor is None:
        _predictor = RSGPTPredictor(device=device)

    if isinstance(smiles, str):
        smiles_list = [smiles]
    else:
        smiles_list = list(smiles)

    results = []
    for smi in smiles_list:
        raw_result = _predictor.run(smi, beam_size=top_k)
        formatted_preds = []
        for pred in raw_result.get('valid_predictions', []):
            formatted_preds.append({
                'smiles': pred,
                'score': 1.0  # RSGPT doesn't provide normalized scores
            })
        results.append({
            'input': smi,
            'predictions': formatted_preds
        })
    return results


if __name__ == '__main__':
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='RSGPT Retrosynthesis Prediction')
    parser.add_argument('smiles', type=str, help='Product SMILES string')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device (cuda:0, cpu, etc.)')
    parser.add_argument('--beam-size', type=int, default=10, help='Beam search size')
    parser.add_argument('--max-length', type=int, default=100, help='Max generation length')

    args = parser.parse_args()

    predictor = RSGPTPredictor(device=args.device)
    result = predictor.run(args.smiles, beam_size=args.beam_size, max_length=args.max_length)

    print(f"\nProduct: {result['product']}")
    print(f"\nPredicted reactants ({len(result['valid_predictions'])} valid):")
    for i, pred in enumerate(result['valid_predictions'][:10], 1):
        print(f"  {i}. {pred}")
