"""
LlaSMol retrosynthesis prediction inference.

Reference: Yu et al., "LlaSMol: Advancing Large Language Models for
Chemistry with a Large-Scale, Comprehensive, High-Quality Instruction
Tuning Dataset", COLM, 2024.

GitHub: https://github.com/OSU-NLP-Group/LLM4Chem

Uses HuggingFace Transformers with a fine-tuned Mistral-7B model.

Implements the uniform inference interface for the benchmark runner.
"""

import os
import sys
import re
from typing import List, Dict, Union

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

_DEFAULT_CONFIG = {
    'model_name': 'osunlp/LlaSMol-Mistral-7B',
    'base_model': 'mistralai/Mistral-7B-v0.1',
    'model_path': None,
    'max_new_tokens': 960,
    'num_beams': 13,
    'num_return_sequences': 10,
    'device': 'cuda:0',
}

_model = None
_tokenizer = None

RETRO_PROMPT_TEMPLATE = (
    "<SMILES> {product} </SMILES> "
    "Based on the given product, suggest possible reactants "
    "that could have been used in the reaction."
)

CHAT_TEMPLATE = "<s>[INST] {prompt} [/INST]"


def _load_model():
    """Lazy-load the LlaSMol model (base Mistral-7B + LoRA adapter) and tokenizer."""
    global _model, _tokenizer
    if _model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"[LlaSMol-Retro] Loading model... CUDA available: {torch.cuda.is_available()}", flush=True)

        adapter_id = _DEFAULT_CONFIG['model_path'] or _DEFAULT_CONFIG['model_name']
        base_model_id = _DEFAULT_CONFIG['base_model']
        device = _DEFAULT_CONFIG['device']
        if not torch.cuda.is_available():
            device = 'cpu'

        dtype = torch.bfloat16 if 'cuda' in device else torch.float32

        _tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        _tokenizer.padding_side = 'left'
        _tokenizer.pad_token_id = 0
        _tokenizer.sep_token = '<unk>'
        _tokenizer.cls_token = '<unk>'
        _tokenizer.mask_token = '<unk>'

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=dtype,
            device_map=device,
        )

        try:
            from peft import PeftModel as _PeftModel
            peft_model = _PeftModel.from_pretrained(base_model, adapter_id)
            _model = peft_model.merge_and_unload()
            _model.eval()
            print(f"[LlaSMol-Retro] Model loaded + LoRA merged on {device}", flush=True)
        except Exception as e:
            print(f"[LlaSMol-Retro] FATAL: Failed to load LoRA adapter: {e}", flush=True)
            import traceback
            traceback.print_exc()
            _model = base_model
            _model.eval()

        _DEFAULT_CONFIG['device'] = device
    return _model, _tokenizer


def load_model(model_path=None, **kwargs):
    """Load model checkpoint. Called by benchmark runner."""
    if model_path:
        _DEFAULT_CONFIG['model_path'] = model_path


def _extract_smiles(text: str) -> str:
    """Extract SMILES from model output."""
    for pattern in [
        r'<SMILES>\s*(.+?)\s*</SMILES>',
        r'\[SMILES\]\s*(.+?)\s*</SMILES>',
    ]:
        match = re.search(pattern, text)
        if match:
            smi = match.group(1).strip()
            while smi.startswith('[') and smi.count('[') > smi.count(']'):
                smi = smi[1:]
            return smi

    match = re.search(r'([A-Za-z0-9@+\-\[\]()=#/\\\\.]+)\s*</SMILES>', text)
    if match:
        return match.group(1).strip()

    first_line = text.strip().split('\n')[0].strip()
    if first_line:
        first_line = first_line.split(' ')[0].strip()
        if any(c in first_line for c in ['C', 'c', 'N', 'O', '(', ')', '=', '#']):
            if not any(word in first_line.lower() for word in ['the', 'product', 'reaction', 'is', 'can', 'be', 'predict', 'note']):
                return first_line
    return ''


def run(smiles: Union[str, List[str]], top_k: int = 10) -> List[Dict]:
    """
    Predict reactants from product using LlaSMol.

    Args:
        smiles: Product SMILES string or list of SMILES strings
        top_k: Number of top predictions to return

    Returns:
        [{'input': '...', 'predictions': [{'smiles': '...', 'score': 0.95}, ...]}]
    """
    model, tokenizer = _load_model()

    if isinstance(smiles, str):
        smiles_list = [smiles]
    else:
        smiles_list = list(smiles)

    import torch

    results = []
    for smi in smiles_list:
        try:
            instruction = RETRO_PROMPT_TEMPLATE.format(product=smi)
            prompt = CHAT_TEMPLATE.format(prompt=instruction)

            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = inputs.input_ids.to(model.device)
            attention_mask = inputs.attention_mask.to(model.device)
            input_len = input_ids.shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=_DEFAULT_CONFIG['max_new_tokens'],
                    num_beams=_DEFAULT_CONFIG['num_beams'],
                    num_return_sequences=min(top_k, _DEFAULT_CONFIG['num_return_sequences']),
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            formatted_preds = []
            from rdkit import Chem
            seen = set()

            sequences = outputs.sequences
            if hasattr(outputs, 'sequences_scores') and outputs.sequences_scores is not None:
                seq_scores = outputs.sequences_scores.cpu().tolist()
            else:
                seq_scores = [1.0 / (i + 1) for i in range(len(sequences))]

            for seq, score in zip(sequences, seq_scores):
                generated = tokenizer.decode(seq[input_len:], skip_special_tokens=True)
                pred_smi = _extract_smiles(generated)
                pred_smi = pred_smi.replace(';', '.')

                if not pred_smi:
                    continue

                mol = Chem.MolFromSmiles(pred_smi)
                if mol is None:
                    continue

                canon = Chem.MolToSmiles(mol)
                if canon in seen:
                    continue
                seen.add(canon)

                import math
                prob = math.exp(float(score)) if score < 0 else float(score)

                formatted_preds.append({
                    'smiles': canon,
                    'score': prob,
                })

                if len(formatted_preds) >= top_k:
                    break

        except Exception as e:
            import traceback
            print(f"[LlaSMol-Retro] Error on '{smi[:50]}': {e}", flush=True)
            traceback.print_exc()
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
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    results = run(smiles, top_k=top_k)
    for r in results:
        print(f"Input: {r['input']}")
        for i, p in enumerate(r['predictions'], 1):
            print(f"  {i}. {p['smiles']} (score: {p['score']:.4f})")
