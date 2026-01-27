#!/usr/bin/env python3
"""
Script to run RSGPT inference on example molecules.
Tests the model on a few example SMILES to verify it works.
"""

import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaConfig
from rdkit import Chem
import json
import re


def load_tokenizer():
    """Load the SMILES BPE tokenizer."""
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file("vocab.json")  # vocab.json has the correct 1000 token vocab
    return tokenizer


def beam_search_gpt(model, tokenizer, input_ids, beam_size=10, max_length=100, device='cpu'):
    """Beam search decoding for the model."""
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)

    sequences = [(input_ids, 0.0)]  # (sequence, cumulative score)
    end_token_id = tokenizer.encode('</s>').ids[0]

    completed_sequences = []

    for _ in range(max_length):
        all_candidates = []
        for seq, score in sequences:
            if seq[0, -1].item() == end_token_id:
                completed_sequences.append((seq, score))
                continue

            with torch.no_grad():
                outputs = model(input_ids=seq)
                logits = outputs.logits[:, -1, :]
            logits = F.log_softmax(logits, dim=-1)

            topk_probs, topk_indices = torch.topk(logits, beam_size, dim=-1)

            for i in range(beam_size):
                candidate_seq = torch.cat([seq, topk_indices[:, i].unsqueeze(0)], dim=1)
                candidate = (candidate_seq, score - topk_probs[0, i].item())
                all_candidates.append(candidate)

        if not all_candidates:
            break

        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        sequences = ordered[:beam_size]

    completed_sequences.extend(sequences)
    completed_sequences = sorted(completed_sequences, key=lambda tup: tup[1])

    # Decode sequences
    decoded_sequences = []
    for seq, score in completed_sequences[:beam_size]:
        tokens = seq[0].tolist()
        decoded = tokenizer.decode(tokens)
        decoded_sequences.append(decoded)

    return decoded_sequences


def extract_reactants(prediction):
    """Extract reactants from the model prediction."""
    # Pattern: <s><Isyn><O>PRODUCT<F1>REACTANT1<F2>REACTANT2...</s>
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


def canonicalize_smiles(smiles):
    """Canonicalize a SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except:
        return None


def main():
    device = 'cpu'  # Use CPU for inference without CUDA

    print("=" * 70)
    print("RSGPT Inference Test")
    print("=" * 70)

    # Example molecules to test (products for retrosynthesis)
    test_molecules = [
        # From the paper/repo examples
        "N#CC1=C(OCC(C)C)C=CC(C2=NC(C)=C(C(O)=O)S2)=C1",
        # Simple aspirin
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        # Ibuprofen
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        # Paracetamol/Acetaminophen
        "CC(=O)NC1=CC=C(C=C1)O",
    ]

    print("\nLoading tokenizer...")
    tokenizer = load_tokenizer()

    print("Loading model configuration...")
    config = LlamaConfig.from_pretrained("configs/rxngpt_llama1B.json")

    print("Creating model...")
    model = LlamaForCausalLM(config)

    print("Loading pretrained weights...")
    weights_path = "weights/finetune_50k.pth"
    if not os.path.exists(weights_path):
        print(f"ERROR: Model weights not found at {weights_path}")
        print("Please download from: https://sandbox.zenodo.org/records/203391")
        return

    state_dict = torch.load(weights_path, map_location=device)

    # Handle DDP 'module.' prefix and map to LlamaForCausalLM structure
    # RxnGPT inherits LlamaModel and has self.model = LlamaForCausalLM
    # Checkpoint has both module.X and module.model.model.X (duplicate due to inheritance)
    # We need to map: module.X -> model.X for LlamaForCausalLM

    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove 'module.' DDP prefix
        if k.startswith('module.'):
            k = k[7:]

        # Skip the nested model.model.X keys (duplicates)
        if k.startswith('model.model.'):
            continue
        if k.startswith('model.lm_head'):
            # model.lm_head -> lm_head
            new_state_dict[k[6:]] = v
        elif k.startswith('model.'):
            # Skip other model.X (nested structure)
            continue
        else:
            # embed_tokens, layers, norm -> model.embed_tokens, model.layers, model.norm
            new_state_dict['model.' + k] = v

    print(f"Mapped {len(new_state_dict)} weight keys")

    # Load with strict=False to handle any missing keys
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")

    model.eval()
    model.to(device)

    print(f"\nModel loaded successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "=" * 70)
    print("Running Predictions")
    print("=" * 70)

    for i, product in enumerate(test_molecules):
        print(f"\n--- Example {i+1} ---")

        # Canonicalize input
        canonical = canonicalize_smiles(product)
        if canonical is None:
            print(f"Invalid SMILES: {product}")
            continue

        print(f"Product: {canonical}")

        # Format input as expected by the model
        # Format: <s><Isyn><O>PRODUCT<F1>
        input_text = f"<s><Isyn><O>{canonical}<F1>"
        input_ids = tokenizer.encode(input_text).ids

        print(f"Input tokens: {len(input_ids)}")

        # Run beam search
        print("Running beam search (this may take a moment on CPU)...")
        try:
            predictions = beam_search_gpt(
                model, tokenizer, input_ids,
                beam_size=5, max_length=50, device=device
            )

            print("\nTop predictions:")
            for j, pred in enumerate(predictions[:5]):
                reactants = extract_reactants(pred)
                if reactants:
                    # Validate reactants
                    valid = canonicalize_smiles(reactants)
                    status = "VALID" if valid else "INVALID"
                    print(f"  {j+1}. {reactants} [{status}]")
                else:
                    print(f"  {j+1}. (Could not parse: {pred[:80]}...)")

        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("Inference complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
