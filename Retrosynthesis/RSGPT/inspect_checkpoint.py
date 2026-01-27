#!/usr/bin/env python3
"""Inspect the checkpoint to understand its structure."""

import torch

weights_path = "weights/finetune_50k.pth"
print(f"Loading checkpoint from {weights_path}...")

state_dict = torch.load(weights_path, map_location='cpu')

print(f"\nCheckpoint type: {type(state_dict)}")

if isinstance(state_dict, dict):
    print(f"Number of keys: {len(state_dict)}")

    # Check for common checkpoint structures
    if 'state_dict' in state_dict:
        print("Found 'state_dict' key")
        state_dict = state_dict['state_dict']
    elif 'model_state_dict' in state_dict:
        print("Found 'model_state_dict' key")
        state_dict = state_dict['model_state_dict']

    print("\nFirst 20 keys:")
    for i, k in enumerate(list(state_dict.keys())[:20]):
        v = state_dict[k]
        print(f"  {k}: {v.shape if hasattr(v, 'shape') else type(v)}")

    # Look for embedding layer to get vocab size
    print("\nSearching for embedding layers...")
    for k, v in state_dict.items():
        if 'embed' in k.lower() and hasattr(v, 'shape'):
            print(f"  {k}: {v.shape}")
            if len(v.shape) == 2:
                print(f"    -> Vocab size: {v.shape[0]}, Hidden size: {v.shape[1]}")

    # Count total parameters
    total_params = sum(v.numel() for v in state_dict.values() if hasattr(v, 'numel'))
    print(f"\nTotal parameters in checkpoint: {total_params:,}")
