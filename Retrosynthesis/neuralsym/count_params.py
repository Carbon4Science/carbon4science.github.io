#!/usr/bin/env python
"""
Script to evaluate the number of parameters in the NeuralSym trained model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Model architecture (copied from model.py to avoid dependency issues)
class Highway(nn.Module):
    def __init__(self, size, num_layers, f, dropout,
                head=False, input_size=None):
        super(Highway, self).__init__()
        self.num_layers = num_layers
        self.f = f
        self.dropout = nn.Dropout(dropout)
        if head:
            assert input_size is not None
            self.nonlinear = nn.ModuleList([nn.Linear(input_size, size)])
            self.linear = nn.ModuleList([nn.Linear(input_size, size)])
            self.gate = nn.ModuleList([nn.Linear(input_size, size)])
        else:
            self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
            self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
            self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

class TemplateNN_Highway(nn.Module):
    def __init__(self, output_size, size=512, num_layers_body=5,
                dropout_head=0.3, dropout_body=0.1,
                f=F.elu, input_size=32681):
        super(TemplateNN_Highway, self).__init__()
        self.highway_head = Highway(
                size=size, num_layers=1, f=f, dropout=dropout_head,
                head=True, input_size=input_size
            )
        if num_layers_body <= 0:
            self.highway_body = None
        else:
            self.highway_body = Highway(
                    size=size, num_layers=num_layers_body,
                    f=f, dropout=dropout_body
                )

        self.classifier = nn.Linear(size, output_size)


def count_parameters(model, verbose=True):
    """Count total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if verbose:
        print("\n" + "="*60)
        print("MODEL PARAMETER COUNT")
        print("="*60)

        # Detailed breakdown by layer
        print("\nParameter breakdown by layer:")
        print("-"*60)
        for name, param in model.named_parameters():
            print(f"  {name:50s} {param.numel():>12,} params  {list(param.shape)}")

        print("-"*60)
        print(f"\n{'Total parameters:':<50} {total_params:>12,}")
        print(f"{'Trainable parameters:':<50} {trainable_params:>12,}")
        print(f"{'Non-trainable parameters:':<50} {total_params - trainable_params:>12,}")

        # Memory estimate (assuming float32)
        memory_mb = total_params * 4 / (1024 * 1024)
        print(f"\n{'Estimated model size (float32):':<50} {memory_mb:>12.2f} MB")
        print("="*60)

    return total_params, trainable_params


def main():
    # Configuration from infer_config.py (default trained model)
    DATA_FOLDER = Path(__file__).resolve().parent / 'data'

    # Count templates to get output size
    templates_file = DATA_FOLDER / '50k_training_templates'
    min_freq = 1

    with open(templates_file, 'r') as f:
        templates = f.readlines()

    templates_filtered = []
    for p in templates:
        pa, cnt = p.strip().split(': ')
        if int(cnt) >= min_freq:
            templates_filtered.append(pa)

    output_size = len(templates_filtered)

    # Model configuration (from infer_config.py - the default trained model)
    config = {
        'hidden_size': 300,
        'depth': 0,  # num_layers_body
        'input_size': 32681,
        'output_size': output_size
    }

    print("\n" + "="*60)
    print("MODEL CONFIGURATION")
    print("="*60)
    print(f"  Model type:        TemplateNN_Highway")
    print(f"  Input size:        {config['input_size']:,} (ECFP4 fingerprint features)")
    print(f"  Hidden size:       {config['hidden_size']}")
    print(f"  Highway body depth: {config['depth']}")
    print(f"  Output size:       {config['output_size']:,} (reaction templates)")
    print("="*60)

    # Create model with the same configuration as the trained model
    model = TemplateNN_Highway(
        output_size=config['output_size'],
        size=config['hidden_size'],
        num_layers_body=config['depth'],
        input_size=config['input_size']
    )

    # Count parameters
    total, trainable = count_parameters(model)

    # Also show alternative configurations for comparison
    print("\n\nCOMPARISON WITH OTHER CONFIGURATIONS:")
    print("="*60)

    configs = [
        {'name': 'Default (from infer_config)', 'hidden': 300, 'depth': 0},
        {'name': 'Paper default (512 hidden, depth 5)', 'hidden': 512, 'depth': 5},
        {'name': 'Smaller (256 hidden, depth 0)', 'hidden': 256, 'depth': 0},
    ]

    for cfg in configs:
        m = TemplateNN_Highway(
            output_size=output_size,
            size=cfg['hidden'],
            num_layers_body=cfg['depth'],
            input_size=32681
        )
        params = sum(p.numel() for p in m.parameters())
        print(f"  {cfg['name']:45s}: {params:>12,} params")

    print("="*60)


if __name__ == "__main__":
    main()
