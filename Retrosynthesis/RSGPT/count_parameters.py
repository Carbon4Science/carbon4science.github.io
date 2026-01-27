#!/usr/bin/env python3
"""
Script to count parameters of the RxnGPT model.
Uses the model configuration from configs/rxngpt_llama1B.json
"""

import json
from transformers import LlamaForCausalLM, LlamaConfig


def count_parameters(model):
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def format_number(num):
    """Format large numbers with B/M/K suffix."""
    if num >= 1e9:
        return f"{num / 1e9:.2f}B ({num:,})"
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M ({num:,})"
    elif num >= 1e3:
        return f"{num / 1e3:.2f}K ({num:,})"
    return str(num)


def main():
    # Load config from file
    config_path = "configs/rxngpt_llama1B.json"
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    print("=" * 60)
    print("RxnGPT Model Parameter Analysis")
    print("=" * 60)

    # Display model configuration
    print("\nModel Configuration:")
    print("-" * 40)
    print(f"  Hidden Size:          {config_dict['hidden_size']}")
    print(f"  Intermediate Size:    {config_dict['intermediate_size']}")
    print(f"  Num Hidden Layers:    {config_dict['num_hidden_layers']}")
    print(f"  Num Attention Heads:  {config_dict['num_attention_heads']}")
    print(f"  Vocab Size:           {config_dict['vocab_size']}")
    print(f"  Architecture:         {config_dict['architectures'][0]}")

    # Create model from config
    config = LlamaConfig.from_pretrained(config_path)
    model = LlamaForCausalLM(config)

    # Count parameters
    total_params, trainable_params = count_parameters(model)

    print("\n" + "=" * 60)
    print("Parameter Count Results")
    print("=" * 60)
    print(f"\n  Total Parameters:     {format_number(total_params)}")
    print(f"  Trainable Parameters: {format_number(trainable_params)}")

    # Breakdown by component
    print("\n" + "-" * 60)
    print("Parameter Breakdown by Component:")
    print("-" * 60)

    embed_params = sum(p.numel() for name, p in model.named_parameters() if 'embed' in name)
    lm_head_params = sum(p.numel() for name, p in model.named_parameters() if 'lm_head' in name)
    layer_params = sum(p.numel() for name, p in model.named_parameters()
                       if 'layers' in name)
    norm_params = sum(p.numel() for name, p in model.named_parameters()
                      if 'norm' in name and 'layers' not in name)

    print(f"  Embedding Layer:      {format_number(embed_params)}")
    print(f"  LM Head:              {format_number(lm_head_params)}")
    print(f"  Transformer Layers:   {format_number(layer_params)}")
    print(f"  Final Norm:           {format_number(norm_params)}")

    # Per-layer breakdown
    print("\n" + "-" * 60)
    print("Per Transformer Layer Breakdown (Layer 0 as example):")
    print("-" * 60)

    layer0_attn = sum(p.numel() for name, p in model.named_parameters()
                      if 'layers.0.self_attn' in name)
    layer0_mlp = sum(p.numel() for name, p in model.named_parameters()
                     if 'layers.0.mlp' in name)
    layer0_norm = sum(p.numel() for name, p in model.named_parameters()
                      if 'layers.0' in name and 'norm' in name)

    print(f"  Self-Attention:       {format_number(layer0_attn)}")
    print(f"  MLP (FFN):            {format_number(layer0_mlp)}")
    print(f"  Layer Norms:          {format_number(layer0_norm)}")
    print(f"  Per-Layer Total:      {format_number(layer0_attn + layer0_mlp + layer0_norm)}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
