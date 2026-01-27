"""
Script to count the number of parameters in the Chemformer BART model.
This recreates the model architecture to compute parameter count.
"""

import json
import torch
import torch.nn as nn


class PreNormEncoderLayer(nn.Module):
    """Pre-norm encoder layer matching Chemformer's implementation."""
    def __init__(self, d_model, num_heads, d_feedforward, dropout, activation):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = getattr(nn.functional, activation)


class PreNormDecoderLayer(nn.Module):
    """Pre-norm decoder layer matching Chemformer's implementation."""
    def __init__(self, d_model, num_heads, d_feedforward, dropout, activation):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = getattr(nn.functional, activation)


class BARTModel(nn.Module):
    """
    Recreate Chemformer's BART model architecture for parameter counting.
    """
    def __init__(
        self,
        vocabulary_size,
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_feedforward=2048,
        max_seq_len=512,
        dropout=0.1,
        activation="gelu",
        pad_token_idx=0,
    ):
        super().__init__()

        self.d_model = d_model
        self.vocabulary_size = vocabulary_size

        # Embedding layer
        self.emb = nn.Embedding(vocabulary_size, d_model, padding_idx=pad_token_idx)
        self.dropout = nn.Dropout(dropout)

        # Encoder
        encoder_layer = PreNormEncoderLayer(d_model, num_heads, d_feedforward, dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, norm=encoder_norm)

        # Decoder
        decoder_layer = PreNormDecoderLayer(d_model, num_heads, d_feedforward, dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers, norm=decoder_norm)

        # Output projection
        self.token_fc = nn.Linear(d_model, vocabulary_size)
        self.log_softmax = nn.LogSoftmax(dim=2)


def count_parameters(model, detailed=False):
    """Count total and trainable parameters in a model."""
    total_params = 0
    trainable_params = 0

    param_details = {}

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params

        if detailed:
            # Group by layer type
            layer_type = name.split('.')[0]
            if layer_type not in param_details:
                param_details[layer_type] = 0
            param_details[layer_type] += num_params

    return total_params, trainable_params, param_details


def format_number(n):
    """Format large numbers with commas and also show in millions."""
    if n >= 1_000_000:
        return f"{n:,} ({n/1_000_000:.2f}M)"
    else:
        return f"{n:,}"


def main():
    # Load vocabulary to get size
    with open("bart_vocab.json", "r") as f:
        vocab = json.load(f)
    vocabulary_size = len(vocab["vocabulary"])

    print("=" * 60)
    print("Chemformer BART Model Parameter Count")
    print("=" * 60)

    # Default model configuration from pretrain.yaml
    config = {
        "vocabulary_size": vocabulary_size,
        "d_model": 512,
        "num_layers": 6,
        "num_heads": 8,
        "d_feedforward": 2048,
        "max_seq_len": 512,
        "dropout": 0.1,
        "activation": "gelu",
    }

    print("\nModel Configuration:")
    print("-" * 40)
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Create model
    model = BARTModel(**config)

    # Count parameters
    total, trainable, details = count_parameters(model, detailed=True)

    print("\n" + "=" * 60)
    print("Parameter Count Summary")
    print("=" * 60)
    print(f"\nTotal parameters:     {format_number(total)}")
    print(f"Trainable parameters: {format_number(trainable)}")

    print("\nBreakdown by Component:")
    print("-" * 40)
    for component, count in sorted(details.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        print(f"  {component:20s}: {format_number(count):>25s} ({pct:5.1f}%)")

    # Detailed layer breakdown
    print("\n" + "=" * 60)
    print("Detailed Parameter Breakdown")
    print("=" * 60)

    # Embedding
    emb_params = vocabulary_size * config["d_model"]
    print(f"\n1. Embedding Layer:")
    print(f"   Shape: ({vocabulary_size}, {config['d_model']})")
    print(f"   Parameters: {format_number(emb_params)}")

    # Encoder layer calculation
    d_model = config["d_model"]
    d_ff = config["d_feedforward"]
    n_heads = config["num_heads"]
    n_layers = config["num_layers"]

    # Self-attention: Q, K, V projections + output projection
    attn_params = 4 * (d_model * d_model + d_model)  # Q, K, V, O projections with biases

    # FFN: linear1 + linear2
    ffn_params = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)

    # Layer norms: 2 in encoder layer
    ln_params = 2 * (d_model + d_model)  # weight + bias for each

    encoder_layer_params = attn_params + ffn_params + ln_params

    print(f"\n2. Single Encoder Layer:")
    print(f"   Self-Attention: {format_number(attn_params)}")
    print(f"   FFN: {format_number(ffn_params)}")
    print(f"   LayerNorms: {format_number(ln_params)}")
    print(f"   Total per layer: {format_number(encoder_layer_params)}")
    print(f"   Total encoder ({n_layers} layers + final norm): {format_number(encoder_layer_params * n_layers + d_model * 2)}")

    # Decoder layer: has extra cross-attention
    cross_attn_params = 4 * (d_model * d_model + d_model)  # Q, K, V, O projections
    decoder_ln_params = 3 * (d_model + d_model)  # 3 layer norms
    decoder_layer_params = attn_params + cross_attn_params + ffn_params + decoder_ln_params

    print(f"\n3. Single Decoder Layer:")
    print(f"   Self-Attention: {format_number(attn_params)}")
    print(f"   Cross-Attention: {format_number(cross_attn_params)}")
    print(f"   FFN: {format_number(ffn_params)}")
    print(f"   LayerNorms: {format_number(decoder_ln_params)}")
    print(f"   Total per layer: {format_number(decoder_layer_params)}")
    print(f"   Total decoder ({n_layers} layers + final norm): {format_number(decoder_layer_params * n_layers + d_model * 2)}")

    # Output projection
    output_params = d_model * vocabulary_size + vocabulary_size
    print(f"\n4. Output Projection (token_fc):")
    print(f"   Shape: ({d_model}, {vocabulary_size})")
    print(f"   Parameters: {format_number(output_params)}")

    # Memory estimation
    bytes_per_param = 4  # float32
    memory_mb = (total * bytes_per_param) / (1024 * 1024)
    memory_fp16_mb = (total * 2) / (1024 * 1024)

    print("\n" + "=" * 60)
    print("Memory Estimation")
    print("=" * 60)
    print(f"  FP32 (training): ~{memory_mb:.1f} MB")
    print(f"  FP16 (inference): ~{memory_fp16_mb:.1f} MB")
    print(f"  Note: Actual memory usage during training is ~3-4x due to gradients and optimizer states")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
