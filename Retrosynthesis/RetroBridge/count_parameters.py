"""
Script to count the number of parameters in the trained RetroBridge model.
"""
import torch

def count_parameters(state_dict):
    """Count total and trainable parameters from a state dict."""
    total_params = 0
    params_by_module = {}

    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            num_params = param.numel()
            total_params += num_params

            # Group by top-level module
            module_name = name.split('.')[0]
            if module_name not in params_by_module:
                params_by_module[module_name] = 0
            params_by_module[module_name] += num_params

    return total_params, params_by_module

def format_params(num):
    """Format parameter count with commas and human-readable form."""
    if num >= 1e9:
        return f"{num:,} ({num/1e9:.2f}B)"
    elif num >= 1e6:
        return f"{num:,} ({num/1e6:.2f}M)"
    elif num >= 1e3:
        return f"{num:,} ({num/1e3:.2f}K)"
    return f"{num:,}"

def main():
    checkpoint_path = "models/retrobridge.ckpt"

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    print("\n" + "="*60)
    print("RetroBridge Model Parameter Count")
    print("="*60)

    # The state dict is typically under 'state_dict' key for PyTorch Lightning
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    total_params, params_by_module = count_parameters(state_dict)

    print(f"\nTotal Parameters: {format_params(total_params)}")
    print("\nParameters by Module:")
    print("-"*60)

    # Sort by parameter count
    sorted_modules = sorted(params_by_module.items(), key=lambda x: x[1], reverse=True)
    for module, count in sorted_modules:
        print(f"  {module:40s} {format_params(count)}")

    print("-"*60)
    print(f"  {'TOTAL':40s} {format_params(total_params)}")

    # Additional info from checkpoint
    if 'hyper_parameters' in checkpoint:
        print("\nModel Hyperparameters:")
        print("-"*60)
        hp = checkpoint['hyper_parameters']
        for key in ['n_layers', 'hidden_dims', 'hidden_mlp_dims']:
            if key in hp:
                print(f"  {key}: {hp[key]}")

if __name__ == "__main__":
    main()
