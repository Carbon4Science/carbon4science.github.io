"""SevenNet fine-tuning module."""

import os
import subprocess
import yaml


def finetune(train_file, val_file, output_dir, epochs=50, device="cuda"):
    """Fine-tune SevenNet-l3i5 on extxyz data using sevenn CLI.

    Returns:
        dict with checkpoint_path and val_errors
    """
    # Generate input.yaml config
    config = {
        "model": {
            "chemical_species": "auto",
            "cutoff": 5.0,
            "channel": 128,
            "is_parity": False,
            "lmax": 3,
            "num_convolution_layer": 5,
            "irreps_manual": [
                "128x0e",
                "128x0e+64x1e+32x2e+32x3e",
                "128x0e+64x1e+32x2e+32x3e",
                "128x0e+64x1e+32x2e+32x3e",
                "128x0e+64x1e+32x2e+32x3e",
                "128x0e",
            ],
            "weight_nn_hidden_neurons": [64, 64],
            "radial_basis": {
                "radial_basis_name": "bessel",
                "bessel_basis_num": 8,
            },
            "cutoff_function": {
                "cutoff_function_name": "poly_cut",
                "poly_cut_p_value": 6,
            },
            "self_connection_type": "linear",
            "train_shift_scale": True,
            "train_denominator": False,
        },
        "train": {
            "random_seed": 1,
            "is_train_stress": True,
            "epoch": epochs,
            "loss": "Huber",
            "loss_param": {"delta": 0.01},
            "optimizer": "adam",
            "optim_param": {"lr": 0.0001},
            "scheduler": "linearlr",
            "scheduler_param": {
                "start_factor": 1.0,
                "total_iters": epochs,
                "end_factor": 0.000001,
            },
            "force_loss_weight": 1.0,
            "stress_loss_weight": 0.01,
            "per_epoch": 10,
            "error_record": [
                ["Energy", "RMSE"],
                ["Force", "RMSE"],
                ["Stress", "RMSE"],
                ["TotalLoss", "None"],
            ],
            "continue": {
                "checkpoint": "7net-l3i5",
                "reset_optimizer": True,
                "reset_scheduler": True,
                "reset_epoch": True,
                "use_statistic_values_of_checkpoint": True,
            },
        },
        "data": {
            "batch_size": 16,
            "shift": "elemwise_reference_energies",
            "scale": "force_rms",
            "data_format_args": {"index": ":"},
            "load_trainset_path": [os.path.abspath(train_file)],
            "load_validset_path": [os.path.abspath(val_file)],
        },
    }

    config_path = os.path.join(output_dir, "input.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Config written to: {config_path}")
    print(f"Training for {epochs} epochs...")

    # Run sevenn train from the output directory
    cmd = ["sevenn", "train", "input.yaml", "-s"]
    subprocess.run(cmd, check=True, cwd=output_dir)

    # Find best checkpoint
    checkpoint_path = os.path.join(output_dir, "checkpoint_best.pth")
    if not os.path.exists(checkpoint_path):
        # Look for any checkpoint
        for f in sorted(os.listdir(output_dir)):
            if f.startswith("checkpoint_") and f.endswith(".pth"):
                checkpoint_path = os.path.join(output_dir, f)
                break

    # Parse validation errors from log
    val_errors = _parse_log(output_dir)

    return {
        "checkpoint_path": checkpoint_path,
        "val_errors": val_errors,
    }


def _parse_log(output_dir):
    """Parse log.sevenn for validation errors."""
    val_errors = {}
    log_path = os.path.join(output_dir, "log.sevenn")
    if not os.path.exists(log_path):
        return val_errors

    try:
        with open(log_path) as f:
            lines = f.readlines()

        # Find last validation error line
        for line in reversed(lines):
            if "Energy" in line and "RMSE" in line:
                # Parse RMSE values
                parts = line.strip().split()
                for i, part in enumerate(parts):
                    if "Energy" in part and i + 1 < len(parts):
                        try:
                            val_errors["energy_RMSE"] = float(parts[i + 1])
                        except (ValueError, IndexError):
                            pass
                    if "Force" in part and i + 1 < len(parts):
                        try:
                            val_errors["force_RMSE"] = float(parts[i + 1])
                        except (ValueError, IndexError):
                            pass
                    if "Stress" in part and i + 1 < len(parts):
                        try:
                            val_errors["stress_RMSE"] = float(parts[i + 1])
                        except (ValueError, IndexError):
                            pass
                if val_errors:
                    break
    except Exception:
        pass

    return val_errors
