"""EquFlash fine-tuning module.

Uses the same sevenn train CLI as SevenNet, with the 7net-mf-ompa checkpoint.
Uses the equflash conda environment.
Config based on sevenn preset: mf_ompa_fine_tune.yaml
"""

import os
import subprocess
import yaml


def finetune(train_file, val_file, output_dir, epochs=50, device="cuda"):
    """Fine-tune 7net-mf-ompa (EquFlash base model) on extxyz data using sevenn CLI.

    The fine-tuned checkpoint can be loaded with SevenNetCalculator and
    optionally used with FlashTP acceleration (enable_flash=True).

    Returns:
        dict with checkpoint_path and val_errors
    """
    # Config based on official mf_ompa_fine_tune.yaml preset
    # 7net-mf-ompa is a multi-fidelity model requiring use_modality: true
    config = {
        "model": {
            "chemical_species": "univ",
            "cutoff": 6.0,
            "channel": 128,
            "is_parity": True,
            "lmax": 3,
            "lmax_edge": -1,
            "lmax_node": -1,
            "num_convolution_layer": 5,
            "irreps_manual": [
                "128x0e",
                "128x0e+64x1o+32x2e+32x3o",
                "128x0e+64x1o+64x1e+32x2o+32x2e+32x3o+32x3e",
                "128x0o+128x0e+64x1o+64x1e+32x2o+32x2e+32x3o+32x3e",
                "128x0e+64x1o+32x2e+32x3o",
                "128x0e",
            ],
            "weight_nn_hidden_neurons": [64, 64],
            "radial_basis": {
                "radial_basis_name": "bessel",
                "bessel_basis_num": 8,
            },
            "cutoff_function": {
                "cutoff_function_name": "XPLOR",
                "cutoff_on": 5.5,
            },
            "act_radial": "silu",
            "act_scalar": {"e": "silu", "o": "tanh"},
            "act_gate": {"e": "silu", "o": "tanh"},
            "train_denominator": False,
            "train_shift_scale": False,
            "use_bias_in_linear": False,
            "use_modal_node_embedding": False,
            "use_modal_self_inter_intro": True,
            "use_modal_self_inter_outro": True,
            "use_modal_output_block": True,
            "readout_as_fcn": False,
            "self_connection_type": "nequip",
            "interaction_type": "nequip",
            "cuequivariance_config": {},
        },
        "train": {
            "random_seed": 1,
            "is_train_stress": True,
            "epoch": epochs,
            "loss": "Huber",
            "loss_param": {"delta": 0.01},
            "optimizer": "adamw",
            "optim_param": {"lr": 0.0002, "weight_decay": 0.001},
            "scheduler": "exponentiallr",
            "scheduler_param": {"gamma": 0.99},
            "force_loss_weight": 1.0,
            "stress_loss_weight": 0.01,
            "csv_log": "log.csv",
            "num_workers": 0,
            "train_shuffle": True,
            "per_epoch": 10,
            "error_record": [
                ["Energy", "MAE"],
                ["Force", "MAE"],
                ["Stress", "MAE"],
                ["Energy", "Loss"],
                ["Force", "Loss"],
                ["Stress", "Loss"],
                ["TotalLoss", "None"],
            ],
            "best_metric": "TotalLoss",
            "use_weight": False,
            "use_modality": True,
            "continue": {
                "checkpoint": "7net-mf-ompa",
                "reset_optimizer": True,
                "reset_scheduler": True,
                "reset_epoch": True,
            },
        },
        "data": {
            "dtype": "single",
            "batch_size": 4,
            "data_format_args": {},
            "preprocess_num_cores": 1,
            "compute_statistics": True,
            "use_modal_wise_shift": True,
            "use_modal_wise_scale": False,
            "load_trainset_path": [
                {
                    "data_modality": "mpa",
                    "file_list": [{"file": os.path.abspath(train_file)}],
                }
            ],
            "load_validset_path": [
                {
                    "data_modality": "mpa",
                    "file_list": [{"file": os.path.abspath(val_file)}],
                }
            ],
        },
    }

    config_path = os.path.join(output_dir, "input.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Config written to: {config_path}")
    print(f"Training for {epochs} epochs...")

    # Run sevenn train from the output directory
    # -flashTP uses FlashTP acceleration if available
    cmd = ["sevenn", "train", "-flashTP", "input.yaml", "-s"]
    subprocess.run(cmd, check=True, cwd=output_dir)

    # Find best checkpoint
    checkpoint_path = os.path.join(output_dir, "checkpoint_best.pth")
    if not os.path.exists(checkpoint_path):
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

        for line in reversed(lines):
            if "Energy" in line and ("RMSE" in line or "MAE" in line):
                parts = line.strip().split()
                for i, part in enumerate(parts):
                    if "Energy" in part and i + 1 < len(parts):
                        try:
                            val_errors["energy_MAE"] = float(parts[i + 1])
                        except (ValueError, IndexError):
                            pass
                    if "Force" in part and i + 1 < len(parts):
                        try:
                            val_errors["force_MAE"] = float(parts[i + 1])
                        except (ValueError, IndexError):
                            pass
                    if "Stress" in part and i + 1 < len(parts):
                        try:
                            val_errors["stress_MAE"] = float(parts[i + 1])
                        except (ValueError, IndexError):
                            pass
                if val_errors:
                    break
    except Exception:
        pass

    return val_errors
