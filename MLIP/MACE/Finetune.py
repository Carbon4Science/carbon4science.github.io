"""MACE fine-tuning module."""

import os
import subprocess
import json
import re


def finetune(train_file, val_file, output_dir, epochs=50, device="cuda"):
    """Fine-tune MACE-MP-0 medium on extxyz data using mace_run_train CLI.

    Returns:
        dict with checkpoint_path and val_errors
    """
    model_name = "MACE_finetuned"

    cmd = [
        "mace_run_train",
        f"--name={model_name}",
        "--foundation_model=medium",
        "--multiheads_finetuning=False",
        f"--train_file={os.path.abspath(train_file)}",
        f"--valid_file={os.path.abspath(val_file)}",
        "--energy_key=energy",
        "--forces_key=forces",
        "--stress_key=stress",
        "--energy_weight=1.0",
        "--forces_weight=1.0",
        "--stress_weight=1.0",
        "--E0s=average",
        "--lr=0.01",
        "--scaling=rms_forces_scaling",
        "--batch_size=16",
        f"--max_num_epochs={epochs}",
        "--ema",
        "--ema_decay=0.99",
        "--amsgrad",
        "--default_dtype=float64",
        f"--device={device}",
        "--seed=3",
        f"--model_dir={os.path.abspath(output_dir)}",
        f"--checkpoints_dir={os.path.abspath(os.path.join(output_dir, 'checkpoints'))}",
        f"--results_dir={os.path.abspath(os.path.join(output_dir, 'results'))}",
        f"--log_dir={os.path.abspath(output_dir)}",
    ]

    print(f"Running: {' '.join(cmd[:5])} ...")
    subprocess.run(cmd, check=True)

    checkpoint_path = os.path.join(output_dir, f"{model_name}.model")
    if not os.path.exists(checkpoint_path):
        # Try compiled version
        compiled = os.path.join(output_dir, f"{model_name}_compiled.model")
        if os.path.exists(compiled):
            checkpoint_path = compiled

    # Parse validation errors from MACE results
    val_errors = _parse_results(output_dir, model_name)

    return {
        "checkpoint_path": checkpoint_path,
        "val_errors": val_errors,
    }


def _parse_results(output_dir, model_name):
    """Parse MACE training results for validation errors."""
    val_errors = {}
    results_dir = os.path.join(output_dir, "results")
    if not os.path.isdir(results_dir):
        return val_errors

    # Look for results JSON files
    for fname in os.listdir(results_dir):
        if fname.endswith(".txt") and "valid" in fname.lower():
            try:
                with open(os.path.join(results_dir, fname)) as f:
                    content = f.read()
                # Parse MAE values from MACE log format
                for match in re.finditer(r"(\w+)_MAE=([0-9.e+-]+)", content):
                    key, value = match.group(1), float(match.group(2))
                    val_errors[f"{key}_MAE"] = value
            except Exception:
                pass

    # Also check the log file in output_dir
    log_file = os.path.join(output_dir, f"{model_name}.log")
    if os.path.exists(log_file):
        try:
            with open(log_file) as f:
                lines = f.readlines()
            # Find last validation line
            for line in reversed(lines):
                if "valid" in line.lower() and "mae" in line.lower():
                    for match in re.finditer(r"(\w+)_MAE=([0-9.e+-]+)", line):
                        key, value = match.group(1), float(match.group(2))
                        val_errors[f"{key}_MAE"] = value
                    break
        except Exception:
            pass

    return val_errors
