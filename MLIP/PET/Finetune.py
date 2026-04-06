"""PET fine-tuning module."""

import os
import subprocess
import yaml


def finetune(train_file, val_file, output_dir, epochs=50, device="cuda"):
    """Fine-tune PET-OAM-XL on extxyz data using metatrain (mtt) CLI.

    Returns:
        dict with checkpoint_path and val_errors
    """
    # Find the pretrained checkpoint path
    checkpoint_path_pretrained = _find_pet_checkpoint()

    # Generate metatrain YAML config
    # metatrain requires structured dataset config with length_unit and targets
    config = {
        "architecture": {
            "name": "pet",
            "training": {
                "finetune": {
                    "method": "full",
                    "read_from": checkpoint_path_pretrained,
                },
                "num_epochs": epochs,
                "batch_size": 1,
                "learning_rate": 1e-5,
            },
        },
        "training_set": [
            {
                "systems": {
                    "read_from": os.path.abspath(train_file),
                    "length_unit": "angstrom",
                },
                "targets": {
                    "energy": {"key": "energy", "unit": "eV"},
                },
            }
        ],
        "validation_set": [
            {
                "systems": {
                    "read_from": os.path.abspath(val_file),
                    "length_unit": "angstrom",
                },
                "targets": {
                    "energy": {"key": "energy", "unit": "eV"},
                },
            }
        ],
    }

    config_path = os.path.join(output_dir, "options.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Config written to: {config_path}")
    print(f"Training for {epochs} epochs...")

    # Run mtt train
    cmd = ["mtt", "train", "options.yaml"]
    subprocess.run(cmd, check=True, cwd=output_dir)

    # Find output model (mtt train saves final model as model.pt)
    checkpoint_path = os.path.join(output_dir, "model.pt")
    if not os.path.exists(checkpoint_path):
        # Look for alternatives
        for name in ["model.pt", "model.ckpt", "best.ckpt", "last.ckpt"]:
            p = os.path.join(output_dir, name)
            if os.path.exists(p):
                checkpoint_path = p
                break
        # Check outputs subdirectory for checkpoints
        outputs_dir = os.path.join(output_dir, "outputs")
        if os.path.isdir(outputs_dir):
            for root, dirs, files in os.walk(outputs_dir):
                for f in files:
                    if f.endswith((".pt", ".ckpt")):
                        checkpoint_path = os.path.join(root, f)
                        break

    # Parse validation errors
    val_errors = _parse_metrics(output_dir)

    return {
        "checkpoint_path": checkpoint_path,
        "val_errors": val_errors,
    }


def _find_pet_checkpoint():
    """Find the pre-downloaded PET-OAM-XL checkpoint."""
    import glob

    # Check common locations (HuggingFace cache)
    candidates = [
        os.path.expanduser("~/.cache/huggingface/hub/models--lab-cosmo--upet/snapshots/*/models/pet-oam-xl-v1.0.0.ckpt"),
        os.path.expanduser("~/.cache/huggingface/hub/models--lab-cosmo--upet/snapshots/*/models/pet-oam-xl-*.ckpt"),
        os.path.expanduser("~/.cache/upet/pet-oam-xl/*/model.ckpt"),
        os.path.expanduser("~/.cache/metatrain/pet-oam-xl/*/model.ckpt"),
    ]

    for pattern in candidates:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[-1]  # most recent

    # Try to get path from upet by triggering download
    try:
        from upet.calculator import UPETCalculator
        calc = UPETCalculator(model="pet-oam-xl", version="1.0.0", device="cpu")
        # Check common attribute names for the model path
        for attr in ["model_path", "_model_path", "checkpoint_path", "_checkpoint_path"]:
            path = getattr(calc, attr, None)
            if path and os.path.exists(path):
                return path
    except Exception:
        pass

    raise FileNotFoundError(
        "Could not find PET-OAM-XL checkpoint. "
        "Make sure the model has been downloaded by running inference first."
    )


def _parse_metrics(output_dir):
    """Parse metatrain output for validation errors."""
    val_errors = {}

    # Check for metrics log
    log_path = os.path.join(output_dir, "train_log.txt")
    if not os.path.exists(log_path):
        # Try outputs directory
        import glob
        logs = glob.glob(os.path.join(output_dir, "**", "train_log.txt"), recursive=True)
        if logs:
            log_path = logs[0]

    if os.path.exists(log_path):
        try:
            with open(log_path) as f:
                lines = f.readlines()
            for line in reversed(lines):
                if "val" in line.lower() and ("mae" in line.lower() or "rmse" in line.lower()):
                    # Parse key=value pairs
                    import re
                    for match in re.finditer(r"(\w+)[=:]\s*([0-9.e+-]+)", line):
                        key, value = match.group(1), float(match.group(2))
                        val_errors[key] = value
                    if val_errors:
                        break
        except Exception:
            pass

    return val_errors
