"""NequIP fine-tuning module.

Uses nequip v2 (Hydra-based) config with 4 required sections:
run, data, trainer, training_module.
"""

import os
import subprocess
import yaml


def _find_cuda_home():
    """Auto-detect CUDA_HOME from nvcc location or known paths."""
    import shutil

    nvcc = shutil.which("nvcc")
    if nvcc:
        # nvcc is at CUDA_HOME/bin/nvcc
        return os.path.dirname(os.path.dirname(os.path.realpath(nvcc)))

    # Check known paths
    for path in ["/HL9/HCom/cuda/12.6.3", "/usr/local/cuda"]:
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "include", "cuda.h")):
            return path
    return None


def _find_nequip_package(model_id):
    """Find the cached .nequip.zip package file for a given model ID."""
    import json
    import glob

    cache_dir = os.path.expanduser("~/.nequip/model_cache")
    for meta_file in glob.glob(os.path.join(cache_dir, "*.metadata.json")):
        with open(meta_file) as f:
            meta = json.load(f)
        if meta.get("model_id") == model_id:
            zip_path = meta_file.replace(".metadata.json", ".nequip.zip")
            if os.path.exists(zip_path):
                return zip_path
    raise FileNotFoundError(
        f"Cached package for '{model_id}' not found in {cache_dir}. "
        "Run inference first to trigger download."
    )


def finetune(train_file, val_file, output_dir, epochs=50, device="cuda"):
    """Fine-tune NequIP-MP-L on extxyz data using nequip-train CLI.

    Returns:
        dict with checkpoint_path and val_errors
    """
    from ase.io import read

    # Get chemical species from training data
    train_frames = read(train_file, index=":")
    all_species = sorted(set(
        s for atoms in train_frames for s in atoms.get_chemical_symbols()
    ))
    n_train = len(train_frames)
    n_val = len(read(val_file, index=":"))

    # Resolve package path from nequip model cache
    package_path = _find_nequip_package("mir-group/NequIP-MP-L:0.1")

    # NequIP v2 Hydra config with 4 required sections
    config = {
        # Global settings
        "run": ["train"],
        "seed": 42,
        "cutoff_radius": 5.0,
        "model_type_names": "${type_names_from_package:" + package_path + "}",
        "chemical_species": "${model_type_names}",

        # Dataloader template
        "dataloader": {
            "_target_": "torch.utils.data.DataLoader",
            "batch_size": 16,
        },

        # Section 1: data
        "data": {
            "_target_": "nequip.data.datamodule.ASEDataModule",
            "seed": "${seed}",
            "train_file_path": os.path.abspath(train_file),
            "val_file_path": os.path.abspath(val_file),
            "transforms": [
                {
                    "_target_": "nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper",
                    "model_type_names": "${model_type_names}",
                    "chemical_species_to_atom_type_map": "${list_to_identity_dict:${chemical_species}}",
                },
                {
                    "_target_": "nequip.data.transforms.NeighborListTransform",
                    "r_max": "${cutoff_radius}",
                },
            ],
            "ase_args": {"format": "extxyz"},
            "train_dataloader": "${dataloader}",
            "val_dataloader": "${dataloader}",
            "stats_manager": {
                "_target_": "nequip.data.CommonDataStatisticsManager",
                "type_names": "${model_type_names}",
            },
        },

        # Section 2: trainer
        "trainer": {
            "_target_": "lightning.Trainer",
            "max_epochs": epochs,
            "check_val_every_n_epoch": 1,
            "log_every_n_steps": 5,
            "callbacks": [
                {
                    "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
                    "monitor": "val0_epoch/weighted_sum",
                    "dirpath": "${hydra:runtime.output_dir}",
                    "filename": "best",
                    "save_last": True,
                },
            ],
        },

        # Section 3: training_module
        "training_module": {
            "_target_": "nequip.train.EMALightningModule",
            "model": {
                "_target_": "nequip.model.ModelFromPackage",
                "package_path": package_path,
                "compile_mode": "eager",
            },
            "loss": {
                "_target_": "nequip.train.EnergyForceStressLoss",
                "per_atom_energy": True,
                "coeffs": {
                    "total_energy": 1.0,
                    "forces": 1.0,
                    "stress": 1.0,
                },
            },
            "val_metrics": {
                "_target_": "nequip.train.EnergyForceStressMetrics",
                "coeffs": {
                    "total_energy_mae": 1.0,
                    "forces_mae": 1.0,
                    "stress_mae": 1.0,
                },
            },
            "test_metrics": "${training_module.val_metrics}",
            "optimizer": {
                "_target_": "torch.optim.Adam",
                "lr": 1.0e-4,
            },
        },
    }

    config_path = os.path.join(output_dir, "finetune_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Config written to: {config_path}")
    print(f"Training for {epochs} epochs ({n_train} train, {n_val} val)...")

    # Run nequip-train (Hydra-based: use --config-dir and --config-name)
    config_dir = os.path.abspath(output_dir)
    cmd = [
        "nequip-train",
        f"--config-dir={config_dir}",
        "--config-name=finetune_config",
    ]
    subprocess.run(cmd, check=True, cwd=output_dir)

    # Find best checkpoint (Hydra outputs to a timestamped subdirectory)
    checkpoint_path = _find_checkpoint(output_dir)

    # Compile for AOTInductor (faster inference, requires GPU)
    # Requires CUDA_HOME to be set for C++ compilation
    compiled_path = os.path.join(output_dir, "NequIP_finetuned.nequip.pt2")
    if checkpoint_path and os.path.exists(checkpoint_path):
        print("Compiling fine-tuned model for AOTInductor...")
        compile_env = os.environ.copy()
        if "CUDA_HOME" not in compile_env:
            # Auto-detect CUDA root from nvcc or known paths
            cuda_home = _find_cuda_home()
            if cuda_home:
                compile_env["CUDA_HOME"] = cuda_home
        compile_cmd = [
            "nequip-compile", checkpoint_path, compiled_path,
            "--mode", "aotinductor", "--device", device, "--target", "ase",
        ]
        try:
            subprocess.run(compile_cmd, check=True, env=compile_env)
            print(f"Compiled: {compiled_path}")
            checkpoint_path = compiled_path
        except subprocess.CalledProcessError as e:
            print(f"Warning: compilation failed ({e}). Using eager-mode checkpoint.")

    # Parse validation errors from NequIP metrics
    val_errors = _parse_metrics(output_dir)

    return {
        "checkpoint_path": checkpoint_path,
        "val_errors": val_errors,
    }


def _find_checkpoint(output_dir):
    """Find best checkpoint in Hydra output directories."""
    import glob

    # Look for best.ckpt or last.ckpt in output subdirectories
    patterns = [
        os.path.join(output_dir, "**", "best.ckpt"),
        os.path.join(output_dir, "**", "last.ckpt"),
        os.path.join(output_dir, "**", "*.ckpt"),
        os.path.join(output_dir, "**", "best_model.pth"),
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern, recursive=True))
        if matches:
            return matches[-1]
    return None


def _parse_metrics(output_dir):
    """Parse NequIP training metrics for validation errors."""
    import glob

    val_errors = {}

    # Look for CSV metric logs in Hydra output directories
    patterns = [
        os.path.join(output_dir, "**", "metrics_epoch.csv"),
        os.path.join(output_dir, "**", "metrics.csv"),
    ]
    metrics_path = None
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            metrics_path = matches[-1]
            break

    if not metrics_path:
        return val_errors

    try:
        import csv
        with open(metrics_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if rows:
            last = rows[-1]
            for key, value in last.items():
                if "val" in key.lower() and "mae" in key.lower():
                    try:
                        name = key.split("/")[-1] if "/" in key else key
                        val_errors[name] = float(value)
                    except ValueError:
                        pass
    except Exception:
        pass

    return val_errors
