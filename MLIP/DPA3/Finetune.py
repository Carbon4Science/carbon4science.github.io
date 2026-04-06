"""DPA-3 fine-tuning module."""

import os
import json
import subprocess
import sys
import numpy as np
from ase.io import read


def _extxyz_to_deepmd_npy(extxyz_file, output_dir):
    """Convert extxyz to deepmd/npy format without dpdata dependency.

    Creates the directory structure expected by deepmd-kit:
        output_dir/type.raw, type_map.raw, set.000/{energy,force,coord,box,virial}.npy
    """
    frames = read(extxyz_file, index=":")
    nframes = len(frames)
    natoms = len(frames[0])

    # Build type map from all frames
    all_symbols = set()
    for atoms in frames:
        all_symbols.update(atoms.get_chemical_symbols())
    type_map = sorted(all_symbols)
    symbol_to_type = {s: i for i, s in enumerate(type_map)}

    # Extract arrays
    energies = np.zeros(nframes)
    forces = np.zeros((nframes, natoms * 3))
    coords = np.zeros((nframes, natoms * 3))
    boxes = np.zeros((nframes, 9))
    virials = np.zeros((nframes, 9))
    atom_types = np.array([symbol_to_type[s] for s in frames[0].get_chemical_symbols()])

    for i, atoms in enumerate(frames):
        energies[i] = atoms.get_potential_energy()
        forces[i] = atoms.get_forces().flatten()
        coords[i] = atoms.get_positions().flatten()
        boxes[i] = atoms.get_cell().array.flatten()
        # virial = stress_tensor * volume (deepmd convention)
        try:
            stress_3x3 = atoms.get_stress(voigt=False)  # eV/A^3
            volume = atoms.get_volume()
            virials[i] = (stress_3x3 * volume).flatten()
        except Exception:
            pass

    # Write files
    os.makedirs(output_dir, exist_ok=True)
    set_dir = os.path.join(output_dir, "set.000")
    os.makedirs(set_dir, exist_ok=True)

    np.save(os.path.join(set_dir, "energy.npy"), energies)
    np.save(os.path.join(set_dir, "force.npy"), forces)
    np.save(os.path.join(set_dir, "coord.npy"), coords)
    np.save(os.path.join(set_dir, "box.npy"), boxes)
    np.save(os.path.join(set_dir, "virial.npy"), virials)

    with open(os.path.join(output_dir, "type.raw"), "w") as f:
        f.write(" ".join(str(t) for t in atom_types) + "\n")
    with open(os.path.join(output_dir, "type_map.raw"), "w") as f:
        for s in type_map:
            f.write(s + "\n")

    return type_map


def _convert_frozen_to_checkpoint(frozen_path, output_path):
    """Convert TorchScript frozen model to deepmd checkpoint format.

    deepmd --finetune expects state_dict with _extra_state.model_params,
    but .pth files are frozen TorchScript archives. This converts them.
    """
    import torch
    import json as _json

    model = torch.jit.load(frozen_path, map_location="cpu")
    state_dict = model.state_dict()

    # Extract model_params from model_def_script attribute
    model_def_script = model.model_def_script
    model_params = _json.loads(model_def_script)

    # Remove keys not recognized by installed deepmd-kit version
    # (model may have been exported with a newer deepmd version)
    _strip_unknown_keys(model_params)

    # Build checkpoint in the format deepmd expects
    # _extra_state must be inside the model state_dict (deepmd extracts
    # state_dict["model"] first, then looks for _extra_state inside it)
    state_dict["_extra_state"] = {"model_params": model_params}
    checkpoint = {"model": state_dict}
    torch.save(checkpoint, output_path)


def _strip_unknown_keys(params):
    """Remove config keys not recognized by installed deepmd-kit v3.1.2.

    The DPA-3.1 model was exported with a newer deepmd version that has
    additional config keys. Strip them to avoid strict validation errors
    and RepFlowArgs construction errors.
    """
    import inspect
    from deepmd.pt.model.descriptor.dpa3 import RepFlowArgs

    # Get valid keys for RepFlowArgs
    valid_repflow = set(inspect.signature(RepFlowArgs.__init__).parameters.keys()) - {"self"}

    # Strip descriptor-level unknown keys
    desc = params.get("descriptor", {})
    desc.pop("use_torch_embed", None)

    # Strip repflow-level unknown keys
    repflow = desc.get("repflow", {})
    unknown = [k for k in repflow if k not in valid_repflow]
    for k in unknown:
        del repflow[k]


def finetune(train_file, val_file, output_dir, epochs=50, device="cuda"):
    """Fine-tune DPA-3.1-MPtrj on extxyz data using dp CLI.

    Converts extxyz -> deepmd/npy, then runs dp --pt train.

    Returns:
        dict with checkpoint_path and val_errors
    """
    # Convert extxyz to deepmd/npy format
    print("Converting training data to deepmd/npy format...")
    train_npy_dir = os.path.join(output_dir, "deepmd_data", "training")
    val_npy_dir = os.path.join(output_dir, "deepmd_data", "validation")

    type_map = _extxyz_to_deepmd_npy(train_file, train_npy_dir)
    _extxyz_to_deepmd_npy(val_file, val_npy_dir)

    n_train = len(read(train_file, index=":"))
    print(f"  Train: {n_train} frames -> {train_npy_dir}")
    print(f"  Val -> {val_npy_dir}")

    # Compute number of training steps
    steps_per_epoch = max(1, n_train // 4)
    numb_steps = epochs * steps_per_epoch

    # Generate input JSON config
    input_config = {
        "model": {
            "type_map": type_map,
            "descriptor": {},
            "fitting_net": {},
        },
        "learning_rate": {
            "type": "exp",
            "start_lr": 1e-4,
            "stop_lr": 1e-8,
            "decay_steps": max(100, numb_steps // 10),
        },
        "loss": {
            "type": "ener",
            "start_pref_e": 0.02,
            "limit_pref_e": 1,
            "start_pref_f": 1000,
            "limit_pref_f": 1,
            "start_pref_v": 0.02,
            "limit_pref_v": 1,
        },
        "training": {
            "training_data": {
                "systems": [os.path.abspath(train_npy_dir)],
                "batch_size": "auto",
            },
            "validation_data": {
                "systems": [os.path.abspath(val_npy_dir)],
                "batch_size": "auto",
                "numb_btch": 1,
            },
            "numb_steps": numb_steps,
            "seed": 42,
            "disp_file": "lcurve.out",
            "disp_freq": max(1, steps_per_epoch),
            "save_freq": max(100, numb_steps // 5),
        },
    }

    config_path = os.path.join(output_dir, "input_finetune.json")
    with open(config_path, "w") as f:
        json.dump(input_config, f, indent=2)

    # Get pretrained model path
    pretrained_path = os.path.join(os.path.dirname(__file__), "dpa-3.1-mptrj.pth")
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(
            f"Pretrained model not found at {pretrained_path}. "
            "Run MLIP/DPA3/download_model.sh first."
        )

    # Convert TorchScript frozen model to deepmd checkpoint format
    # deepmd's --finetune expects a checkpoint with _extra_state.model_params
    # but the .pth file is a frozen TorchScript model
    print(f"Converting frozen model to checkpoint format...")
    converted_path = os.path.join(output_dir, "pretrained_converted.pt")
    _convert_frozen_to_checkpoint(pretrained_path, converted_path)

    # Use wrapper to bypass strict config validation (model was exported
    # with a newer deepmd version that has extra config keys)
    print(f"Training for {numb_steps} steps ({epochs} epochs)...")
    wrapper_script = os.path.join(output_dir, "_dp_train.py")
    with open(wrapper_script, "w") as f:
        f.write(
            "import sys\n"
            "import deepmd.utils.argcheck as _ac\n"
            "from dargs import Argument\n"
            "def _patched_normalize(data, multi_task=False):\n"
            "    base = Argument('base', dict, _ac.gen_args(multi_task=multi_task))\n"
            "    data = base.normalize_value(data, trim_pattern='_*')\n"
            "    base.check_value(data, strict=False)\n"
            "    return data\n"
            "_ac.normalize = _patched_normalize\n"
            "from deepmd.main import main\n"
            "sys.exit(main())\n"
        )
    cmd = [
        sys.executable, wrapper_script,
        "--pt", "train",
        "input_finetune.json",
        "--finetune", converted_path,
        "--use-pretrain-script",
        "--skip-neighbor-stat",
    ]
    subprocess.run(cmd, check=True, cwd=output_dir)

    # Find checkpoint
    checkpoint_path = os.path.join(output_dir, "model.ckpt.pt")
    if not os.path.exists(checkpoint_path):
        for name in ["model.ckpt.pt", "model.pt", "frozen_model.pth"]:
            p = os.path.join(output_dir, name)
            if os.path.exists(p):
                checkpoint_path = p
                break

    # Parse validation errors from lcurve.out
    val_errors = _parse_lcurve(output_dir)

    return {
        "checkpoint_path": checkpoint_path,
        "val_errors": val_errors,
    }


def _parse_lcurve(output_dir):
    """Parse lcurve.out for validation errors."""
    val_errors = {}
    lcurve_path = os.path.join(output_dir, "lcurve.out")
    if not os.path.exists(lcurve_path):
        return val_errors

    try:
        with open(lcurve_path) as f:
            lines = f.readlines()

        if len(lines) >= 2:
            header = lines[0].strip().lstrip("#").split()
            last_line = lines[-1].strip().split()
            for i, col in enumerate(header):
                if i < len(last_line):
                    try:
                        val = float(last_line[i])
                    except ValueError:
                        continue
                    if "rmse_val" in col and "e" in col:
                        val_errors["energy_RMSE"] = val
                    elif "rmse_val" in col and "f" in col:
                        val_errors["force_RMSE"] = val
                    elif "rmse_val" in col and "v" in col:
                        val_errors["virial_RMSE"] = val
    except Exception:
        pass

    return val_errors
