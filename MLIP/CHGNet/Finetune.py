"""CHGNet fine-tuning module."""

import os
import numpy as np
from ase.io import read


def finetune(train_file, val_file, output_dir, epochs=50, device="cuda"):
    """Fine-tune CHGNet on extxyz data.

    Returns:
        dict with checkpoint_path and val_errors
    """
    import torch
    from chgnet.model import CHGNet
    from chgnet.trainer import Trainer
    from chgnet.data.dataset import StructureData, get_train_val_test_loader
    from pymatgen.io.ase import AseAtomsAdaptor

    # CHGNet StructureData expects VASP-style stress in kbar (positive=compressive)
    # ASE provides eV/A^3 (positive=tensile)
    # Conversion: eV/A^3 -> kbar = * 1602.1766, flip sign = * (-1)
    EV_A3_TO_KBAR_VASP = -1602.1766208

    # Load pretrained model
    print("Loading pretrained CHGNet...")
    chgnet = CHGNet.load()

    adaptor = AseAtomsAdaptor()

    def load_extxyz(filepath):
        """Convert extxyz to CHGNet StructureData."""
        frames = read(filepath, index=":")
        structures, energies_per_atom, forces_list, stresses_list = [], [], [], []
        for atoms in frames:
            structures.append(adaptor.get_structure(atoms))
            energies_per_atom.append(atoms.get_potential_energy() / len(atoms))
            forces_list.append(atoms.get_forces().tolist())
            try:
                stress_3x3 = atoms.get_stress(voigt=False)  # 3x3, eV/A^3
                stresses_list.append((stress_3x3 * EV_A3_TO_KBAR_VASP).tolist())
            except Exception:
                pass

        return StructureData(
            structures=structures,
            energies=energies_per_atom,
            forces=forces_list,
            stresses=stresses_list if stresses_list else None,
        )

    print("Loading training data...")
    train_dataset = load_extxyz(train_file)
    print("Loading validation data...")
    val_dataset = load_extxyz(val_file)

    # Create data loaders: use full dataset for each (split is already done)
    train_loader = get_train_val_test_loader(
        train_dataset, batch_size=16, train_ratio=1.0, val_ratio=0.0, return_test=False
    )[0]
    val_loader = get_train_val_test_loader(
        val_dataset, batch_size=16, train_ratio=1.0, val_ratio=0.0, return_test=False
    )[0]

    # Train
    print(f"Training for {epochs} epochs...")
    trainer = Trainer(
        model=chgnet,
        targets="efs",
        optimizer="Adam",
        scheduler="CosLR",
        criterion="MSE",
        epochs=epochs,
        learning_rate=1e-3,
        use_device=device,
    )

    save_dir = os.path.join(output_dir, "checkpoints")
    trainer.train(train_loader, val_loader, save_dir=save_dir)

    # Find best checkpoint and copy to stable name
    best_checkpoint = None
    if os.path.isdir(save_dir):
        for f in sorted(os.listdir(save_dir)):
            if f.startswith("bestE_"):
                best_checkpoint = os.path.join(save_dir, f)
                break
    if best_checkpoint:
        import shutil
        stable_path = os.path.join(save_dir, "bestE.pth.tar")
        shutil.copy2(best_checkpoint, stable_path)
        print(f"Copied best checkpoint: {best_checkpoint} -> {stable_path}")
        best_checkpoint = stable_path

    # Evaluate best model on validation set
    val_errors = {}
    if best_checkpoint:
        best_model = CHGNet.from_file(best_checkpoint)
        best_model = best_model.to(device)

        val_frames = read(val_file, index=":")
        e_errors, f_errors = [], []

        # CHGNet computes forces via torch.autograd.grad, which requires
        # gradient tracking — use torch.enable_grad() to ensure it works
        with torch.enable_grad():
            for atoms in val_frames:
                structure = adaptor.get_structure(atoms)
                pred = best_model.predict_structure(structure)
                e_pred = pred["e"]  # eV/atom
                e_true = atoms.get_potential_energy() / len(atoms)
                e_errors.append(abs(e_pred - e_true))
                f_pred = pred["f"]  # eV/A
                f_true = atoms.get_forces()
                f_errors.append(np.mean(np.abs(f_pred - f_true)))

        val_errors["energy_mae_eV_atom"] = float(np.mean(e_errors))
        val_errors["force_mae_eV_A"] = float(np.mean(f_errors))

    return {
        "checkpoint_path": best_checkpoint or save_dir,
        "val_errors": val_errors,
    }
