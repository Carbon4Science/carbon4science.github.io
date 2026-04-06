"""NequIP-MP-L inference module for MLIP benchmarking."""

import os
from typing import List, Dict

_calculator = None


def _get_calculator(device=None, checkpoint_path=None):
    global _calculator
    if checkpoint_path is None and _calculator is not None:
        return _calculator

    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    from nequip.ase import NequIPCalculator

    if checkpoint_path:
        if checkpoint_path.endswith(".pt2"):
            calc = NequIPCalculator.from_compiled_model(
                compile_path=checkpoint_path,
                device=device,
                chemical_species_to_atom_type_map=True,
            )
        else:
            calc = NequIPCalculator._from_saved_model(
                model_path=checkpoint_path,
                device=device,
                chemical_species_to_atom_type_map=True,
            )
    else:
        # Try compiled model first, fall back to saved model
        compiled_path = os.path.join(os.path.dirname(__file__), "NequIP-MP-L.nequip.pt2")
        if os.path.exists(compiled_path):
            calc = NequIPCalculator.from_compiled_model(
                compile_path=compiled_path,
                device=device,
                chemical_species_to_atom_type_map=True,
            )
        else:
            calc = NequIPCalculator._from_saved_model(
                model_path="nequip.net:mir-group/NequIP-MP-L:0.1",
                device=device,
                chemical_species_to_atom_type_map=True,
            )

    if checkpoint_path is None:
        _calculator = calc
    return calc


def run_production(config_path, structure_index=0, track_carbon=False, checkpoint_path=None):
    """Run production MD (equilibration + production) and return accuracy metrics."""
    calc = _get_calculator(checkpoint_path=checkpoint_path)
    from MLIP.production.run_production_md import load_config, run_md_simulation, run_analysis

    config = load_config(config_path)
    struct_cfg = config["structures"][structure_index]
    md_info = run_md_simulation(struct_cfg, calculator=calc, model_name="NequIP", track_carbon=track_carbon)
    timing = {"equil_seconds": md_info["equil_seconds"], "prod_seconds": md_info["prod_seconds"]}
    analysis = run_analysis("NequIP", struct_cfg, timing=timing)
    return {**md_info, "accuracy": analysis.get("accuracy", {})}


def run(input_data, top_k: int = 10) -> List[Dict]:
    """
    Run NVT MD simulation using NequIP calculator.

    Args:
        input_data: Path to CIF structure file.
        top_k: Not used for MLIP (kept for interface compatibility).

    Returns:
        [{'input': '...', 'predictions': [{'output': 'md_completed', 'score': 1.0, ...}]}]
    """
    calc = _get_calculator()
    from MLIP.run_md import run_md

    md_results = run_md(input_data, calc)
    return [
        {
            "input": str(input_data),
            "predictions": [{"output": "md_completed", "score": 1.0, **md_results}],
        }
    ]
