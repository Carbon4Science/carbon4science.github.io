"""ORB v2 MPtrj inference module for MLIP benchmarking."""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # No C compiler available for torch.compile

from typing import List, Dict

_calculator = None


def _get_calculator(device=None):
    global _calculator
    if _calculator is not None:
        return _calculator

    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    from orb_models.forcefield import pretrained
    from orb_models.forcefield.calculator import ORBCalculator

    orbff = pretrained.orb_mptraj_only_v2(device=device)
    _calculator = ORBCalculator(orbff, device=device)
    return _calculator


def run_production(config_path, structure_index=0, track_carbon=False):
    """Run production MD (equilibration + production) and return accuracy metrics."""
    calc = _get_calculator()
    from MLIP.production.run_production_md import load_config, run_md_simulation, run_analysis

    config = load_config(config_path)
    struct_cfg = config["structures"][structure_index]
    md_info = run_md_simulation(struct_cfg, calculator=calc, model_name="ORB", track_carbon=track_carbon)
    timing = {"equil_seconds": md_info["equil_seconds"], "prod_seconds": md_info["prod_seconds"]}
    analysis = run_analysis("ORB", struct_cfg, timing=timing)
    return {**md_info, "accuracy": analysis.get("accuracy", {})}


def run(input_data, top_k: int = 10) -> List[Dict]:
    """
    Run NVT MD simulation using ORB calculator.

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
