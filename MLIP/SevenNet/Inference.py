"""SevenNet-l3i5 inference module for MLIP benchmarking."""

from typing import List, Dict

_calculator = None


def _get_calculator(device=None):
    global _calculator
    if _calculator is not None:
        return _calculator

    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    from sevenn.calculator import SevenNetCalculator

    _calculator = SevenNetCalculator(model="7net-l3i5", device=device)
    return _calculator


def run_production(config_path, structure_index=0, track_carbon=False):
    """Run production MD (equilibration + production) and return accuracy metrics."""
    calc = _get_calculator()
    from MLIP.production.run_production_md import load_config, run_md_simulation, run_analysis

    config = load_config(config_path)
    struct_cfg = config["structures"][structure_index]
    md_info = run_md_simulation(struct_cfg, calculator=calc, model_name="SevenNet", track_carbon=track_carbon)
    timing = {"equil_seconds": md_info["equil_seconds"], "prod_seconds": md_info["prod_seconds"]}
    analysis = run_analysis("SevenNet", struct_cfg, timing=timing)
    return {**md_info, "accuracy": analysis.get("accuracy", {})}


def run(input_data, top_k: int = 10) -> List[Dict]:
    """
    Run NVT MD simulation using SevenNet calculator.

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
