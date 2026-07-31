"""ASE calculator factory for DPA-4.0.1-Pro-MPtrj.

The DPA4 checkpoint is supplied explicitly because its distribution location
and filename may differ between cluster installations.  Set ``DPA4_MODEL``
or pass ``checkpoint_path`` to select it.
"""

import os


_calculator = None


def _get_calculator(device=None, checkpoint_path=None):
    global _calculator
    if checkpoint_path is None and _calculator is not None:
        return _calculator

    model_path = checkpoint_path or os.environ.get("DPA4_MODEL")
    if not model_path:
        model_path = os.path.join(os.path.dirname(__file__), "dpa-4.0-pro-mptrj-21.88-32.10.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "DPA-4.0.1-Pro-MPtrj checkpoint not found. Set DPA4_MODEL or "
            f"pass --dpa4-checkpoint. Expected: {model_path}"
        )

    from deepmd.calculator import DP

    # DeePMD's DP calculator selects the backend/device from the installed
    # model; keeping this call identical to the supported ASE interface also
    # works for DPA4 checkpoints loaded by newer deepmd-kit releases.
    calc = DP(model=model_path)
    if checkpoint_path is None:
        _calculator = calc
    return calc
