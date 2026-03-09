# Nequix MP

## Environment
- **Conda env:** `nequix`
- **Package:** `pip install nequix` (**JAX-based**, not PyTorch)
- **Python:** >=3.10
- **Params:** ~708K
- **Checkpoint:** Auto-downloads to `~/.cache/nequix/models/`

## ASE Calculator
```python
from nequix.calculator import NequixCalculator
calc = NequixCalculator("nequix-mp-1", backend="jax")
```

## Notes
- JAX-based — device selection via `CUDA_VISIBLE_DEVICES`, not `device="cuda:0"`
- Set `os.environ["CUDA_VISIBLE_DEVICES"]` before importing nequix
- Also supports `backend="torch"` if PyTorch backend preferred
- Sets `jax.config.update("jax_default_matmul_precision", "highest")` for consistency
