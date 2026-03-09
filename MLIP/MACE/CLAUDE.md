# MACE-MP-0

## Environment
- **Conda env:** `mace`
- **Package:** `pip install mace-torch` (NOT `mace` — different package!)
- **Python:** >=3.10
- **Checkpoint:** Auto-downloads to `~/.cache/mace/`

## ASE Calculator
```python
from mace.calculators import mace_mp
calc = mace_mp(model="medium", device="cuda", default_dtype="float64")
```

## Notes
- CRITICAL: pins `e3nn==0.4.4` — will conflict with other models' e3nn versions
- Model sizes: `"small"`, `"medium"`, `"large"`
- `default_dtype="float64"` for accuracy; `"float32"` for speed
