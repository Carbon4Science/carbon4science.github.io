# CHGNet

## Environment
- **Conda env:** `chgnet`
- **Package:** `pip install chgnet` (requires torch >=2.4.1)
- **Python:** >=3.10
- **Params:** ~412K
- **Checkpoint:** Auto-downloads

## ASE Calculator
```python
from chgnet.model.dynamics import CHGNetCalculator
calc = CHGNetCalculator(use_device="cuda")
```

## Notes
- `CHGNetCalculator(use_device=...)` not `device=...`
- Model versions: `"0.3.0"` (default), `"0.2.0"`, `"r2scan"`
- Properties: energy, forces, stress, magnetic moments
