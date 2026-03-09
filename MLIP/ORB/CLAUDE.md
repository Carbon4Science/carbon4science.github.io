# ORB v2 MPtrj

## Environment
- **Conda env:** `orb`
- **Package:** `pip install orb-models` (requires torch >=2.6)
- **Python:** >=3.11
- **Checkpoint:** Auto-downloads from S3

## ASE Calculator
```python
import torch
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

device = "cuda" if torch.cuda.is_available() else "cpu"
orbff = pretrained.orb_mptraj_only_v2(device=device)
calc = ORBCalculator(orbff, device=device)
```

## Notes
- `orb_mptraj_only_v2` trained on MPTraj only
- For general use: `pretrained.orb_v2()` (MPTraj + Alexandria)
- Supports `torch.compile` (auto-detected)
