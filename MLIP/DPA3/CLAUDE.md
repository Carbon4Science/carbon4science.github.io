# DPA-3.1-MPtrj

## Environment
- **Conda env:** `deepmd`
- **Package:** `pip install deepmd-kit[torch]`
- **Python:** >=3.9
- **Params:** ~4.81M
- **Checkpoint:** Download from Figshare (see `download_model.sh`)

## ASE Calculator
```python
from deepmd.calculator import DP
calc = DP(model="MLIP/DPA3/dpa-3.1-mptrj.pth")
```

## Setup
```bash
./MLIP/DPA3/download_model.sh
```

## Notes
- `type_dict` is auto-inferred from model's type map
- No `head` parameter needed for DPA-3.1-MPtrj (single-task model)
- For multi-task DPA-3.1-3M, use `head="MP_traj_v024_alldata_mixu"`
