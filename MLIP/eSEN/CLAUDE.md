# eSEN-30M-MP

## Environment
- **Conda env:** `esen`
- **Package:** `pip install fairchem-core==1.10.0 torch_geometric torch_scatter torch_sparse scipy<1.15`
- **Python:** >=3.10
- **Params:** ~30M (30,086,018)
- **Checkpoint:** HuggingFace gated (`facebook/OMAT24`), requires `huggingface-cli login`

## ASE Calculator
```python
from huggingface_hub import hf_hub_download
from fairchem.core import OCPCalculator

checkpoint_path = hf_hub_download(repo_id="facebook/OMAT24", filename="esen_30m_mptrj.pt")
calc = OCPCalculator(checkpoint_path=checkpoint_path, cpu=False)
```

## Notes
- Uses fairchem-core v1 API (`OCPCalculator`); the v2 API (`FAIRChemCalculator`) is NOT compatible with the OMAT24 checkpoint format
- Requires `torch_geometric`, `torch_scatter`, `torch_sparse` (install from PyG wheels matching your torch+CUDA version)
- Requires `scipy<1.15` (v1.15+ removed `scipy.special.sph_harm`)
- Set `HF_TOKEN` env var for reproducible access without interactive login
