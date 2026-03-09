# eSEN-30M-OAM

## Environment
- **Conda env:** `esen` (shared with eSEN-30M-MP)
- **Package:** `pip install fairchem-core==1.10.0 torch_geometric torch_scatter torch_sparse scipy<1.15`
- **Python:** >=3.10
- **Params:** ~30M (30,086,018)
- **Checkpoint:** HuggingFace gated (`facebook/OMAT24`), requires `huggingface-cli login`

## ASE Calculator
```python
from huggingface_hub import hf_hub_download
from fairchem.core import OCPCalculator

checkpoint_path = hf_hub_download(repo_id="facebook/OMAT24", filename="esen_30m_oam.pt")
calc = OCPCalculator(checkpoint_path=checkpoint_path, cpu=False)
```

## Notes
- Same architecture as eSEN-30M-MP, different training data (OMat24 + sAlex + MPtrj)
- Uses fairchem-core v1 API (`OCPCalculator`); v2 API NOT compatible
- OAM variant is non-compliant on Matbench Discovery (trained on datasets beyond MPtrj)
- Set `HF_TOKEN` env var for reproducible access
