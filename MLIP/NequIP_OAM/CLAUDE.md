# NequIP-OAM-L

## Environment
- **Conda env:** `nequip` (shared with NequIP-MP-L)
- **Package:** `pip install nequip` (requires torch >=2.6)
- **Python:** >=3.10
- **Params:** ~9.6M
- **Checkpoint:** Auto-downloads from `nequip.net:mir-group/NequIP-OAM-L:0.1`

## ASE Calculator

Compiled (recommended, requires `setup_model.sh` first):
```python
from nequip.ase import NequIPCalculator
calc = NequIPCalculator.from_compiled_model(
    compile_path="NequIP-OAM-L.nequip.pt2",
    device="cuda",
    chemical_species_to_atom_type_map=True,
)
```

Alternative (skip compilation):
```python
calc = NequIPCalculator._from_saved_model(
    model_path="nequip.net:mir-group/NequIP-OAM-L:0.1",
    device="cuda",
    chemical_species_to_atom_type_map=True,
)
```

## Notes
- Same architecture as NequIP-MP-L, trained on OMat24 + sAlex + MPtrj
- OAM variant recommended for production over MP-L
- Models cached at `~/.nequip/model_cache`
- Paper: arXiv:2504.16068, 2025
