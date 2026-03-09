# NequIP-MP-L

## Environment
- **Conda env:** `nequip`
- **Package:** `pip install nequip` (requires torch >=2.6)
- **Python:** >=3.10
- **Checkpoint:** Auto-downloads from `nequip.net:mir-group/NequIP-MP-L:0.1`

## ASE Calculator

Compiled (recommended, requires `setup_model.sh` first):
```python
from nequip.ase import NequIPCalculator
calc = NequIPCalculator.from_compiled_model(
    compile_path="NequIP-MP-L.nequip.pt2",
    device="cuda",
    chemical_species_to_atom_type_map=True,
)
```

Alternative (skip compilation):
```python
calc = NequIPCalculator._from_saved_model(
    model_path="nequip.net:mir-group/NequIP-MP-L:0.1",
    device="cuda",
    chemical_species_to_atom_type_map=True,
)
```

## Notes
- Key deps: e3nn >=0.5.9 <0.6.0, matscipy, lightning
- Models cached at `~/.nequip/model_cache`
- `setup_model.sh` handles the one-time `nequip-compile` step
