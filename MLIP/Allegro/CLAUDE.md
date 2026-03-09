# Allegro-OAM-L

## Environment
- **Conda env:** `nequip` (shared with NequIP; needs `pip install nequip-allegro`)
- **Package:** `pip install nequip-allegro` (extends nequip)
- **Python:** >=3.10
- **Params:** ~9.7M
- **Checkpoint:** Auto-downloads from `nequip.net:mir-group/Allegro-OAM-L:0.1`

## ASE Calculator

Compiled (recommended, requires `setup_model.sh` first):
```python
from nequip.ase import NequIPCalculator
calc = NequIPCalculator.from_compiled_model(
    compile_path="Allegro-OAM-L.nequip.pt2",
    device="cuda",
    chemical_species_to_atom_type_map=True,
)
```

Alternative (skip compilation):
```python
calc = NequIPCalculator._from_saved_model(
    model_path="nequip.net:mir-group/Allegro-OAM-L:0.1",
    device="cuda",
    chemical_species_to_atom_type_map=True,
)
```

## Notes
- Uses same `NequIPCalculator` as NequIP models
- Strictly local architecture (no message passing), highly parallelizable
- Trained on OMat24 + sAlex + MPtrj
- Architecture paper: Nat. Commun. 2023; OAM models: arXiv:2504.16068, 2025
