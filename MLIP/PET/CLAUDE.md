# PET-OAM-XL

## Environment
- **Conda env:** `pet-oam`
- **Package:** `pip install upet`
- **Python:** >=3.10
- **Params:** ~730M
- **Checkpoint:** Auto-downloads from HuggingFace (`lab-cosmo/upet`)

## ASE Calculator
```python
from upet.calculator import UPETCalculator
calc = UPETCalculator(model="pet-oam-xl", version="1.0.0", device="cuda")
```

## Notes
- Requires PyTorch >= 2.1 (via metatrain dependency)
- Checkpoints auto-download on first use from HuggingFace
- OAM = OMat24 + sAlexandria + MPtrj training data
- Paper: PET-MAD, Nat. Commun. 2025
