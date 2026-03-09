# EquFlash (SevenNet + FlashTP)

## Environment
- **Conda env:** `equflash`
- **Package:** `pip install sevenn` + FlashTP from GitHub
- **Python:** >=3.10
- **Params:** ~28.7M
- **Checkpoint:** Uses `7net-mf-ompa` pretrained model (auto-downloads)

## ASE Calculator
```python
from sevenn.calculator import SevenNetCalculator
calc = SevenNetCalculator(model="7net-mf-ompa", modal="mpa", device="cuda", enable_flash=True)
```

## Notes
- FlashTP: fused, sparsity-aware CUDA tensor product library for acceleration
- Install FlashTP: `pip install git+https://github.com/SNU-ARC/flashTP@v0.1.0 --no-build-isolation`
- FlashTP requires CUDA Toolkit 12+ and a C compiler for kernel compilation
- Set `CUDA_ARCH_LIST` for your GPU (e.g., "86" for RTX 3090, "89" for RTX 5000 Ada)
- Falls back to standard SevenNet (no flash) if FlashTP compilation fails
- Paper: FlashTP, ICML 2025
