# SevenNet-l3i5

## Environment
- **Conda env:** `sevennet`
- **Package:** `pip install sevenn` (requires torch >=2.0)
- **Python:** >=3.10
- **Checkpoint:** Bundled in pip package (no download needed)

## ASE Calculator
```python
from sevenn.calculator import SevenNetCalculator
calc = SevenNetCalculator(model="7net-l3i5", device="cuda")
```

## Notes
- Requires `torch_geometric` extensions (scatter, sparse, cluster) — install AFTER PyTorch
- `"7net-l3i5"` or `"sevennet-l3i5"` both work
- Architecture: l_max=3, 5 interaction layers
