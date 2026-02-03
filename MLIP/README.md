# MLIP (Machine Learning Interatomic Potentials)

**Task Leader:** Junyoung Choi

Machine learning interatomic potentials: Predict atomic energies, forces, and stresses for molecular dynamics simulations.

## Metrics

| Metric | Description |
|--------|-------------|
| `energy_mae` | Mean absolute error on energy predictions (meV/atom) |
| `force_mae` | Mean absolute error on force predictions (meV/A) |
| `force_cosine` | Cosine similarity of predicted vs true force vectors |
| `stress_mae` | Mean absolute error on stress predictions (GPa) |

## Test Dataset

- Location: `data/` (to be added)

## Models

| Model | Paper | Environment | License |
|-------|-------|-------------|---------|
| *TBD* | - | - | - |

## Current Results

*No results yet.*

## Usage

```python
from MLIP.evaluate import load_test_data, evaluate, METRICS

# Load test data
test_cases = load_test_data(limit=100)

# Run model inference
from MLIP.SomeModel.Inference import run
predictions = [run(tc['input']) for tc in test_cases]

# Evaluate
results = evaluate(predictions, test_cases)
print(f"Energy MAE: {results['energy_mae']:.4f} meV/atom")
print(f"Force MAE: {results['force_mae']:.4f} meV/A")
```

## Adding a New Model

See `/add-model MLIP <ModelName>` skill or `../.claude/skills/add-model.md`.
