# MatGen (Material Generation)

**Task Leader:** Junkil Park

Material generation: Generate novel crystal structures and materials.

## Metrics

| Metric | Description |
|--------|-------------|
| `validity` | Fraction of valid crystal structures |
| `uniqueness` | Fraction of unique structures |
| `stability` | Fraction predicted to be thermodynamically stable |
| `coverage` | Fraction of target compositions covered |

## Test Dataset

- Location: `data/` (to be added)

## Models

| Model | Paper | Environment |
|-------|-------|-------------|
| *TBD* | - | - |

## Current Results

*No results yet.*

## Usage

```python
from MatGen.evaluate import evaluate, METRICS

# Generate structures with your model
generated_structures = model.generate(num_samples=1000)

# Evaluate
results = evaluate(generated_structures, reference_structures=train_structures)
print(f"Validity: {results['validity']*100:.2f}%")
print(f"Uniqueness: {results['uniqueness']*100:.2f}%")
print(f"Stability: {results['stability']*100:.2f}%")
print(f"Coverage: {results['coverage']*100:.2f}%")
```

## Adding a New Model

See `/add-model MatGen <ModelName>` skill or `../.claude/skills/add-model.md`.
