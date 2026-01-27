# MolGen (Molecule Generation)

**Task Leader:** Gunwook Nam

Molecular generation: Generate novel molecules with desired properties.

## Metrics

| Metric | Description |
|--------|-------------|
| `validity` | Fraction of valid SMILES strings |
| `uniqueness` | Fraction of unique molecules among valid ones |
| `novelty` | Fraction of molecules not in training set |
| `diversity` | Internal Tanimoto diversity (1 - avg similarity) |

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
from MolGen.evaluate import evaluate, METRICS

# Generate molecules with your model
generated_smiles = model.generate(num_samples=1000)

# Evaluate
results = evaluate(generated_smiles, reference_smiles=train_smiles)
print(f"Validity: {results['validity']*100:.2f}%")
print(f"Uniqueness: {results['uniqueness']*100:.2f}%")
print(f"Novelty: {results['novelty']*100:.2f}%")
print(f"Diversity: {results['diversity']:.4f}")
```

## Adding a New Model

See `/add-model MolGen <ModelName>` skill or `../.claude/skills/add-model.md`.
