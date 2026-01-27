# Skill: Run Evaluation

Run task-specific evaluation on predictions.

## Usage
```
/evaluate [task]
```

## Examples
```
/evaluate Retrosynthesis
/evaluate MolGen
```

## Instructions

When the user invokes this skill:

### For Retrosynthesis

1. **Load the evaluation module:**
   ```python
   from Retrosynthesis.evaluate import load_test_data, evaluate, METRICS
   ```

2. **Available metrics:**
   - `top_1`: Exact match at rank 1
   - `top_5`: Correct in top 5 predictions
   - `top_10`: Correct in top 10 predictions
   - `top_50`: Correct in top 50 predictions

3. **Run evaluation:**
   ```python
   # Load test data
   test_cases = load_test_data(limit=100)

   # Run model inference
   predictions = [model.run(tc['product'], top_k=50) for tc in test_cases]

   # Evaluate
   results = evaluate(predictions, test_cases, metrics=['top_10', 'top_50'])
   print(f"Top-10: {results['top_10']*100:.2f}%")
   print(f"Top-50: {results['top_50']*100:.2f}%")
   ```

### For MolGen (Template)

1. **Available metrics:**
   - `validity`: Fraction of valid SMILES
   - `uniqueness`: Fraction of unique molecules
   - `novelty`: Fraction not in training set
   - `diversity`: Internal Tanimoto diversity

2. **Evaluation:**
   ```python
   from MolGen.evaluate import evaluate, METRICS

   results = evaluate(generated_smiles, reference_smiles=train_smiles)
   ```

### For MatGen (Template)

1. **Available metrics:**
   - `validity`: Fraction of valid structures
   - `uniqueness`: Fraction of unique structures
   - `stability`: Fraction predicted stable
   - `coverage`: Fraction of compositions covered

## Test Data Locations

- Retrosynthesis: `Retrosynthesis/data/USPTO_50K_test.pickle`
- MolGen: `MolGen/data/` (to be added)
- MatGen: `MatGen/data/` (to be added)

## Notes
- Each task defines its own metrics in `<Task>/evaluate.py`
- Use `METRICS` constant to see available metrics for a task
- Test data is loaded via `load_test_data(limit=N)`
