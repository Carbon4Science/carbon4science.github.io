# Protein Folding Benchmark Description

This contribution evaluates protein structure prediction models on the first five CASP-derived targets prepared by `Protein-Folding-Benchmark`.

The benchmark records structural quality and carbon/energy metadata for each target/model run. It is intentionally a first-five smoke benchmark, not a full CASP benchmark.

## Inputs

- `data/protein_folding_first5_targets.csv`: target metadata, sequences, reference PDB paths, and notes exported from Protein-Folding-Benchmark.

## Outputs

- `results/protein_folding_six_models_first5_run_metadata.csv`: per-target/model timing and CodeCarbon metadata.
- `results/protein_folding_six_models_first5_run_status.csv`: controller success/failure table.
- `results/protein_folding_six_models_first5_scores.csv`: per-target/model structural metrics.
- `results/protein_folding_six_models_first5_model_summary.csv`: cross-target score summary by model.
- `results/protein_folding_six_models_first5_merged.csv`: compact merged table for downstream Carbon4Science analysis.
