# Protein Folding Contribution Generation Log

Date/time: 2026-05-19
Source benchmark repo: `/home/chen/projects/Protein-Folding-Benchmark`
Contribution repo used: `/home/chen/projects/carbon4science.github.io`

Note: the instruction file named `/home/chen/projects/caron4science.github.io`, but that path did not exist. The existing repo on disk was `/home/chen/projects/carbon4science.github.io`; this folder was generated there and the path mismatch is documented in the Protein-Folding-Benchmark execution log.

## Source run

- Results root: `results/six_backend_first5_carbon_smoke`
- Targets: first five CASP-derived targets from `data/targets/targets_first5.csv`
- Models: esmfold, omegafold, boltz2, chai1, colabfold, openfold
- Carbon tracker: CodeCarbon offline, country ISO `CHE`
- Status: all 30 target/model runs succeeded

## Exported files

- `data/protein_folding_first5_targets.csv`
- `results/protein_folding_six_models_first5_run_metadata.csv`
- `results/protein_folding_six_models_first5_run_status.csv`
- `results/protein_folding_six_models_first5_model_summary.csv`
- `results/protein_folding_six_models_first5_scores.csv`
- `results/protein_folding_six_models_first5_merged.csv`
- `results/protein_folding_six_models_first5_carbon_by_model.csv`
- Per-target score CSVs and per-model JSON result files

## Validation

Run:

```bash
python protein_folding/scripts/summarize_results.py
python protein_folding/evaluate.py
```
