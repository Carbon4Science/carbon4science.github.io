# Protein Folding Benchmark Infrastructure

The real model execution lives in `/home/chen/projects/Protein-Folding-Benchmark`. This folder mirrors the Carbon4Science task-branch template and contains lightweight exported artifacts for analysis. Current default-mode protein-folding runs keep canonical model IDs and write MSA/carbon provenance into metadata rather than encoding mode in the model name.

Rebuild the compact merged CSV:

```bash
python protein_folding/scripts/summarize_results.py
```

Source controller command shape:

```bash
conda run -n folding-benchmark python scripts/run_benchmark_from_targets.py \
  --targets data/targets/targets_first5.csv \
  --config tmp/backend_smoke/models_six_default_modes.yaml \
  --models esmfold,omegafold,boltz2,chai1,colabfold,openfold \
  --top-k 1 \
  --track-carbon
```

Omitting `--carbon-country-iso-code` now records world-average carbon metadata. Use `--carbon-country-iso-code CHE` only for explicit country-specific reruns.
