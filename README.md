# Protein Folding

**Task description:** Protein structure prediction from amino-acid sequence.

## Goal

Compare structure-prediction accuracy against runtime, energy use, and CO2 emissions for six validated folding backends from `Protein-Folding-Benchmark`.

## Dataset

- **Dataset name:** First five CASP-derived targets from Protein-Folding-Benchmark.
- **N:** 5 targets.
- **Location:** `data/protein_folding_first5_targets.csv`.
- **References:** Reference PDB paths are recorded in the exported target CSV.

## Models

| Model | Mode | Notes |
|---|---|---|
| ESMFold | single sequence | sequence-only baseline |
| OmegaFold | single sequence | sequence-only baseline |
| Boltz-2 | single-sequence MSA input | `boltz2` backend, top_k=1 for this contribution |
| Chai-1 | single FASTA input | `chai1` backend, top_k=1 for this contribution |
| ColabFold | `--msa-mode single_sequence` | no MMseqs2 search or local ColabFold DB |
| OpenFold | single-sequence/dummy MSA smoke path | no full AF2/OpenFold MSA database tree |

## Metrics

| Metric | Description |
|---|---|
| `lddt_ca` | lDDT over C-alpha atoms |
| `tmalign_tm_score_ref` | TM-score normalized by reference length from US-align/TM-align |
| `ca_rmsd` | C-alpha RMSD after sequential alignment |
| `inference_time_sec` | controller wall-clock runtime per target/model |
| `carbon_emissions_g` | CodeCarbon offline emissions in grams CO2e |
| `carbon_energy_consumed_kwh` | CodeCarbon energy consumption in kWh |

## Carbon Method

- Tracker: CodeCarbon `2.2.2` offline tracker.
- Country ISO code: `CHE`.
- Raw CodeCarbon CSVs are retained in the benchmark repo under `results/six_backend_first5_carbon_smoke/carbon/` and summarized in the exported metadata CSV.

## Hardware

- GPU: 3 x NVIDIA RTX A5000, 24 GB each.
- Driver: 580.126.09.
- CPU/RAM as reported by CodeCarbon: Intel(R) Xeon(R) Gold 6240R CPU @ 2.40GHz, 48 CPU threads visible, about 251 GB RAM visible.

## Results

Full first-five smoke benchmark. All models ran on the same machine with `top_k=1`.

### Accuracy

| Model | Mean lDDT-Ca | Mean TM-score | Mean C-alpha RMSD | Success |
|---|---:|---:|---:|---:|
| `esmfold` | 0.748 | 0.750 | 5.89 | 5/5 |
| `chai1` | 0.722 | 0.707 | 7.71 | 5/5 |
| `omegafold` | 0.671 | 0.700 | 7.38 | 5/5 |
| `boltz2` | 0.571 | 0.589 | 11.79 | 5/5 |
| `openfold` | 0.384 | 0.347 | 19.81 | 5/5 |
| `colabfold` | 0.359 | 0.304 | 20.38 | 5/5 |

### Carbon Efficiency

| Model | Duration (s) | Energy (Wh) | CO2 (g) |
|---|---:|---:|---:|
| `esmfold` | 262.9 | 18.331 | 0.863 |
| `boltz2` | 264.6 | 20.386 | 0.960 |
| `openfold` | 269.4 | 25.807 | 1.216 |
| `omegafold` | 434.3 | 41.380 | 1.949 |
| `colabfold` | 492.6 | 36.553 | 1.722 |
| `chai1` | 516.2 | 44.377 | 2.090 |

## Limitations

- This is a first-five smoke benchmark, not a full CASP15/CASP16 benchmark.
- ColabFold and OpenFold are single-sequence or dummy-MSA runs here; the ColabFold MMseqs database was not required and was not used.
- Results should be interpreted as an initial Carbon4Science contribution artifact rather than a final scientific leaderboard.

## Reproduction

The source benchmark repo is `/home/chen/projects/Protein-Folding-Benchmark`. The main command was:

```bash
conda run -n folding-benchmark python scripts/run_benchmark_from_targets.py \
  --targets data/targets/targets_first5.csv \
  --config tmp/backend_smoke/models_six_single_sequence.yaml \
  --models esmfold,omegafold,boltz2,chai1,colabfold,openfold \
  --top-k 1 \
  --predictions-dir results/six_backend_first5_carbon_smoke/predictions \
  --sequences-dir results/six_backend_first5_carbon_smoke/sequences \
  --logs-dir results/six_backend_first5_carbon_smoke/logs \
  --results-dir results/six_backend_first5_carbon_smoke \
  --run-metadata results/six_backend_first5_carbon_smoke/run_metadata.csv \
  --run-status results/six_backend_first5_carbon_smoke/run_status.csv \
  --max-trials 1 \
  --gpu-cleanup-sleep-sec 10 \
  --track-carbon \
  --carbon-country-iso-code CHE
```

Regenerate the compact merged CSV from exported files:

```bash
python protein_folding/scripts/summarize_results.py
```
