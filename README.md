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
| ColabFold | canonical default MSA mode for new default-mode runs | first-five table below is an earlier single-sequence smoke; MSA provenance is now recorded in metadata |
| OpenFold | canonical default with ColabFold/MMseqs2-generated A3M for new default-mode runs | first-five table below is an earlier single-sequence smoke; MSA provenance is now recorded in metadata |

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
- Historical first-five export: `CHE` offline accounting, preserved as originally generated.
- Current default benchmark policy: world-average accounting. New `--track-carbon` runs omit `--carbon-country-iso-code` and record `carbon_country_iso_code=WORLD`, `carbon_intensity_mode=world_average`, and `carbon_intensity_source=configurable_default_world_average`. Use `--carbon-country-iso-code CHE` for explicit Switzerland-specific accounting.
- Raw CodeCarbon CSVs are retained in the benchmark repo under each result directory's `carbon/` folder and summarized in exported metadata CSVs.

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

## Supplemental: Canonical Default-Mode Metadata Smoke

A 7ROA one-target smoke on 2026-05-20 validated canonical model names, MSA provenance fields, and world-average carbon metadata for `esmfold`, `omegafold`, `boltz2`, `chai1`, `colabfold`, and `openfold`. `colabfold` and `openfold` used fresh local ColabFold/MMseqs2 MSAs; `esmfold` and `omegafold` are native no-MSA models; `boltz2` uses an explicit empty MSA/single-sequence mode; Chai-1 MSA use remains marked unknown because the runner passes FASTA only. Exported supplemental files:

- `results/protein_folding_default_modes_7ROA_world_carbon_metadata.csv`
- `results/protein_folding_default_modes_7ROA_world_score_summary.csv`

The first-five table below remains a historical smoke artifact and should not be interpreted as the updated default-mode leaderboard until a full first-five default-mode rerun is exported.

## Supplemental: ColabFold Single-Sequence vs MSA Mode

A follow-up benchmark on 2026-05-20 compared explicit ColabFold variants:

| Model | Mean lDDT-Ca | Mean TM-score | Mean C-alpha RMSD | Runtime (s) | Energy (kWh) | CO2 (g) | Success |
|---|---:|---:|---:|---:|---:|---:|---:|
| `colabfold_msa` | 0.840 | 0.819 | 4.75 | 3067.4 | 0.195630 | 9.214 | 5/5 |
| `colabfold_single` | 0.352 | 0.291 | 19.87 | 490.1 | 0.036727 | 1.730 | 5/5 |

`colabfold_single` uses `--msa-mode single_sequence`. `colabfold_msa` reruns local ColabFold/MMseqs2 MSA search during every benchmarked inference using `/data/chen/protein_folding_databases/colabfold`, so the reported runtime, energy, and CO2 include both MSA search and structure prediction. The supplemental exported tables are:

- `results/colabfold_single_vs_msa_first5_carbon.csv`
- `results/colabfold_single_vs_msa_first5_model_summary.csv`

## Supplemental: OpenFold Single-Sequence vs ColabFold-Generated MSA

A follow-up benchmark on 2026-05-20 compared explicit OpenFold variants:

| Model | Mean lDDT-Ca | Mean TM-score | Mean C-alpha RMSD | Runtime (s) | Energy (kWh) | CO2 (g) | Success |
|---|---:|---:|---:|---:|---:|---:|---:|
| `openfold_msa` | 0.819 | 0.812 | 4.15 | 2726.4 | 0.175815 | 8.281 | 5/5 |
| `openfold_single` | 0.385 | 0.340 | 19.85 | 265.4 | 0.025607 | 1.206 | 5/5 |

`openfold_msa` reruns local ColabFold/MMseqs2 MSA search during every benchmarked inference using `/data/chen/protein_folding_databases/colabfold`, then feeds the generated A3M to OpenFold through its precomputed alignment path. The reported runtime, energy, and CO2 include both MSA search and OpenFold inference. The supplemental exported tables are:

- `results/openfold_single_vs_msa_first5_carbon.csv`
- `results/openfold_single_vs_msa_first5_model_summary.csv`

## Limitations

- This is a first-five smoke benchmark, not a full CASP15/CASP16 benchmark.
- The first-five main table is a historical single-sequence/dummy-MSA smoke for ColabFold/OpenFold; the current default policy uses canonical model names with MSA provenance metadata and world-average carbon accounting.
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

Current default-mode smoke command shape uses world-average carbon by omitting the country override and uses `tmp/backend_smoke/models_six_default_modes.yaml`:

```bash
conda run -n folding-benchmark python scripts/run_benchmark_from_targets.py \
  --targets tmp/backend_smoke/targets_7ROA_chainA.csv \
  --config tmp/backend_smoke/models_six_default_modes.yaml \
  --models esmfold,omegafold,boltz2,chai1,colabfold,openfold \
  --top-k 1 \
  --track-carbon
```

Regenerate the compact merged CSV from exported files:

```bash
python scripts/summarize_results.py
```
