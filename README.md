# Protein Folding

**Task description:** Protein structure prediction from amino-acid sequence.

## Goal

Compare structure-prediction accuracy against runtime, energy use, and CO2 emissions for locally runnable folding backends from `Protein-Folding-Benchmark`. This export now reports the current default-mode first-five benchmark, explicit MSA provenance, and experimental shared-MSA runs for Protenix and OpenFold3.

## Dataset

- **Dataset name:** First five CASP-derived targets from `Protein-Folding-Benchmark`.
- **N:** 5 targets.
- **Location:** `data/protein_folding_first5_targets.csv`.
- **References:** Reference PDB paths are recorded in the exported target CSV and per-target score CSVs.

## Benchmark Protocol

The source benchmark repo is `/home/chen/projects/Protein-Folding-Benchmark`. Each backend uses the standard runner interface:

```bash
bash runners/run_MODEL.sh input.fasta output_dir top_k
```

Each runner writes standardized predictions as `rank_001.pdb`, `rank_002.pdb`, and so on, plus `metadata.json`. For this Carbon4Science contribution the reported first-five runs use `top_k=1` for compact, comparable carbon accounting.

Scoring reports:

| Metric | Description |
|---|---|
| `lddt_ca` | lDDT over C-alpha atoms |
| `tmalign_tm_score_ref` | TM-score normalized by reference length from USalign/TM-align |
| `ca_rmsd` | C-alpha RMSD after sequential alignment |
| `inference_time_sec` | controller wall-clock runtime per target/model |
| `carbon_emissions_g` | CodeCarbon offline emissions in grams CO2e |
| `carbon_energy_consumed_kwh` | CodeCarbon energy consumption in kWh |

## Models

| Model | Mode / MSA in current export | Notes |
|---|---|---|
| ESMFold | no MSA; native single-sequence | sequence-language-model baseline |
| OmegaFold | no MSA; native single-sequence | sequence-only baseline |
| Boltz-2 | no MSA in local runner | `boltz2` backend with explicit no-MSA/default local mode |
| Chai-1 | no MSA; native embeddings | default Chai-1 CLI uses embeddings without external MSAs/templates; metadata is no longer `unknown` |
| ColabFold | fresh local ColabFold/MMseqs2 MSA | canonical default MSA mode |
| OpenFold | fresh local ColabFold/MMseqs2 A3M | canonical default mode passes generated A3M to OpenFold |
| Protenix | shared precomputed ColabFold/MMseqs2 MSA | experimental shared-MSA backend |
| OpenFold3 | shared precomputed ColabFold/MMseqs2 MSA | experimental low-memory shared-MSA backend on RTX A5000 |

## Carbon Method

- Tracker: CodeCarbon offline tracker.
- Current default benchmark policy: world-average accounting. New `--track-carbon` runs omit `--carbon-country-iso-code` and record `carbon_country_iso_code=WORLD`, `carbon_intensity_mode=world_average`, and `carbon_intensity_source=configurable_default_world_average`.
- Default world-average intensity in the benchmark repo: `475 g CO2e/kWh`.
- Historical first-five single-sequence export: `CHE` offline accounting, preserved in older CSVs for provenance.
- Raw CodeCarbon CSVs are retained in the benchmark repo under each result directory's `carbon/` folder and summarized in exported metadata CSVs here.

## Hardware

- CPU: Intel Xeon Gold 6240R, 1 socket, 24 cores / 48 threads.
- RAM: 251 GiB visible.
- GPU: 3 x NVIDIA RTX A5000, 24,564 MiB each.
- Driver: 580.159.03.
- CUDA reported by driver: 13.0.
- OS/kernel: Ubuntu Linux, kernel `6.8.0-117-generic`, x86_64.

## Shared MSA Accounting

The shared-MSA workflow separates homology search from model inference:

1. ColabFold/MMseqs2 is run once per target against `/data/chen/protein_folding_databases/colabfold`.
2. The resulting per-target A3M files and MSA search carbon metadata are stored in `shared_msa_colabfold_first5_msa_metadata.csv`.
3. Compatible models reuse those A3M files. Protenix converts the shared A3M into paired/unpaired MSA inputs; OpenFold3 copies it as `cfdb_hits.a3m`.
4. Model inference rows mark `msa_generation_included_in_timing=false`, `msa_generation_included_in_carbon=false`, and `msa_reused=true`.
5. `protenix_openfold3_shared_msa_first5_score_cost_summary.csv` joins MSA cost, model inference cost, and structure scores for total-cost reporting.

In the shared-MSA table below, model-only CO2e is reported separately from total CO2e with shared MSA cost added back for end-to-end comparability.

## Results

### Default/native first-five benchmark

| Model | Mode / MSA | n_success | Mean lDDT-Ca | Mean TM-score | Mean Ca RMSD (A) | Mean inference time (s) | Mean model CO2e (g) | MSA cost included? | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| esmfold | no; native_single_sequence | 5 | 0.748 | 0.750 | 5.888 | 57.8 | 1.04 | no | single-sequence language model |
| omegafold | no; native_single_sequence | 5 | 0.671 | 0.700 | 7.378 | 86.6 | 1.56 | no | single-sequence model |
| boltz2 | no; model_default_no_msa | 5 | 0.564 | 0.591 | 11.693 | 54.3 | 0.98 | no | local runner uses explicit no-MSA mode |
| chai1 | no; native_embedding_no_msa | 5 | 0.721 | 0.718 | 6.552 | 104.5 | 1.89 | no | default Chai-1 uses embeddings without MSAs/templates |
| colabfold | yes; default_msa | 5 | 0.842 | 0.819 | 4.718 | 618.1 | 11.15 | yes | fresh ColabFold/MMseqs search per target |
| openfold | yes; default_msa | 5 | 0.852 | 0.848 | 3.693 | 545.1 | 9.83 | yes | fresh ColabFold/MMseqs A3M passed to OpenFold |

### Shared-MSA experimental first-five benchmark

| Model | Mode / MSA | n_success | Mean lDDT-Ca | Mean TM-score | Mean Ca RMSD (A) | Mean model time (s) | Mean model CO2e (g) | Mean total time with shared MSA (s) | Mean total CO2e with shared MSA (g) | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| protenix | yes; shared_precomputed_msa | 5 | 0.867 | 0.855 | 4.708 | 82.1 | 2.45 | 545.3 | 15.75 | shared A3M converted to Protenix paired/unpaired inputs |
| openfold3 | yes; shared_precomputed_msa | 5 | 0.851 | 0.835 | 5.200 | 103.4 | 3.42 | 566.6 | 16.71 | shared A3M copied as cfdb_hits.a3m; low-memory experimental run |

### ColabFold single-sequence vs MSA ablation

| Model | Mode / MSA | n_success | Mean lDDT-Ca | Mean TM-score | Mean Ca RMSD (A) | Mean inference time (s) | Mean model CO2e (g) |
|---|---|---:|---:|---:|---:|---:|---:|
| colabfold_single | no; forced_single_sequence_ablation | 5 | 0.352 | 0.291 | 19.875 | 98.0 | 0.35 |
| colabfold_msa | yes; msa_ablation | 5 | 0.840 | 0.819 | 4.753 | 613.5 | 1.84 |

### OpenFold single-sequence vs MSA ablation

| Model | Mode / MSA | n_success | Mean lDDT-Ca | Mean TM-score | Mean Ca RMSD (A) | Mean inference time (s) | Mean model CO2e (g) |
|---|---|---:|---:|---:|---:|---:|---:|
| openfold_single | no; forced_single_sequence_ablation | 5 | 0.385 | 0.340 | 19.854 | 53.1 | 0.24 |
| openfold_msa | yes; msa_ablation | 5 | 0.819 | 0.812 | 4.148 | 545.3 | 1.66 |

## Exported Files

Current default-mode first-five exports:

- `results/default_modes_first5_model_summary.csv`
- `results/default_modes_first5_run_metadata.csv`
- `results/default_modes_first5_run_status.csv`
- `results/7ROA_chainA_scores.csv`
- `results/7QIH_chainA_scores.csv`
- `results/8ORK_chainA_scores.csv`
- `results/7UYX_chainA_scores.csv`
- `results/7UTD_chainA_scores.csv`

Shared-MSA experimental exports:

- `results/shared_msa_colabfold_first5_msa_metadata.csv`
- `results/protenix_openfold3_shared_msa_first5_model_summary.csv`
- `results/protenix_openfold3_shared_msa_first5_run_metadata.csv`
- `results/protenix_openfold3_shared_msa_first5_score_cost_summary.csv`
- `results/protenix_shared_msa_first5.json`
- `results/openfold3_shared_msa_first5.json`

README-ready generated table:

- `results/protein_folding_readme_benchmark_performance.md`

Historical and ablation exports retained for provenance:

- `results/protein_folding_six_models_first5_*`
- `results/colabfold_single_vs_msa_first5_carbon.csv`
- `results/colabfold_single_vs_msa_first5_model_summary.csv`
- `results/openfold_single_vs_msa_first5_carbon.csv`
- `results/openfold_single_vs_msa_first5_model_summary.csv`

## Reproduction

Current default-mode first-five benchmark command shape from the source repo:

```bash
conda run -n folding-benchmark python scripts/run_benchmark_from_targets.py \
  --targets data/targets/targets_first5.csv \
  --config configs/models.yaml \
  --models esmfold,omegafold,boltz2,chai1,colabfold,openfold \
  --top-k 1 \
  --predictions-dir results/default_modes_first5_carbon_metadata/predictions \
  --sequences-dir results/default_modes_first5_carbon_metadata/sequences \
  --logs-dir results/default_modes_first5_carbon_metadata/logs \
  --results-dir results/default_modes_first5_carbon_metadata \
  --run-metadata results/default_modes_first5_carbon_metadata/run_metadata.csv \
  --run-status results/default_modes_first5_carbon_metadata/run_status.csv \
  --max-trials 1 \
  --track-carbon
```

Shared-MSA Protenix/OpenFold3 command shape:

```bash
conda run -n folding-benchmark python scripts/run_benchmark_from_targets.py \
  --targets data/targets/targets_first5.csv \
  --config tmp/backend_smoke/models_protenix_openfold3_shared_msa.yaml \
  --models protenix,openfold3 \
  --top-k 1 \
  --shared-msa-metadata results/shared_msa_colabfold_first5/msa_metadata.csv \
  --shared-msa-root results/shared_msa_colabfold_first5/msas \
  --track-carbon
```

Regenerate the compact merged CSV from exported historical files:

```bash
python scripts/summarize_results.py
```

## Limitations

- This is a five-target smoke benchmark, not a full CASP benchmark.
- Protenix and OpenFold3 are experimental shared-MSA exports and are not canonical enabled backends in `Protein-Folding-Benchmark/configs/models.yaml`.
- OpenFold3 uses a low-memory configuration validated on the local 24 GB RTX A5000 setup; broader validation is still needed.
- Official AlphaFold2 is not reported as a benchmarked method here because the official parameters and AlphaFold database layout are not installed in the source repo. ColabFold is reported as ColabFold, not relabeled as AF2.
