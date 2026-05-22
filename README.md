# Protein Folding

**Task description:** Protein structure prediction from amino-acid sequence.

## Goal

Compare structure-prediction accuracy against runtime, energy use, and CO2 emissions for locally runnable folding backends from `Protein-Folding-Benchmark`. The current clean results export includes default/no-MSA first-five rows for ESMFold, OmegaFold, Boltz-2, and Chai-1 plus the latest unified shared-MSA first-five rows for ColabFold, OpenFold, Protenix, and OpenFold3.

## Dataset

- **Dataset name:** First five CASP-derived targets from `Protein-Folding-Benchmark`.
- **N:** 5 targets.
- **Location:** `data/protein_folding_first5_targets.csv`.
- **References:** Reference PDB paths are recorded in the exported target CSV and per-target score CSVs.


## Method References

| Method | Paper / report | DOI | GitHub repository |
|---|---|---|---|
| ESMFold | Lin et al., "Evolutionary-scale prediction of atomic-level protein structure with a language model" | [`10.1126/science.ade2574`](https://doi.org/10.1126/science.ade2574) | [`facebookresearch/esm`](https://github.com/facebookresearch/esm) |
| OmegaFold | Wu et al., "High-resolution de novo structure prediction from primary sequence" | [`10.1101/2022.07.21.500999`](https://doi.org/10.1101/2022.07.21.500999) | [`HeliXonProtein/OmegaFold`](https://github.com/HeliXonProtein/OmegaFold) |
| Boltz-2 | Passaro et al., "Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction" | [`10.1101/2025.06.14.659707`](https://doi.org/10.1101/2025.06.14.659707) | [`jwohlwend/boltz`](https://github.com/jwohlwend/boltz) |
| Chai-1 | Boitreaud et al., "Chai-1: Decoding the molecular interactions of life" | [`10.1101/2024.10.10.615955`](https://doi.org/10.1101/2024.10.10.615955) | [`chaidiscovery/chai-lab`](https://github.com/chaidiscovery/chai-lab) |
| ColabFold | Mirdita et al., "ColabFold: making protein folding accessible to all" | [`10.1038/s41592-022-01488-1`](https://doi.org/10.1038/s41592-022-01488-1) | [`sokrypton/ColabFold`](https://github.com/sokrypton/ColabFold) |
| OpenFold | Ahdritz et al., "OpenFold: retraining AlphaFold2 yields new insights into its learning mechanisms and capacity for generalization" | [`10.1038/s41592-024-02272-z`](https://doi.org/10.1038/s41592-024-02272-z) | [`aqlaboratory/openfold`](https://github.com/aqlaboratory/openfold) |
| OpenFold3-preview | The OpenFold3 Team, OpenFold3-preview software / report | [`10.5281/zenodo.19001000`](https://doi.org/10.5281/zenodo.19001000) | [`aqlaboratory/openfold-3`](https://github.com/aqlaboratory/openfold-3) |
| Protenix | Zhang et al., "Protenix-v1: Toward High-Accuracy Open-Source Biomolecular Structure Prediction" | [`10.64898/2026.02.05.703733`](https://doi.org/10.64898/2026.02.05.703733) | [`bytedance/Protenix`](https://github.com/bytedance/Protenix) |
| AlphaFold2 | Jumper et al., "Highly accurate protein structure prediction with AlphaFold" | [`10.1038/s41586-021-03819-2`](https://doi.org/10.1038/s41586-021-03819-2) | [`google-deepmind/alphafold`](https://github.com/google-deepmind/alphafold) |

AlphaFold2 is listed as a reference method only; it is not reported as a successful benchmarked backend in this export because the source benchmark repo does not currently have the official AlphaFold2 parameters and database layout installed.

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
| ColabFold | shared precomputed ColabFold/MMseqs2 MSA | shared A3M used directly as ColabFold input; no MMseqs search inside model runner |
| OpenFold | shared precomputed ColabFold/MMseqs2 MSA | shared A3M copied into OpenFold precomputed-alignment layout |
| Protenix | shared precomputed ColabFold/MMseqs2 MSA | shared A3M converted to paired/unpaired Protenix inputs |
| OpenFold3 | shared precomputed ColabFold/MMseqs2 MSA | shared A3M copied as `cfdb_hits.a3m`; low-memory backend on RTX A5000 |

## Carbon Method

- Tracker: CodeCarbon offline tracker.
- Current default benchmark policy: world-average accounting. New `--track-carbon` runs omit `--carbon-country-iso-code` and record `carbon_country_iso_code=WORLD`, `carbon_intensity_mode=world_average`, and `carbon_intensity_source=configurable_default_world_average`.
- Default world-average intensity in the benchmark repo: `475 g CO2e/kWh`.
- The Carbon4Science `results/` directory is kept clean and currently contains only the latest clean export: default/no-MSA rows for ESMFold, OmegaFold, Boltz-2, and Chai-1 plus unified shared-MSA rows for ColabFold, OpenFold, Protenix, and OpenFold3.
- Raw CodeCarbon CSVs are retained in the benchmark repo under each result directory's `carbon/` folder and summarized in `results/benchmark-metadata.csv`.

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
2. The resulting per-target A3M files and MSA search carbon metadata are stored in the source benchmark result directory.
3. ColabFold, OpenFold, Protenix, and OpenFold3 reuse those same A3M files; no model runner reruns MMseqs in this benchmark.
4. Model inference rows mark `msa_generation_included_in_timing=false`, `msa_generation_included_in_carbon=false`, and `msa_reused=true`.
5. `benchmark-score.csv` records per-target scores and runtime/carbon totals. For shared-MSA models it joins MSA cost, model inference cost, and structure scores; for default/no-MSA models the total cost is the model run cost.

In the shared-MSA table below, model-only CO2e is reported separately from total CO2e with shared MSA cost added back for end-to-end comparability.

## Results

<!-- Generated by scripts/make_readme_benchmark_tables.py -->

### Default/native first-five benchmark

| Model | Mode / MSA | n_success | Mean lDDT-Ca | Mean TM-score | Mean Ca RMSD (A) | Mean inference time (s) | Mean model CO2e (g) | MSA cost included? | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| esmfold | no; native_single_sequence | 5 | 0.748 | 0.750 | 5.888 | 57.8 | 1.04 | no | single-sequence language model |
| omegafold | no; native_single_sequence | 5 | 0.671 | 0.700 | 7.378 | 86.6 | 1.56 | no | single-sequence model |
| boltz2 | no; model_default_no_msa | 5 | 0.564 | 0.591 | 11.693 | 54.3 | 0.98 | no | local runner uses explicit no-MSA mode |
| chai1 | no; native_embedding_no_msa | 5 | 0.721 | 0.718 | 6.552 | 104.5 | 1.89 | no | default Chai-1 uses embeddings without MSAs/templates |
| colabfold | yes; default_msa | 5 | 0.842 | 0.819 | 4.718 | 618.1 | 11.15 | yes | fresh ColabFold/MMseqs search per target |
| openfold | yes; default_msa | 5 | 0.852 | 0.848 | 3.693 | 545.1 | 9.83 | yes | fresh ColabFold/MMseqs A3M passed to OpenFold |

### Unified shared-MSA first-five benchmark

| Model | Mode / MSA | n_success | Mean lDDT-Ca | Mean TM-score | Mean Ca RMSD (A) | Mean model time (s) | Mean model CO2e (g) | Mean total time with shared MSA (s) | Mean total CO2e with shared MSA (g) | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| colabfold | yes; shared_precomputed_msa | 5 | 0.840 | 0.816 | 4.743 | 127.6 | 4.24 | 601.7 | 17.82 | shared A3M used directly as ColabFold input; no MMseqs search inside model runner |
| openfold | yes; shared_precomputed_msa | 5 | 0.829 | 0.819 | 3.897 | 55.3 | 2.52 | 529.5 | 16.11 | shared A3M copied into OpenFold precomputed-alignment layout |
| protenix | yes; shared_precomputed_msa | 5 | 0.867 | 0.855 | 4.452 | 91.6 | 2.72 | 565.7 | 16.31 | shared A3M converted to Protenix paired/unpaired inputs |
| openfold3 | yes; shared_precomputed_msa | 5 | 0.852 | 0.835 | 5.195 | 113.9 | 3.72 | 588.0 | 17.31 | shared A3M copied as cfdb_hits.a3m; low-memory experimental run |

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

The `results/` directory is intentionally clean and contains only the latest export:

- `results/benchmark-score.csv` - per-target score and cost rows for all exported models.
- `results/benchmark-metadata.csv` - per-target runtime, carbon, and MSA metadata rows.
- `results/esmfold.json`
- `results/omegafold.json`
- `results/boltz2.json`
- `results/chai1.json`
- `results/colabfold.json`
- `results/openfold.json`
- `results/openfold3.json`
- `results/protenix.json`

## Reproduction

Latest unified shared-MSA first-five benchmark command shape from the source repo:

```bash
conda run -n folding-benchmark python scripts/run_benchmark_from_targets.py \
  --targets data/targets/targets_first5.csv \
  --config tmp/backend_smoke/models_four_msa_shared.yaml \
  --models colabfold,openfold,protenix,openfold3 \
  --top-k 1 \
  --shared-msa-metadata results/four_msa_models_shared_msa_first5/msa/msa_metadata.csv \
  --shared-msa-root results/four_msa_models_shared_msa_first5/msa/msas \
  --track-carbon
```

## Limitations

- This is a five-target smoke benchmark, not a full CASP benchmark.
- Protenix and OpenFold3 are experimental shared-MSA exports and are not canonical enabled backends in `Protein-Folding-Benchmark/configs/models.yaml`; ColabFold and OpenFold are included here through shared-MSA side-study runners.
- OpenFold3 uses a low-memory configuration validated on the local 24 GB RTX A5000 setup; broader validation is still needed.
- Official AlphaFold2 is not reported as a benchmarked method here because the official parameters and AlphaFold database layout are not installed in the source repo. ColabFold is reported as ColabFold, not relabeled as AF2.
