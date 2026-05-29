# Protein Folding

**Task description:** Protein structure prediction from amino-acid sequence.

## Goal

Compare structure-prediction accuracy against runtime, energy use, and CO2 emissions for locally runnable folding backends from `Protein-Folding-Benchmark`.

The current clean Carbon4Science export is the 52-target CASP15/CASP16 unique `<1000` residue benchmark copied from:

```text
/home/chen/projects/Protein-Folding-Benchmark/results/casp15_casp16_unique_lt1000_all_default_20260529_resume
```

It contains 416 scored protein/model rows: 52 targets x 8 scored models. The scored models are `af2`, `colabfold`, `openfold`, `chai1`, `esmfold`, `omegafold`, `boltz2`, and `openfold3`. The exported JSON bundle includes those 8 scored model files plus `protenix.json`; `protenix.json` is an explicit not-included placeholder for this export, rather than stale historical data.

## Dataset

- **Dataset name:** CASP15/CASP16 unique PDB-chain target set, residue count `<1000`.
- **N:** 52 targets.
- **Source target CSV:** `data/targets/targets_casp15_casp16_unique_lt1000_prepared.csv` in `Protein-Folding-Benchmark`.
- **Export manifest:** `results/benchmark_collection_manifest.csv`.
- **References:** Reference PDB paths are recorded in `results/benchmark_scores_all_models.csv` and model JSON files.

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

AlphaFold2 is reported as the official `af2` backend using DeepMind AlphaFold2 source, official parameters, full AlphaFold database search, and split MSA/features versus JAX inference carbon accounting. ColabFold remains reported separately as ColabFold, not relabeled as AF2.

## Benchmark Protocol

The source benchmark repo is `/home/chen/projects/Protein-Folding-Benchmark`. Each backend uses the standard runner interface:

```bash
bash runners/run_MODEL.sh input.fasta output_dir top_k
```

Each runner writes standardized predictions as `rank_001.pdb`, `rank_002.pdb`, and so on, plus `metadata.json`. This export uses `top_k=1` for compact, comparable carbon accounting.

Scoring reports:

| Metric | Description |
|---|---|
| `lddt_ca` | lDDT over C-alpha atoms; primary ranking metric. |
| `tmalign_tm_score_ref` | TM-score normalized by reference length from USalign/TM-align. |
| `ca_rmsd` | C-alpha RMSD after sequential alignment. |
| `GDT_TS` / `gdt_ts` | Global Distance Test - Total Score on a 0-1 scale. |
| `GDT_TS_percent` / `gdt_ts_percent` | Global Distance Test - Total Score on a 0-100 scale. |
| `inference_time_sec` | Controller wall-clock runtime per target/model. |
| `carbon_emissions_g` | CodeCarbon offline emissions in grams CO2e. |
| `carbon_energy_consumed_kwh` | CodeCarbon energy consumption in kWh. |

`GDT_TS` is the visible alias used in the exported Carbon4Science CSVs; the lowercase `gdt_ts` column is retained for compatibility with the benchmark scripts.

## Models

| Model | Mode / MSA in current export | Notes |
|---|---|---|
| `af2` | MSA; `official_af2_database_search` | official DeepMind AlphaFold2 database search; split MSA/features and JAX inference accounting |
| `colabfold` | MSA; `shared_precomputed_msa` | reuses precomputed ColabFold/MMseqs2 A3M metadata from the benchmark run |
| `openfold` | MSA; `shared_precomputed_msa` | reuses precomputed ColabFold/MMseqs2 A3M in OpenFold-compatible layout |
| `chai1` | no MSA; `native_embedding_no_msa` | default Chai-1 embeddings without external MSAs/templates |
| `esmfold` | no MSA; `native_single_sequence` | sequence-language-model baseline |
| `omegafold` | no MSA; `native_single_sequence` | sequence-only baseline |
| `boltz2` | no MSA; `model_default_no_msa` | canonical Boltz backend ID is `boltz2` |
| `openfold3` | MSA metadata unknown in current runner export | experimental backend; current metadata records `model_default_unknown` |
| `protenix` | not scored in this export | included as `results/protenix.json` placeholder; no 52-target score/runtime rows in the current all-default run |

## Carbon Method

- Tracker: CodeCarbon offline tracker.
- Accounting policy: world-average emissions intensity.
- Recorded country code: `WORLD`.
- Recorded intensity source: `configurable_default_world_average`.
- Default intensity: `475 g CO2e/kWh`.
- Raw CodeCarbon CSVs remain in the source benchmark result directory. Carbon4Science stores cleaned per-protein metadata in `results/benchmark-metadata.csv`.

## Hardware

- CPU: Intel Xeon Gold 6240R, 1 socket, 24 cores / 48 threads.
- RAM: 251 GiB visible.
- GPU: 3 x NVIDIA RTX A5000, 24,564 MiB each.
- Driver: 580.159.03.
- CUDA reported by driver: 13.0.
- OS/kernel: Ubuntu Linux, kernel `6.8.0-117-generic`, x86_64.

## Current Results

Source summary: `results/benchmark_model_summary_all_models.csv`.

| Model | Targets successful | Mean lDDT-Ca | Mean TM-score | Mean GDT_TS | Mean Ca RMSD (A) |
|---|---:|---:|---:|---:|---:|
| af2 | 52/52 | 0.845 | 0.773 | 0.246 | 6.527 |
| colabfold | 52/52 | 0.825 | 0.761 | 0.247 | 7.417 |
| openfold | 52/52 | 0.821 | 0.759 | 0.247 | 6.846 |
| chai1 | 52/52 | 0.727 | 0.662 | 0.242 | 11.431 |
| esmfold | 52/52 | 0.722 | 0.664 | 0.243 | 10.042 |
| omegafold | 52/52 | 0.686 | 0.627 | 0.165 | 11.885 |
| boltz2 | 52/52 | 0.546 | 0.494 | 0.182 | 16.132 |
| openfold3 | 52/52 | 0.406 | 0.364 | 0.157 | 19.353 |
| protenix | 0/52 |  |  |  |  |

The all-target summary includes both lowercase compatibility columns such as `mean_best_gdt_ts` and visible aliases such as `mean_best_GDT_TS`.

`protenix` is listed here for completeness because `results/protenix.json` is part of the exported JSON bundle; it has no scored rows in `benchmark_model_summary_all_models.csv` for this run.

## Exported Files

The `results/` directory is intentionally clean and contains the latest export:

- `results/benchmark-score.csv` - compact per-protein/model score, GDT_TS, runtime, and carbon rows.
- `results/benchmark_scores_all_models.csv` - detailed per-protein/model score rows, including references and prediction paths.
- `results/benchmark-metadata.csv` - latest successful per-protein/model runtime, carbon, MSA, and provenance rows.
- `results/benchmark_metadata_all_models.csv` - same cleaned metadata table retained for compatibility with prior exports.
- `results/benchmark_model_summary_all_models.csv` - one summary row per exported model.
- `results/benchmark_collection_manifest.csv` - source result directory, target set, scored row counts, JSON file count, and GDT_TS scale notes.
- `results/score_metric_definitions.csv` and `results/score_metric_definitions.md` - score metric definitions.
- `results/af2.json`
- `results/colabfold.json`
- `results/openfold.json`
- `results/chai1.json`
- `results/esmfold.json`
- `results/omegafold.json`
- `results/boltz2.json`
- `results/openfold3.json`
- `results/protenix.json` - included in the JSON bundle as an explicit not-included placeholder for this export.

## Reproduction

Command shape used for the 52-target benchmark in the source repo:

```bash
conda run -n folding-benchmark python scripts/run_benchmark_from_targets.py \
  --targets data/targets/targets_casp15_casp16_unique_lt1000_prepared.csv \
  --config results/casp15_casp16_unique_lt1000_all_default_20260529_resume/models_all_available_default.yaml \
  --models af2,colabfold,openfold,chai1,esmfold,omegafold,boltz2,openfold3 \
  --top-k 1 \
  --predictions-dir results/casp15_casp16_unique_lt1000_all_default_20260529_resume/predictions \
  --sequences-dir results/casp15_casp16_unique_lt1000_all_default_20260529_resume/sequences \
  --logs-dir results/casp15_casp16_unique_lt1000_all_default_20260529_resume/logs \
  --results-dir results/casp15_casp16_unique_lt1000_all_default_20260529_resume \
  --run-metadata results/casp15_casp16_unique_lt1000_all_default_20260529_resume/run_metadata.csv \
  --run-status results/casp15_casp16_unique_lt1000_all_default_20260529_resume/run_status.csv \
  --resume \
  --track-carbon \
  --carbon-country-iso-code WORLD
```

Scoring command shape:

```bash
conda run -n folding-benchmark python scripts/score_benchmark_from_targets.py \
  --targets data/targets/targets_casp15_casp16_unique_lt1000_prepared.csv \
  --config results/casp15_casp16_unique_lt1000_all_default_20260529_resume/models_all_available_default.yaml \
  --models openfold,openfold3,boltz2,chai1,esmfold,colabfold,af2,omegafold \
  --top-k 1 \
  --predictions-dir results/casp15_casp16_unique_lt1000_all_default_20260529_resume/predictions \
  --scores-dir results/casp15_casp16_unique_lt1000_all_default_20260529_resume/scores \
  --results-dir results/casp15_casp16_unique_lt1000_all_default_20260529_resume \
  --run-metadata results/casp15_casp16_unique_lt1000_all_default_20260529_resume/run_metadata.csv \
  --use-tmalign \
  --use-gdt-ts
```

## Limitations

- This is a 52-target local benchmark, not the full CASP15/CASP16 benchmark universe.
- Model outputs are local-run artifacts from one workstation and should be interpreted with that hardware and software context.
- `openfold3` remains experimental in this harness; current metadata does not confirm its MSA provenance.
- `protenix` is not included in the current 52-target all-default export.
- Carbon estimates use CodeCarbon offline world-average accounting, not direct facility metering.
