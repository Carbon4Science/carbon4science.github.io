# Protein Folding

**Task description:** Protein structure prediction from amino-acid sequence.

## Goal

Compare structure-prediction accuracy against runtime, energy use, and CO2 emissions for locally runnable folding backends from `Protein-Folding-Benchmark`.

The current Carbon4Science export is the **45-target CASP15/CASP16 unique `<1000` residue benchmark** built from three combined benchmark runs:

```text
/home/chen/projects/Protein-Folding-Benchmark/results/2026-06-06-combine-8models
```

It contains **360 scored protein/model rows: 45 targets × 8 scored models**. The scored models are `af2`, `colabfold`, `openfold`, `protenix`, `chai1`, `esmfold`, `omegafold`, and `boltz2`. All 45 targets are successfully predicted and scored for every model.

## Dataset

- **Dataset name:** CASP15/CASP16 unique PDB-chain target set, residue count `<1000`.
- **N:** 45 targets (common across all three combined runs).
- **Source target CSV:** `data/targets/targets_casp15_casp16_unique_lt1000_prepared.csv` in `Protein-Folding-Benchmark`.
- **Exported dataset CSV:** `results/benchmark-dataset.csv`.
- **Export manifest:** `results/benchmark_collection_manifest.csv`.
- **References:** Per-target reference PDBs are extracted from cached mmCIF files using the manifest's explicit `chain_id` and `residue_start`/`residue_end` range. Reference paths are recorded in `results/benchmark_scores_all_models.csv` and each model's JSON file.

## Method References

| Method | Paper / report | DOI | GitHub repository |
|---|---|---|---|
| ESMFold | Lin et al., "Evolutionary-scale prediction of atomic-level protein structure with a language model" | [`10.1126/science.ade2574`](https://doi.org/10.1126/science.ade2574) | [`facebookresearch/esm`](https://github.com/facebookresearch/esm) |
| OmegaFold | Wu et al., "High-resolution de novo structure prediction from primary sequence" | [`10.1101/2022.07.21.500999`](https://doi.org/10.1101/2022.07.21.500999) | [`HeliXonProtein/OmegaFold`](https://github.com/HeliXonProtein/OmegaFold) |
| Boltz-2 | Passaro et al., "Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction" | [`10.1101/2025.06.14.659707`](https://doi.org/10.1101/2025.06.14.659707) | [`jwohlwend/boltz`](https://github.com/jwohlwend/boltz) |
| Chai-1 | Boitreaud et al., "Chai-1: Decoding the molecular interactions of life" | [`10.1101/2024.10.10.615955`](https://doi.org/10.1101/2024.10.10.615955) | [`chaidiscovery/chai-lab`](https://github.com/chaidiscovery/chai-lab) |
| ColabFold | Mirdita et al., "ColabFold: making protein folding accessible to all" | [`10.1038/s41592-022-01488-1`](https://doi.org/10.1038/s41592-022-01488-1) | [`sokrypton/ColabFold`](https://github.com/sokrypton/ColabFold) |
| OpenFold | Ahdritz et al., "OpenFold: retraining AlphaFold2 yields new insights into its learning mechanisms and capacity for generalization" | [`10.1038/s41592-024-02272-z`](https://doi.org/10.1038/s41592-024-02272-z) | [`aqlaboratory/openfold`](https://github.com/aqlaboratory/openfold) |
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
| `GDT_TS` / `gdt_ts` | Global Distance Test - Total Score on a 0–1 scale. |
| `GDT_TS_percent` / `gdt_ts_percent` | Global Distance Test - Total Score on a 0–100 scale. |
| `inference_time_sec` | Controller wall-clock runtime per target/model. |
| `carbon_emissions_g` | CodeCarbon offline emissions in grams CO2e. |
| `carbon_energy_consumed_kwh` | CodeCarbon energy consumption in kWh. |

`GDT_TS` is the visible alias used in the exported Carbon4Science CSVs; the lowercase `gdt_ts` column is retained for compatibility with the benchmark scripts. Scoring uses `--match-mode sequence` (Needleman-Wunsch Cα alignment against per-target reference PDBs).

## Models

| Model | Mode / MSA in current export | Notes |
|---|---|---|
| `af2` | MSA; `official_af2_database_search` | official DeepMind AlphaFold2 database search; split MSA/features and JAX inference carbon accounting |
| `colabfold` | MSA; `shared_precomputed_msa` | reuses precomputed ColabFold/MMseqs2 A3M metadata from the benchmark run |
| `openfold` | MSA; `shared_precomputed_msa` | loads AlphaFold2 `params_model_1` weights through the OpenFold inference engine (weights byte-identical to `af2`; see Training-data cutoffs) |
| `protenix` | MSA; `shared_precomputed_msa` | ByteDance AF3-like diffusion model; reuses ColabFold/MMseqs2 A3M |
| `chai1` | no MSA; `native_embedding_no_msa` | default Chai-1 embeddings without external MSAs/templates |
| `esmfold` | no MSA; `native_single_sequence` | sequence-language-model baseline |
| `omegafold` | no MSA; `native_single_sequence` | sequence-only baseline |
| `boltz2` | no MSA; `model_default_no_msa` | canonical Boltz backend ID is `boltz2` |

## Training-data cutoffs

Training-run timestamps are not embedded in any checkpoint. The benchmark-relevant date is the **PDB release-date cutoff** used when building each model's training set — the newest structure the model was allowed to see, which determines potential leakage into the CASP15/16 target set.

| Model | PDB cutoff | Status | Source |
|---|---|---|---|
| af2 | 2018-04-30 | Confirmed | [Jumper et al. 2021 suppl.](https://www.nature.com/articles/s41586-021-03819-2) (CASP13-aligned monomer training cutoff) |
| colabfold | 2018-04-30 | Confirmed | inherits AF2 `alphafold2_ptm` weights → same cutoff |
| openfold | 2018-04-30 | Confirmed | **this benchmark** runs AF2 `params_model_1` weights via the OpenFold engine; not OpenFold's own retrained (Dec-2021) checkpoint |
| esmfold | 2020-05-01 | Confirmed | folding-head PDB cutoff; [Lin et al. 2023](https://www.science.org/doi/10.1126/science.ade2574) |
| chai1 | 2021-01-12 | Confirmed | [Chai-1 technical report](https://chaiassets.com/chai-1/paper/technical_report_v1.pdf) |
| protenix | 2021-09-30 | Confirmed | `protenix_base_default_v1.0.0` follows AF3 training cutoff; [Protenix README](https://github.com/bytedance/Protenix/blob/main/README.md) |
| boltz2 | 2023-06-01 | Confirmed | [Passaro et al. 2025](https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1.full): "every PDB structure up to 06/01/2023" |
| omegafold | not documented | Unconfirmed | no explicit PDB date published by the authors; [~2022 inferred from release](https://github.com/HeliXonProtein/OmegaFold/issues/13) |

Our 45-target set contains **33 CASP15 targets (2022)** and **12 CASP16 targets (2024)**. All cutoffs predate CASP16. Only **Boltz-2 (cutoff 2023-06-01)** postdates CASP15, meaning its 33 CASP15-target results should be interpreted with a potential training-data leakage caveat. All other models predated CASP15.

## Carbon Method

- Tracker: CodeCarbon offline tracker.
- Accounting policy: world-average emissions intensity.
- Recorded country code: `WORLD`.
- Recorded intensity source: `configurable_default_world_average`.
- Default intensity: `475 g CO2e/kWh`.
- Raw CodeCarbon CSVs remain in the source benchmark result directory under `predictions/<target>/<model>/carbon/`. Carbon4Science stores cleaned per-protein metadata in `results/benchmark-metadata.csv`.

## Hardware

- CPU: Intel Xeon Gold 6240R, 1 socket, 24 cores / 48 threads.
- RAM: 251 GiB visible.
- GPU: 3 × NVIDIA RTX A5000, 24,564 MiB each.
- Driver: 580.159.03.
- CUDA reported by driver: 13.0.
- OS/kernel: Ubuntu Linux, kernel `6.8.0-117-generic`, x86_64.

## Current Results

Source summary: `results/benchmark_model_summary_all_models.csv`.

**Dataset:** CASP15/CASP16 unique <1000-residue monomers · **N =** 45 targets · **Metric:** lDDT-Cα (primary) · **CO₂/job:** per target (exp = 45 targets)

**Hardware:** 3 × NVIDIA RTX A5000 (24 GB) · Intel Xeon Gold 6240R (24c/48t) · 251 GiB RAM

| Year | Venue        | Model     | Architecture         | Params | lDDT-Cα   | TM-score  | GDT_TS (%) | Cα-RMSD (Å) | CO₂/exp (g) | CO₂/job (g) | Time/exp (s) | Time/job (s) |
| ---- | ------------ | --------- | -------------------- | ------ | --------- | --------- | ---------- | ----------- | ----------- | ----------- | ------------ | ------------ |
| 2021 | Nature       | af2       | Evoformer + MSA      | 93.2 M              | 0.868     | 0.761     | 59.15      | 11.379      | 2,103.0     | 46.73       | 77,832       | 1,729.6      |
| 2022 | Nat. Methods | colabfold | Evoformer + MMseqs2  | 93.2 M              | **0.876** | 0.770     | **60.96**  | 11.972      | 522.2       | 11.60       | 30,126       | 669.5        |
| 2022 | bioRxiv      | omegafold | PLM + Geoformer      | 795 M               | 0.770     | 0.669     | 47.18      | 17.345      | **180.1**   | **4.00**    | **5,535**    | **123.0**    |
| 2023 | Science      | esmfold   | ESM-2 LM + folding   | 693 M (+2.84B ESM2) | 0.811     | 0.704     | 52.36      | 15.395      | 237.0       | 5.27        | 16,345       | 363.2        |
| 2024 | bioRxiv      | chai1     | Diffusion (AF3-like) | 316 M (+2.84B ESM2) | 0.798     | 0.695     | 48.79      | 17.648      | 892.4       | 19.83       | 63,738       | 1,416.4      |
| 2024 | Nat. Methods | openfold  | Evoformer + MSA      | 93.2 M              | 0.875     | **0.771** | 60.84      | **11.734**  | 477.6       | 10.61       | 26,854       | 596.8        |
| 2025 | bioRxiv      | boltz2    | Diffusion (AF3-like) | 521 M               | 0.765     | 0.714     | 51.82      | 17.605      | 796.5       | 17.70       | 53,137       | 1,180.8      |
| 2025 | bioRxiv      | protenix  | Diffusion (AF3-like) | 368 M               | 0.871     | 0.744     | 57.50      | 15.361      | 442.3       | 9.83        | 27,555       | 612.3        |

All 45/45 targets scored successfully for every model. **Bold** = best value in column (for Cα-RMSD, lower is better). GDT_TS is shown on a 0–100 scale (`gdt_ts_percent`). CO₂/exp and Time/exp are totals over all 45 targets using `total_carbon_with_shared_msa_g` / `total_time_with_shared_msa_sec` from `results/benchmark-score.csv`; per-job columns divide by 45.

Parameter counts were measured directly from the local model weights: JAX `.npz` array sizes (summed `arr.size`) for the AF2-family models (af2, colabfold, openfold reuse the same Evoformer weights at 93.2M each); summed `tensor.numel()` over PyTorch/TorchScript checkpoint weights containers for the remaining models. `(+2.84B ESM2)` denotes the separate ESM-2 3B language model (`esm2_t36_3B_UR50D`) that `chai1` and `esmfold` load as a sequence embedder at inference; the folding-trunk parameters are listed first.

## Exported Files

The `results/` directory is intentionally clean and contains the latest export:

- `results/benchmark-dataset.csv` — the 45-target benchmark dataset: target IDs, sequences, reference PDB paths, and CASP notes.
- `results/benchmark-score.csv` — compact per-protein/model score, GDT_TS, runtime, and carbon rows.
- `results/benchmark_scores_all_models.csv` — detailed per-protein/model score rows, including references and prediction paths.
- `results/benchmark-metadata.csv` — per-protein/model runtime, carbon, MSA, and provenance rows.
- `results/benchmark_metadata_all_models.csv` — same metadata table retained for compatibility with prior exports.
- `results/benchmark_model_summary_all_models.csv` — one summary row per exported model.
- `results/benchmark_collection_manifest.csv` — source result directory, target set, scored row counts, JSON file count, and GDT_TS scale notes.
- `results/score_metric_definitions.csv` and `results/score_metric_definitions.md` — score metric definitions.
- `results/af2.json`
- `results/colabfold.json`
- `results/openfold.json`
- `results/protenix.json`
- `results/chai1.json`
- `results/esmfold.json`
- `results/omegafold.json`
- `results/boltz2.json`

## Reproduction

The 45-target benchmark is assembled from three separate backend runs and scored with the combined-run scripts. From the `Protein-Folding-Benchmark` repo root:

**1. Combine predictions from the three source runs:**
```bash
conda run -n folding-benchmark python scripts/combine_8model_predictions.py \
  --dest results/2026-06-06-combine-8models
```

**2. Build per-target reference PDBs (correct chain + residue range from manifest):**
```bash
conda run -n folding-benchmark python scripts/build_combined_references.py \
  --run-dir results/2026-06-06-combine-8models
```

**3. Score all 45 targets × 8 models:**
```bash
conda run -n folding-benchmark python scripts/score_combined_8models.py \
  --run-dir results/2026-06-06-combine-8models \
  --match-mode sequence
```

**4. Export to carbon4science format:**
```bash
conda run -n folding-benchmark python scripts/export_combined_to_carbon4science.py \
  --run-dir results/2026-06-06-combine-8models \
  --out-dir ../carbon4science.github.io/results
```

Source runs:

| Group | Result directory | Models |
|---|---|---|
| MSA-free | `results/20260601_104732_casp15_casp16_unique_lt1000_all_default_msa-free` | chai1, esmfold, omegafold |
| ColabFold MSA | `results/20260603_142659_casp15_casp16_unique_lt1000_all_default_colabfold` | boltz2, colabfold, openfold, protenix |
| AlphaFold2 | `results/20260604_134750_casp15_casp16_unique_lt1000_all_default-af2` | af2 |

## Limitations

- This is a 45-target local benchmark, not the full CASP15/CASP16 benchmark universe.
- Model outputs are local-run artifacts from one workstation and should be interpreted with that hardware and software context.
- Carbon estimates use CodeCarbon offline world-average accounting, not direct facility metering.
- References are extracted per-target from mmCIF files using the manifest's `chain_id` and `residue_start`/`residue_end`; results are not directly comparable to benchmarks using different reference extraction strategies.
- Training-data cutoffs are documented in the `## Training-data cutoffs` section. Boltz-2 (cutoff 2023-06-01) postdates the CASP15 targets (33/45) and may have trained on structures overlapping that target set. OmegaFold's cutoff is not publicly documented.
