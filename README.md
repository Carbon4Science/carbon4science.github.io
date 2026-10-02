# Protein Folding

**Task description:** Protein structure prediction from amino-acid sequence.

## Goal

Compare structure-prediction accuracy against runtime, energy use, and CO2 emissions for locally runnable folding backends from `Protein-Folding-Benchmark`.

The current Carbon4Science export is the **45-target CASP15/CASP16 unique `<1000` residue benchmark** built from three combined benchmark runs:

```text
/home/chen/projects/Protein-Folding-Benchmark/results/2026-06-09-combine-8models-gpu
```

It contains **360 scored protein/model rows: 45 targets × 8 scored models**. The scored models are `af2`, `colabfold`, `openfold`, `protenix`, `chai1`, `esmfold`, `omegafold`, and `boltz2`. All 45 targets are successfully predicted and scored for every model. `chai1`, `esmfold`, and `boltz2` were re-run on GPU (2026-06-09) to make all eight models GPU-measured; the other five models are unchanged from the original runs.

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
| `boltz2` | MSA; `shared_precomputed_msa` | reuses precomputed ColabFold/MMseqs2 A3M; charged the shared MSA-build cost like openfold/protenix |

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

Source summary: `results/benchmark_model_summary_all_models.csv`. Per-replicate values and across-replicate statistics: the `replicates`, `aggregate_summary_across_reps` and `cost_totals_across_reps` blocks in each `results/{model}.json`.

**Dataset:** CASP15/CASP16 unique <1000-residue monomers · **N =** 45 targets · **Replicates:** n = 3 · **Metric:** lDDT-Cα (primary) · **CO₂/job:** per target (exp = 45 targets)

**Hardware:** 3 × NVIDIA RTX A5000 (24 GB) · Intel Xeon Gold 6240R (24c/48t) · 251 GiB RAM

Every value is **mean ± sample standard deviation across three independent replicates** of the complete benchmark (2026-06-09, 2026-08-02, 2026-08-03). All **1080/1080 predictions succeeded** (45 targets × 8 models × 3 replicates).

| Year | Venue        | Model     | Architecture         | Params | lDDT-Cα   | TM-score  | GDT_TS (%) | Cα-RMSD (Å) | CO₂/exp (g) | CO₂/job (g) | Time/exp (s) | Time/job (s) |
| ---- | ------------ | --------- | -------------------- | ------ | --------- | --------- | ---------- | ----------- | ----------- | ----------- | ------------ | ------------ |
| 2021 | Nature       | af2       | Evoformer + MSA      | 93.2 M              | 0.8673 ± 0.0034 | 0.7626 ± 0.0026 | 70.15 ± 0.12 | **11.138 ± 0.244** | 2,090.5 †     | 46.46 †     | 77,686 †      | 1,726.3 †     |
| 2022 | Nat. Methods | colabfold | Evoformer + MMseqs2  | 93.2 M              | **0.8760 ± 0.0002** | **0.7698 ± 0.0007** | 70.87 ± 0.03 | 11.964 ± 0.023 | 392.7 ± 112.4 | 8.73 ± 2.50 | 30,109 ± 638  | 669.1 ± 14.2  |
| 2022 | bioRxiv      | omegafold | PLM + Geoformer      | 795 M               | 0.7697 ± 0.0000 | 0.6690 ± 0.0000 | 59.19 ± 0.00 | 17.345 ± 0.000 | 209.9 ± 25.8  | 4.66 ± 0.57 | 5,110 ± 368   | 113.6 ± 8.2   |
| 2023 | Science      | esmfold   | ESM-2 LM + folding   | 693 M (+2.84B ESM2) | 0.8101 ± 0.0000 | 0.7042 ± 0.0000 | 64.08 ± 0.00 | 15.397 ± 0.000 | **79.3 ± 6.9**  | **1.76 ± 0.15** | **2,826 ± 355** | **62.8 ± 7.9** |
| 2024 | bioRxiv      | chai1     | Diffusion (AF3-like) | 316 M (+2.84B ESM2) | 0.7977 ± 0.0009 | 0.6928 ± 0.0025 | 62.12 ± 0.13 | 17.479 ± 0.679 | 207.9 ± 20.5  | 4.62 ± 0.45 | 6,019 ± 1,095 | 133.8 ± 24.3  |
| 2024 | Nat. Methods | openfold  | Evoformer + MSA      | 93.2 M              | 0.8753 ± 0.0018 | **0.7698 ± 0.0033** | **71.22 ± 0.33** | 11.695 ± 0.033 | 400.6 ± 67.2  | 8.90 ± 1.49 | 26,764 ± 615  | 594.7 ± 13.7  |
| 2025 | bioRxiv      | boltz2    | Diffusion (AF3-like) | 521 M               | 0.8627 ± 0.0018 | 0.7321 ± 0.0026 | 68.39 ± 0.13 | 17.579 ± 0.151 | 362.1 ± 58.1  | 8.05 ± 1.29 | 26,555 ± 614  | 590.1 ± 13.6  |
| 2025 | bioRxiv      | protenix  | Diffusion (AF3-like) | 368 M               | 0.8709 ± 0.0008 | 0.7444 ± 0.0006 | 69.45 ± 0.09 | 15.361 ± 0.001 | 377.8 ± 56.8  | 8.40 ± 1.26 | 27,808 ± 666  | 618.0 ± 14.8  |

**Bold** = best value in column (for Cα-RMSD, lower is better). GDT_TS is shown on a 0–100 scale (`gdt_ts_percent`). CO₂/exp and Time/exp are totals over all 45 targets; per-job columns divide by 45. For the four shared-MSA models (colabfold, openfold, protenix, boltz2) the total includes the shared ColabFold/MMseqs2 MSA-build cost; each model is charged the same per-target build time even though the MSA is computed once, so these totals compare models to each other but are **not additive across models**. Counting the shared MSA once, a whole replicate costs **16.5 ± 0.27 h**.

**colabfold and openfold are not separable in this benchmark.** colabfold leads on lDDT-Cα by 0.0007, which is smaller than openfold's own across-replicate std of 0.0018; they tie on TM-score to four decimals, and openfold leads on GDT_TS. Treat the top two as a tie rather than a ranking.

**Only two of the eight models are deterministic.** esmfold and omegafold reproduce exactly across replicates (0.000000 spread on all 45 targets). At the per-target level, 152 of the 315 non-af2 (target, model) pairs exceed a 1e-3 lDDT-Cα tolerance — largest spreads boltz2 0.105, openfold 0.089, chai1 0.082, protenix 0.062, colabfold 0.0055. chai1 uses no MSA, so its spread cannot come from MSA rebuilding; that isolates genuine sampling nondeterminism in the diffusion-based models. Per-target variation nevertheless averages out at the benchmark level, where every across-replicate std is ≤ 0.0034 lDDT-Cα.

† **af2's cost is a composite, not a mean.** Replicates 2 and 3 reuse replicate 1's MSA features (`features.pkl`), which saves ~20 h each. af2's MSA build is therefore measured once — 19.83 h and 1,873 g CO₂, the single largest cost item in the benchmark, exceeding an entire replicate's incremental cost — while only af2 *inference* (1.75 ± 0.04 h) has an error bar. The MSA stage is deterministic given fixed databases, so this limits the cost error bar, not accuracy.

**Energy and CO₂ error bars are inflated by replicate 1, and the cause is now identified.** The per-device breakdown below shows replicate 1 assigning a mean **69.3%** of measured energy to the GPU, against **51.1%** for replicates 2 and 3 — which agree with each other to within ~1%. Replicate 1's CPU energy is 3–4× lower for most models (boltz2 0.0075 vs 0.0337/0.0339 kWh; esmfold 0.0087 vs 0.0300/0.0302; openfold 0.0081 vs 0.0350/0.0353) even though total wall-clock is nearly identical, so its CPU tracking under-measured. This is a CodeCarbon attribution difference, not run-to-run variance. **Prefer replicates 2 and 3 for any energy or CO₂ figure**; the runtime columns and all accuracy metrics are unaffected.

### Energy by device (CPU / GPU / RAM)

CPU / GPU / RAM energy from the per-(target, model, stage) CodeCarbon records, summed over all stages, as mean ± std across the three replicates. **This is a device split of energy, not of wall time** — no device is recorded per prediction, and a stage occupies wall-clock while both CPU and GPU are partly busy, so per-device wall time is not a defined quantity here. Per-replicate values are in each `results/{model}.json` under `replicates[].cost_totals` and `cost_totals_across_reps`.

As in the Current Results table, the shared ColabFold MSA is **charged to every model that consumes it** (openfold, boltz2, protenix) even though it is built once, under colabfold — so these figures compare models to each other but are **not additive across models**.

| Model | CPU (kWh) | GPU (kWh) | RAM (kWh) | GPU share (%) | Measured wall time (h) |
| --------- | ---------------: | ---------------: | ---------------: | -----------: | ----------------: |
| af2 — MSA build (n=1) † | 0.8987 | 1.3230 | 1.9858 | 31.4 | 21.15 |
| af2 — inference (n=3) | 0.0736 ± 0.0015 | 0.2345 ± 0.0230 | 0.1496 ± 0.0050 | 51.2 | 1.73 ± 0.04 |
| colabfold | 0.0824 ± 0.0018 | 0.4571 ± 0.2356 | 0.2871 ± 0.0063 | 53.1 ± 11.4 | 8.31 ± 0.18 |
| protenix | 0.0893 ± 0.0237 | 0.3952 ± 0.1832 | 0.3109 ± 0.0421 | 48.2 ± 14.6 | 7.67 ± 0.18 |
| openfold | 0.0793 ± 0.0161 | 0.4766 ± 0.1838 | 0.2874 ± 0.0278 | 55.2 ± 11.5 | 7.38 ± 0.17 |
| boltz2 | 0.0782 ± 0.0157 | 0.4001 ± 0.1644 | 0.2841 ± 0.0282 | 51.2 ± 12.3 | 7.32 ± 0.17 |
| chai1 | 0.0474 ± 0.0266 | 0.2745 ± 0.0233 | 0.1158 ± 0.0398 | 63.5 ± 12.3 | 1.65 ± 0.30 |
| omegafold | 0.0422 ± 0.0256 | 0.2981 ± 0.0133 | 0.1016 ± 0.0421 | 68.5 ± 12.3 | 1.40 ± 0.10 |
| esmfold | 0.0229 ± 0.0124 | 0.0893 ± 0.0182 | 0.0547 ± 0.0204 | 54.4 ± 16.5 | 0.76 ± 0.10 |

Three stages are measured: `inference` (all eight models), `msa_build` (colabfold's shared MMseqs2 search, charged on to its three consumers), and `msa_features` (af2's jackhmmer/HHblits stage, present in replicate 1 only). The wide ± on several energy rows is dominated by the replicate-1 attribution difference described above rather than by genuine variation.

† **af2 is split into two rows because a mean across its replicates would be meaningless.** Replicate 1 built the MSA while replicates 2 and 3 reused `features.pkl`, so af2's per-replicate totals are 22.92 / 1.71 / 1.71 h. Averaging them gives 8.78 ± 12.24 h — a standard deviation larger than the mean, and RAM energy is more extreme still (2.1412 / 0.1467 / 0.1467 kWh, CV = 1.42). That is arithmetically correct for a sample whose largest value is ~15× the other two, but it summarises a mixture of two different experiments rather than a distribution, so no ± over it is interpretable. Split by stage, both halves are well behaved: the MSA build is a single n=1 measurement and af2 inference reproduces to ±1–3%.

The wall-time column is CodeCarbon's own stage timer, measured independently of the runtime columns in the Current Results table, so the two cross-check each other. With the shared MSA charged on, all four shared-MSA models agree with their Current Results totals to within 0.6% (colabfold 8.31 vs 8.36 h, protenix 7.67 vs 7.72, openfold 7.38 vs 7.43, boltz2 7.32 vs 7.38), as do the MSA-free models (chai1 1.65 vs 1.67, omegafold 1.40 vs 1.42, esmfold 0.76 vs 0.78).

**GDT_TS changed method in this export.** Values now come from `external_tmscore_matched` (TMscore binary on sequence-matched Cα atoms) rather than the earlier `internal_iterative_ca`, so they are not comparable with GDT_TS figures published before 2026-08-04. lDDT-Cα, TM-score and Cα-RMSD are unaffected.

Parameter counts were measured directly from the local model weights: JAX `.npz` array sizes (summed `arr.size`) for the AF2-family models (af2, colabfold, openfold reuse the same Evoformer weights at 93.2M each); summed `tensor.numel()` over PyTorch/TorchScript checkpoint weights containers for the remaining models. `(+2.84B ESM2)` denotes the separate ESM-2 3B language model (`esm2_t36_3B_UR50D`) that `chai1` and `esmfold` load as a sequence embedder at inference; the folding-trunk parameters are listed first.

## MSA Cross-Mode Variants (2026-07-09)

The main export above runs each model in its default MSA mode. This follow-up
experiment runs four of them in the **opposite** MSA mode on the **same 45
targets**, to isolate the effect of the MSA alone. Each variant is a distinct
model label; the eight-model table above is unchanged.

- **Source run:** `results/20260709_casp15_casp16_unique_lt1000_msa-variants` (in `Protein-Folding-Benchmark`).
- **MSA source:** for `chai1_msa`, the shared ColabFold/MMseqs2 A3M already computed for the CASP targets, converted to Chai-1's `.aligned.pqt` format. The three `*_nomsa` variants use no MSA (single-sequence input).
- **Exported JSONs:** `results/chai1_msa.json`, `results/boltz2_nomsa.json`, `results/colabfold_nomsa.json`, `results/openfold_nomsa.json`. All 45/45 targets predicted (GPU) and scored with the same `--match-mode sequence` protocol.

| Variant | Mode vs default | lDDT-Cα | TM-score | GDT_TS (%) | Cα-RMSD (Å) | CO₂/job (g) | Time/job (s) | Δ lDDT-Cα (vs base) |
| --------------- | ---------------------------- | ------- | -------- | ---------- | ----------- | ----------- | ------------ | ------------------- |
| chai1_msa       | **+**ColabFold MSA (default: none) | 0.562   | 0.533    | 43.89      | 24.996      | 3.29        | 141          | **−0.236** (chai1 0.798) |
| boltz2_nomsa    | **−**MSA, single-sequence (default: MSA) | 0.421   | 0.387    | 27.12      | 29.835      | 1.31        | 67           | −0.442 (boltz2 0.863) |
| colabfold_nomsa | **−**MSA, single-sequence (default: MSA) | 0.307   | 0.290    | 16.89      | 32.431      | 2.07        | 128          | −0.569 (colabfold 0.876) |
| openfold_nomsa  | **−**MSA, single-sequence (default: MSA) | 0.307   | 0.288    | 16.92      | 32.839      | 1.92        | 70           | −0.568 (openfold 0.875) |

All 45/45 targets scored for every variant (mean **284 aligned Cα** per target, range 70–632; identical across variants since they share the target sequences). **GDT_TS** is on a 0–100 scale and is computed with the TMscore binary on the sequence-matched Cα atoms (`external_tmscore_matched`); for Cα-RMSD lower is better. **CO₂/job** is CodeCarbon world-average emissions per target (`total_carbon_with_shared_msa_g`) and **Time/job** is wall-clock per target — the three `*_nomsa` variants build no MSA, and `chai1_msa` reuses the shared ColabFold MSA and is not re-charged its build cost, so its carbon is inference-only. Per-protein scores, carbon, and runtime are in `results/benchmark-score.csv`, `results/benchmark_scores_all_models.csv`, and `results/benchmark_model_summary_all_models.csv` alongside the eight base models, and in each variant's JSON.

**Removing the MSA (rows 2–4)** collapses the three MSA-dependent models, as
expected: ColabFold and OpenFold both run AlphaFold2 Evoformer weights, which
lose almost all accuracy without an alignment (lDDT-Cα ≈ 0.31), and the Boltz-2
diffusion trunk drops from 0.863 to 0.421. These confirm how much of each
model's accuracy is carried by the MSA.

**Adding the MSA to Chai-1 (row 1) is the notable result: it makes Chai-1 worse
on average** (0.562 vs 0.798 single-sequence). The effect is strongly bimodal —
the MSA *improves* 10/45 targets (e.g. T1145 +0.32, T1159 +0.30) but *degrades*
22/45, several to near-unfolded structures (T1185s2 −0.72, T1185s4 −0.68,
T1272s4 −0.67), concentrated on the multi-domain CASP subdomain targets
(T1137sX / T1272sX / T1114sX / T1185sX).

This is **genuine Chai-1 behavior, not an artifact of our ColabFold A3M.** We
verified it by re-folding all 45 targets with Chai-1's *own* online MSA server
(properly per-database source-tagged and species-paired): the native-MSA mean is
**0.563 vs our ColabFold-A3M 0.562** (mean |Δ| = 0.005, max 0.025 lDDT-Cα per
target), collapsing on the same targets. So `chai1_msa` reflects the model, not
our MSA pipeline. It is consistent with reported "MSA can hurt AF3-style models"
behavior — Chai-1's maintainers acknowledge predictions can be worse with MSAs
that are "not consistent with the model's expectations"
([chai-lab discussion #277](https://github.com/chaidiscovery/chai-lab/discussions/277)),
and a related effect is reported for Boltz
([issue #627](https://github.com/jwohlwend/boltz/issues/627)) — though those
reports concern irrelevant MSAs on designed/orphan proteins, whereas here the
degradation is on natural targets with deep, relevant alignments.

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

Each of the eight model JSONs above carries, in addition to the single-run blocks (`aggregate_summary`, `per_protein_scores`, `per_prediction_scores`, `per_protein_runtime_metadata`, `per_protein_status`):

- `n_replicates` and `primary_replicate` — how many replicates the file covers, and which one the per-protein blocks come from (`rep1`).
- `replicates[]` — one entry per replicate with its `source_result_dir`, full `aggregate_summary`, and `cost_totals`. The latter holds the manifest cost columns (`total_runtime_sec`, `inference_runtime_sec`, `msa_build_runtime_sec`, energy and CO₂) plus the CodeCarbon device breakdown: `cpu_energy_kwh`, `gpu_energy_kwh`, `ram_energy_kwh`, `gpu_share_pct`, and `codecarbon_wall_time_sec` (CodeCarbon's own stage timer, independent of the manifest runtimes; inference-only for models that consume rather than build the shared MSA).
- `aggregate_summary_across_reps` and `cost_totals_across_reps` — per metric, `{mean, std, n_reps, values}`, where `std` is the sample standard deviation (ddof = 1) and `values` keeps the three individual replicate numbers.
- `cost_totals_across_reps_comparable` and `cost_totals_across_reps_note` — whether a mean over the cost figures is interpretable at all. It is `false` exactly when a model's `stages_run` differ between replicates, which today means **af2**: it builds its MSA in replicate 1 and reuses `features.pkl` afterwards, so averaging gives 8.78 ± 12.24 h — a mixture of two experiments rather than a distribution. When it is `false`, read the per-replicate values in `replicates[]` and compare like stages only. Accuracy metrics are unaffected and stay comparable for every model.

The four MSA cross-mode variant JSONs below are from a single run and carry no replicate blocks.

MSA cross-mode variants (see the section above; separate 2026-07-09 run):

- `results/chai1_msa.json`
- `results/boltz2_nomsa.json`
- `results/colabfold_nomsa.json`
- `results/openfold_nomsa.json`

## Reproduction

The 45-target benchmark is assembled from five source runs (three original GPU runs plus two GPU reruns for the three CPU-executed models) and scored with the combined-run scripts. From the `Protein-Folding-Benchmark` repo root:

**1. Combine predictions from the five source runs:**
```bash
conda run -n folding-benchmark python scripts/combine_8model_predictions.py \
  --dest results/2026-06-09-combine-8models-gpu
```

**2. Build per-target reference PDBs (correct chain + residue range from manifest):**
```bash
conda run -n folding-benchmark python scripts/build_combined_references.py \
  --run-dir results/2026-06-09-combine-8models-gpu
```

**3. Score all 45 targets × 8 models:**
```bash
conda run -n folding-benchmark python scripts/score_combined_8models.py \
  --run-dir results/2026-06-09-combine-8models-gpu \
  --match-mode sequence
```

**4. Export to carbon4science format:**
```bash
conda run -n folding-benchmark python scripts/export_combined_to_carbon4science.py \
  --run-dir results/2026-06-09-combine-8models-gpu \
  --out-dir ../carbon4science.github.io/results
```

Source runs:

| Group | Result directory | Models |
|---|---|---|
| MSA-free (GPU rerun 2026-06-09) | `results/20260609_gpu_rerun_msa-free` | chai1, esmfold |
| MSA-free (original) | `results/20260601_104732_casp15_casp16_unique_lt1000_all_default_msa-free` | omegafold |
| ColabFold MSA (GPU rerun 2026-06-09) | `results/20260609_gpu_rerun_colabfold` | boltz2 |
| ColabFold MSA (original) | `results/20260603_142659_casp15_casp16_unique_lt1000_all_default_colabfold` | colabfold, openfold, protenix |
| AlphaFold2 | `results/20260604_134750_casp15_casp16_unique_lt1000_all_default-af2` | af2 |

## Limitations

- This is a 45-target local benchmark, not the full CASP15/CASP16 benchmark universe.
- Model outputs are local-run artifacts from one workstation and should be interpreted with that hardware and software context.
- Carbon estimates use CodeCarbon offline world-average accounting, not direct facility metering.
- References are extracted per-target from mmCIF files using the manifest's `chain_id` and `residue_start`/`residue_end`; results are not directly comparable to benchmarks using different reference extraction strategies.
- Training-data cutoffs are documented in the `## Training-data cutoffs` section. Boltz-2 (cutoff 2023-06-01) postdates the CASP15 targets (33/45) and may have trained on structures overlapping that target set. OmegaFold's cutoff is not publicly documented.
