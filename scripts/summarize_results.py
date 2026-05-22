#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (row.get("target_id", ""), row.get("pdb_id", ""), row.get("chain_id", ""), row.get("model", ""), row.get("rank", ""))


def status_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (row.get("target_id", ""), row.get("pdb_id", ""), row.get("chain", row.get("chain_id", "")), row.get("model", ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact Protein Folding Carbon4Science summary CSV.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output", default="results/protein_folding_six_models_first5_merged.csv")
    args = parser.parse_args()

    root = Path(args.root)
    results = root / "results"
    metadata = {key(row): row for row in read_csv(results / "protein_folding_six_models_first5_run_metadata.csv")}
    status = {status_key(row): row.get("status", "") for row in read_csv(results / "protein_folding_six_models_first5_run_status.csv")}
    scores = read_csv(results / "protein_folding_six_models_first5_scores.csv")

    columns = [
        "target_id", "pdb_id", "chain", "model", "status", "inference_time_sec",
        "carbon_emissions_g", "carbon_energy_consumed_kwh", "lddt_ca", "ca_rmsd",
        "tmalign_tm_score_ref", "tmalign_rmsd", "top_k", "msa_mode", "benchmark_notes",
    ]
    rows = []
    for score in scores:
        merged_key = key(score)
        meta = metadata.get(merged_key, {})
        skey = (score.get("target_id", ""), score.get("pdb_id", ""), score.get("chain_id", ""), score.get("model", ""))
        model = score.get("model", "")
        rows.append({
            "target_id": score.get("target_id", ""),
            "pdb_id": score.get("pdb_id", ""),
            "chain": score.get("chain_id", ""),
            "model": model,
            "status": status.get(skey, ""),
            "inference_time_sec": meta.get("inference_time_sec", ""),
            "carbon_emissions_g": meta.get("carbon_emissions_g", ""),
            "carbon_energy_consumed_kwh": meta.get("carbon_energy_consumed_kwh", ""),
            "lddt_ca": score.get("lddt_ca", ""),
            "ca_rmsd": score.get("ca_rmsd", ""),
            "tmalign_tm_score_ref": score.get("tmalign_tm_score_ref", ""),
            "tmalign_rmsd": score.get("tmalign_rmsd", ""),
            "top_k": "1",
            "msa_mode": "single_sequence" if model in {"colabfold", "openfold"} else "single_sequence/native",
            "benchmark_notes": "First-five CASP target smoke benchmark; top_k=1; CodeCarbon offline CHE.",
        })

    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
