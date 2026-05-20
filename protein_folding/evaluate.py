#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

METRICS = ["lddt_ca", "tmalign_tm_score_ref", "ca_rmsd", "carbon_emissions_g", "carbon_energy_consumed_kwh"]
DEFAULT_RESULTS = Path(__file__).resolve().parent / "results/protein_folding_six_models_first5_merged.csv"


def read_rows(path: Path = DEFAULT_RESULTS) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def evaluate(path: Path = DEFAULT_RESULTS) -> list[dict[str, float | str]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in read_rows(path):
        model = row.get("model", "")
        for metric in METRICS:
            value = row.get(metric, "")
            if value == "":
                continue
            try:
                grouped[model][metric].append(float(value))
            except ValueError:
                pass
    return [
        {"model": model, **{metric: mean(values.get(metric, [])) for metric in METRICS}}
        for model, values in sorted(grouped.items())
    ]


if __name__ == "__main__":
    rows = evaluate()
    print("model," + ",".join(METRICS))
    for row in rows:
        print(row["model"] + "," + ",".join(f"{row[m]:.6g}" for m in METRICS))
