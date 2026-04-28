#!/usr/bin/env python3
"""Nequix model-specific relaxation (MBD original protocol).

Matbench Discovery uses FIRE + FrechetCellFilter + fmax=0.02 (a bit tighter
than the common 0.05). Results are written to results/specific/Nequix/ so they
can be compared against the unified-protocol results from run_relaxation.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MLIP.relaxation.run_relaxation import run as _run_relax


PROTOCOL = {
    "name": "specific",
    "optimizer": "FIRE",
    "cell_filter": "FrechetCellFilter",
    "fmax": 0.02,
    "max_steps": 500,
}


def build_config(config_json: Path) -> dict:
    import json

    with open(config_json) as f:
        cfg = json.load(f)
    cfg["protocol"] = PROTOCOL
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Nequix MBD-protocol relaxation")
    parser.add_argument(
        "--config",
        default=str(ROOT / "MLIP" / "relaxation" / "configs" / "wbm_100.json"),
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--variant", default="pretrained", choices=["pretrained", "finetuned"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_carbon", dest="track_carbon", action="store_false", default=True)
    args = parser.parse_args()

    cfg = build_config(Path(args.config))
    _run_relax(
        model="Nequix",
        config=cfg,
        variant=args.variant,
        track_carbon=args.track_carbon,
        checkpoint_path=args.checkpoint,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
