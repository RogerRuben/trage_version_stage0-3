"""Placeholder entry for the Deep v3 RTS correlation extension.

The RTS stress-correlation branch is intentionally not part of the first main
RC-MSTNet run.  This script records that decision and creates a reproducible
manifest so later work can add a train-fold-only correlation graph without
touching the main Model B/C protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/deep_v3/ablations/rts_correlation_extension"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": "RTS correlation extension",
        "status": "deferred",
        "reason": "Model D is an optional target-specific ablation. Build only after RC-MSTNet Model B/C feasibility is stable.",
        "required_constraints": [
            "correlation graph built from train fold only",
            "no validation/test leakage",
            "compare only as RTS residual or hybrid branch",
        ],
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
