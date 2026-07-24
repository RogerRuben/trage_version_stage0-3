"""Rebuild the full-day Simulator v3 capability mapping without rebuilding supply.

This entry point evaluates frozen capability scenarios on the existing RT-Base
demand table.  Main-profile thresholds are validated by
``build_capability`` and cannot be derived from the 2016-10-23 test day.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_decoupled_abm_environment import build_capability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demand",
        type=Path,
        default=Path("stage4/data/decoupled_abm/demand_20161023_RT-Base.parquet"),
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path("stage4/config/vehicle_capability_profiles.json"),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("stage3/output/stage4_inputs_final/fold=3/stage4_inputs.parquet"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("stage4/output/decoupled_environment/capability_mapping"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("stage4/docs/results"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demand = pd.read_parquet(args.demand)
    mapping = build_capability(
        demand=demand,
        profiles_path=args.profiles,
        reference_path=args.reference,
        calibration_mode="none",
        output_root=args.output_root,
        results_dir=args.results_dir,
    )
    manifest = json.loads((args.output_root / "manifest.json").read_text(encoding="utf-8"))
    print(json.dumps(manifest | {"mapping_rows": int(len(mapping))}, indent=2))
    if manifest.get("status") != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
