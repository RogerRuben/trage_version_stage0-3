"""Recompute stable order-level metrics from saved Deep v3 link predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage2_deep_v3_utils import LINK_TARGETS, order_level_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for fold_root in sorted(args.prediction_root.glob("fold=*")):
        fold = int(fold_root.name.split("=", 1)[-1])
        for split in ["validation", "test"]:
            path = fold_root / f"{split}_predictions.parquet"
            if not path.exists():
                continue
            frame = pd.read_parquet(path)
            for target in LINK_TARGETS:
                rows.append({"fold": fold, "split": split, "target": target.upper(), **order_level_metrics(frame, target)})
    result = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    summary_columns = [
        "order_q90_auc", "order_q90_ap", "order_q90_spearman",
        "order_q90_lift_top5", "order_q90_lift_top10",
    ]
    summary = result.groupby(["split", "target"], as_index=False)[summary_columns].mean()
    summary.to_csv(args.output_csv.with_name(args.output_csv.stem + "_summary.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
