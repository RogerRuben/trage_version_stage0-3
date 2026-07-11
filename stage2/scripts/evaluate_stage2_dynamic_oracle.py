"""Evaluate strictly lagged state as an oracle-route predictability diagnostic."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from estimate_stage2_predictability_ceiling import top_metrics  # noqa: E402


TARGETS = ["lcs", "rts", "pmis"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--causal-root", type=Path, default=Path("stage2/output/causal_features"))
    parser.add_argument("--output", type=Path, default=Path("stage2/output/predictability_ceiling/dynamic_oracle_metrics.csv"))
    return parser.parse_args()


def load(root: Path, split: str) -> pd.DataFrame:
    features = pd.read_parquet(root / f"stage2_oracle_lagged_features_{split}.parquet")
    labels = pd.read_parquet(root / "audit_labels" / f"{split}_dynamic_oracle_labels.parquet")
    keys = ["order_id", "date", "link_id", "link_seq"]
    return features.merge(labels[keys + [f"{target}_raw" for target in TARGETS]], on=keys, how="inner", validate="one_to_one")


def main() -> None:
    args = parse_args()
    train = load(args.causal_root, "train")
    evaluation = pd.concat([load(args.causal_root, "validation"), load(args.causal_root, "test")], ignore_index=True)
    rows = []
    for target in TARGETS:
        label = f"{target}_raw"
        threshold = float(train[label].dropna().quantile(0.90))
        candidates = [
            column for column in evaluation.columns
            if column.endswith(tuple(f"{target}_raw_{window}m" for window in [5, 15, 30, 60]))
        ]
        for feature in candidates:
            valid = evaluation[label].notna() & evaluation[feature].notna()
            if valid.sum() < 100:
                continue
            metrics = top_metrics(
                evaluation.loc[valid, label].to_numpy(dtype=float),
                evaluation.loc[valid, feature].to_numpy(dtype=float),
                threshold,
            )
            rows.append({
                "label_scale": "raw", "target": label, "oracle": feature,
                "experiment_track": "oracle_route_upper_bound", "strictly_lagged": True,
                **metrics,
            })
    output = pd.DataFrame(rows).sort_values(["target", "ap"], ascending=[True, False]) if rows else pd.DataFrame()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(output.groupby("target", as_index=False).first().to_string(index=False) if not output.empty else "no dynamic oracle rows")


if __name__ == "__main__":
    main()

