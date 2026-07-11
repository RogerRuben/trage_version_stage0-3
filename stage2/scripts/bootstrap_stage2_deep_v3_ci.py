"""Order-block bootstrap confidence intervals for Deep v3 predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stage2_deep_v3_utils import LINK_TARGETS, metric_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-file", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_parquet(args.prediction_file)
    rng = np.random.default_rng(args.seed)
    rows = []
    for target in LINK_TARGETS:
        valid = frame[f"{target}_valid"].fillna(False)
        data = frame.loc[valid].reset_index(drop=True)
        groups = [idx.to_numpy() for _, idx in pd.Series(np.arange(len(data)), index=data["order_id"]).groupby(level=0)]
        values = {"auc": [], "ap": [], "lift_top5": [], "lift_top10": []}
        for _ in range(args.rounds):
            sampled = rng.integers(0, len(groups), len(groups))
            idx = np.concatenate([groups[i] for i in sampled])
            part = data.iloc[idx]
            metrics = metric_dict(
                part[f"target_{target}_raw"].to_numpy(dtype=float),
                part[f"pred_{target}_raw"].to_numpy(dtype=float),
                part[f"pred_{target}_tail_prob"].to_numpy(dtype=float),
                part[f"target_{target}_tail"].to_numpy(dtype=bool),
            )
            for key in values:
                values[key].append(metrics.get(key, np.nan))
        for key, vals in values.items():
            rows.append({
                "target": target.upper(),
                "metric": key,
                "ci_low": float(np.nanquantile(vals, 0.025)),
                "ci_high": float(np.nanquantile(vals, 0.975)),
                "rounds": args.rounds,
                "block": "order_id",
            })
    result = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    (args.output_csv.with_suffix(".json")).write_text(json.dumps({"prediction_file": str(args.prediction_file), "rounds": args.rounds}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
