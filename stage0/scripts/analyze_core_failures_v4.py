"""Decompose Stage0 v4 strict-Core failures under frozen diagnostic rules."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CONDITIONS = {
    "direction_continuous": ("core_direction_continuous", "hard"),
    "no_unreasonable_detour": ("core_no_unreasonable_detour", "hard"),
    "fallback_share": ("core_fallback_share_ok", "soft"),
    "p90_projection_distance": ("core_projection_ok", "soft"),
    "route_length_ratio": ("core_route_length_ratio_ok", "soft"),
    "interpolated_distance_share": ("core_interpolation_ok", "soft"),
    "origin_endpoint_error": ("core_origin_error_ok", "hard"),
    "destination_endpoint_error": ("core_destination_error_ok", "hard"),
    "match_confidence": ("core_confidence_ok", "soft"),
    "u_turn": ("core_u_turn_ok", "hard"),
    "repeated_link_share": ("core_repeated_link_ok", "soft"),
    "minimum_route_links": ("core_route_link_count_ok", "hard"),
}


def breakdown(frame: pd.DataFrame, date: str) -> list[dict]:
    condition_columns = [column for column, _ in CONDITIONS.values()]
    flags = frame[condition_columns].fillna(False).astype(bool)
    failed_count = (~flags).sum(axis=1)
    strict = flags.all(axis=1)
    hard_columns = [column for column, kind in CONDITIONS.values() if kind == "hard"]
    analysis = flags[hard_columns].all(axis=1)
    rows = []
    for name, (column, kind) in CONDITIONS.items():
        failed = ~flags[column]
        rows.append({
            "date": date,
            "condition_name": name,
            "condition_type": kind,
            "orders": int(len(frame)),
            "single_condition_failure_orders": int(failed.sum()),
            "single_condition_failure_rate": float(failed.mean()),
            "unique_failure_orders": int((failed & failed_count.eq(1)).sum()),
            "joint_failure_orders": int((failed & failed_count.gt(1)).sum()),
            "strict_core_candidates": int(strict.sum()),
            "analysis_set_candidates": int(analysis.sum()),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frames, rows = [], []
    for path in args.quality:
        frame = pd.read_parquet(path)
        date = str(frame.date.iloc[0]) if "date" in frame else path.stem.split("=")[-1]
        frames.append(frame)
        rows.extend(breakdown(frame, date))
    rows.extend(breakdown(pd.concat(frames, ignore_index=True), "ALL"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
