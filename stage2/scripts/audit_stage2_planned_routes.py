"""Compare planned route proxies with the sampled actual-route oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shortest-root", type=Path, default=Path("stage2/output/routes/shortest_path"))
    parser.add_argument("--fastest-root", type=Path, default=Path("stage2/output/routes/historical_fastest_path"))
    parser.add_argument("--actual-root", type=Path, default=Path("stage2/output/routes/actual_route_oracle"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/routes/audit"))
    parser.add_argument("--dates", default="20161009,20161010,20161011,20161012,20161013,20161014,20161015,20161016,20161017,20161018,20161019")
    return parser.parse_args()


def parse_dates(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def order_sets(frame: pd.DataFrame, link: str) -> dict[str, list[str]]:
    frame = frame.sort_values(["order_id", "planned_link_seq" if "planned_link_seq" in frame else "actual_link_seq"])
    return frame.groupby("order_id")[link].agg(list).to_dict()


def sequence_lcs_ratio(left: list[str], right: list[str]) -> float:
    # Exact dynamic LCS is intentionally capped to avoid pathological routes.
    if not left or not right:
        return np.nan
    if len(left) * len(right) > 200_000:
        return np.nan
    previous = np.zeros(len(right) + 1, dtype=np.int32)
    for value in left:
        current = np.zeros(len(right) + 1, dtype=np.int32)
        for j, other in enumerate(right, start=1):
            current[j] = previous[j - 1] + 1 if value == other else max(previous[j], current[j - 1])
        previous = current
    return float(previous[-1] / max(len(right), 1))


def audit_method(date: str, method: str, planned: pd.DataFrame, actual: pd.DataFrame) -> dict:
    planned_routes = order_sets(planned, "planned_link_id")
    actual_routes = order_sets(actual, "actual_link_id")
    common = sorted(set(planned_routes) & set(actual_routes))
    records = []
    for order in common:
        p = planned_routes[order]
        a = actual_routes[order]
        ps, aset = set(p), set(a)
        overlap = len(ps & aset)
        records.append({
            "order_id": order,
            "planned_actual_jaccard": overlap / max(len(ps | aset), 1),
            "planned_precision": overlap / max(len(ps), 1),
            "actual_link_coverage": overlap / max(len(aset), 1),
            "ordered_lcs_actual_coverage": sequence_lcs_ratio(p, a),
            "planned_link_count": len(p), "actual_link_count": len(a),
        })
    detail = pd.DataFrame(records)
    return {
        "date": date, "route_source": method,
        "planned_orders": int(planned["order_id"].nunique()),
        "actual_orders": int(actual["order_id"].nunique()),
        "common_orders": len(common),
        "planned_link_count_p50": float(planned.groupby("order_id").size().median()),
        "planned_link_count_p90": float(planned.groupby("order_id").size().quantile(.90)),
        "estimated_entry_time_coverage": float(planned["estimated_link_entry_time"].notna().mean()),
        "realized_label_link_ratio": float(planned.get("realized_label_available", pd.Series(False, index=planned.index)).mean()),
        "planned_actual_jaccard_p50": float(detail["planned_actual_jaccard"].median()) if len(detail) else None,
        "planned_actual_jaccard_p90": float(detail["planned_actual_jaccard"].quantile(.90)) if len(detail) else None,
        "actual_link_coverage_p50": float(detail["actual_link_coverage"].median()) if len(detail) else None,
        "ordered_lcs_actual_coverage_p50": float(detail["ordered_lcs_actual_coverage"].median()) if len(detail) else None,
    }


def label_coverage_slices(date: str, method: str, planned: pd.DataFrame) -> list[dict]:
    frame = planned.copy()
    frame["label_available"] = frame.get("realized_label_available", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    frame["route_position_bucket"] = pd.cut(
        frame["position_ratio"], [-0.001, 1 / 3, 2 / 3, 1.001], labels=["early", "middle", "late"],
    )
    rows = []
    for dimension, column in [
        ("routing_fallback", "routing_fallback"),
        ("route_position", "route_position_bucket"),
        ("road_class", "road_class"),
    ]:
        if column not in frame:
            continue
        for value, group in frame.groupby(column, observed=True, dropna=False):
            rows.append({
                "date": date, "route_source": method, "slice_dimension": dimension,
                "slice_value": str(value), "rows": len(group),
                "realized_label_link_ratio": float(group["label_available"].mean()),
            })
    return rows


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows, slice_rows = [], []
    for date in parse_dates(args.dates):
        actual = pd.read_parquet(args.actual_root / f"day={date}.parquet")
        for method, root in [("shortest_path", args.shortest_root), ("historical_fastest_path", args.fastest_root)]:
            planned = pd.read_parquet(root / f"day={date}.parquet")
            if planned.empty or actual.empty:
                continue
            row = audit_method(date, method, planned, actual)
            rows.append(row)
            slice_rows.extend(label_coverage_slices(date, method, planned))
            print(f"route audit {method} day={date} overlap_p50={row['actual_link_coverage_p50']}", flush=True)
    report = pd.DataFrame(rows)
    report.to_csv(args.output_root / "planned_route_audit.csv", index=False)
    pd.DataFrame(slice_rows).to_csv(args.output_root / "planned_route_label_coverage_slices.csv", index=False)
    summary = {
        "methods": report.groupby("route_source").mean(numeric_only=True).reset_index().to_dict("records") if len(report) else [],
        "audit_scope": "sampled common orders; actual route is post-trip oracle only",
        "ordered_overlap_note": "ordered LCS is omitted for pathological route pairs with >200k DP cells",
    }
    (args.output_root / "planned_route_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Planned route audit", "", report.to_markdown(index=False, floatfmt=".4f"), "",
             "Actual-route columns are audit-only and are forbidden from deployable Stage2 features."]
    (args.output_root / "planned_route_audit.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
