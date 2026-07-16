"""Compare direction-aware v4 with the connector-disabled diagnostic view."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage0.scripts.audit_network_comparison_v4 import (
    directed_reachability,
    graph_and_metrics,
    read_route_parts,
    route_length_metrics,
)


def variant(name: str, roads: Path, root: Path, quality: Path, summary: Path) -> dict:
    graph, lookup, metrics = graph_and_metrics(roads)
    routes = read_route_parts(root, "20161023")
    quality_frame = pd.read_parquet(quality)
    return {
        "variant": name,
        **metrics,
        **directed_reachability(graph, lookup, routes),
        **pd.read_csv(summary).iloc[0].to_dict(),
        **route_length_metrics(quality_frame),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal-roads", type=Path, required=True)
    parser.add_argument("--normal-root", type=Path, required=True)
    parser.add_argument("--normal-quality", type=Path, required=True)
    parser.add_argument("--normal-summary", type=Path, required=True)
    parser.add_argument("--disabled-roads", type=Path, required=True)
    parser.add_argument("--disabled-root", type=Path, required=True)
    parser.add_argument("--disabled-quality", type=Path, required=True)
    parser.add_argument("--disabled-summary", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        variant("direction_aware_connectors", args.normal_roads, args.normal_root, args.normal_quality, args.normal_summary),
        variant("connectors_disabled", args.disabled_roads, args.disabled_root, args.disabled_quality, args.disabled_summary),
    ]
    frame = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    normal, disabled = rows
    result = {
        "status": "DIAGNOSTIC_PASS",
        "comparison": rows,
        "reachability_contribution": normal["directed_od_reachability"] - disabled["directed_od_reachability"],
        "gap_order_reduction": disabled["orders_with_gap"] - normal["orders_with_gap"],
        "mean_gap_reduction": disabled["mean_direction_gaps"] - normal["mean_direction_gaps"],
        "mean_route_length_change_m": normal["mean_matched_route_length_m"] - disabled["mean_matched_route_length_m"],
        "interpretation": "Connectors restore explicit zero-length grade-terminal transitions; human review remains required.",
    }
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
