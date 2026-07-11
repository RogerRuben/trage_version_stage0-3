"""Evaluate compact daily Stage0 products before expanding the experiment gate."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--comparison-collection", default="matcher_comparison")
    parser.add_argument("--link-quality-report", default="link_level_quality_summary.csv")
    parser.add_argument("--matched-point-collection", default="hmm_matched_points")
    parser.add_argument("--route-collection", default="hmm_route_parts")
    parser.add_argument("--link-traversal-collection", default="hmm_link_traversals")
    parser.add_argument("--turn-movement-collection", default="hmm_turn_movements")
    parser.add_argument("--poi-behavior-collection", default="stage0_order_link_poi_behavior")
    return parser.parse_args()


def path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def retained_day_size(root: Path, date: str, args: argparse.Namespace) -> int:
    paths = [
        root / "order_base" / f"day={date}.parquet",
        root / args.link_traversal_collection / f"day={date}",
        root / args.turn_movement_collection / f"day={date}",
        root / args.poi_behavior_collection / f"day={date}",
        root / args.comparison_collection / f"day={date}",
        root / "case_traces" / f"day={date}",
        root / "reports" / "threshold_sensitivity" / f"day={date}.json",
    ]
    return sum(path_size(path) for path in paths)


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    quality_path = root / "reports" / args.link_quality_report
    link_quality = pd.read_csv(quality_path) if quality_path.exists() else pd.DataFrame()
    rows = []
    for day_dir in sorted((root / args.comparison_collection).glob("day=*")):
        date = day_dir.name.split("=", 1)[1]
        comparison_path = day_dir / "order_comparison.parquet"
        if not comparison_path.exists():
            continue
        comparison = pd.read_parquet(comparison_path)
        links = link_quality[link_quality.date.astype(str) == date] if not link_quality.empty else pd.DataFrame()
        observed_ratio = float(links.observed_traversal_ratio.iloc[0]) if len(links) else float("nan")
        observed_length_ratio = float(links.observed_link_length_ratio.iloc[0]) if len(links) else float("nan")
        low_ratio = float(links.low_traversal_quality_ratio.iloc[0]) if len(links) else float("nan")
        artifacts = {
            "link_traversals": root / args.link_traversal_collection / f"day={date}",
            "turn_movements": root / args.turn_movement_collection / f"day={date}",
            "poi_behavior": root / args.poi_behavior_collection / f"day={date}",
            "case_index": root / "case_traces" / f"day={date}" / "case_index.csv",
            "sensitivity": root / "reports" / "threshold_sensitivity" / f"day={date}.json",
        }
        artifact_complete = all(path.exists() for path in artifacts.values())
        fallback_ratio = float(comparison.fallback_used.mean())
        success_column = "matcher_matching_success" if "matcher_matching_success" in comparison.columns else "hmm_matching_success"
        success_ratio = float(comparison[success_column].mean())
        median_p90 = float(comparison.p90_match_dist.median())
        matcher_gap = float(comparison.topology_gap_count.mean())
        geometric_gap = float(comparison.geo_topology_gap_count.mean())
        checks = {
            "artifact_complete": artifact_complete,
            "fallback_ratio_le_20pct": fallback_ratio <= 0.20,
            "matcher_success_ge_85pct": success_ratio >= 0.85,
            "median_order_p90_le_30m": median_p90 <= 30.0,
            "topology_not_degraded": matcher_gap <= geometric_gap + 0.01,
            "observed_link_length_ge_70pct": observed_length_ratio >= 0.70,
            "low_traversal_quality_le_20pct": low_ratio <= 0.20,
        }
        retained_bytes = retained_day_size(root, date, args)
        row = {
            "date": date,
            "orders": len(comparison),
            "retained_bytes": retained_bytes,
            "retained_mb": retained_bytes / (1024 ** 2),
            "fallback_ratio": fallback_ratio,
            "matcher_success_ratio": success_ratio,
            "median_order_p90_match_dist_m": median_p90,
            "mean_matcher_topology_gaps": matcher_gap,
            "mean_geometric_topology_gaps": geometric_gap,
            "observed_traversal_ratio": observed_ratio,
            "observed_link_length_ratio": observed_length_ratio,
            "low_traversal_quality_ratio": low_ratio,
            **checks,
            "stage0_gate": "PASS" if all(checks.values()) else "REVIEW",
        }
        rows.append(row)

    report = pd.DataFrame(rows)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    report.to_csv(reports / "stage0_readiness.csv", index=False)
    lines = ["# Stage0 compact readiness", ""]
    if report.empty:
        lines.append("No completed matcher-comparison days were found.")
    else:
        for row in report.itertuples(index=False):
            lines.append(
                f"- {row.date}: **{row.stage0_gate}**; fallback={row.fallback_ratio:.2%}, "
                f"matcher success={row.matcher_success_ratio:.2%}, median order P90="
                f"{row.median_order_p90_match_dist_m:.2f} m, observed traversal="
                f"{row.observed_traversal_ratio:.2%}, observed route length="
                f"{row.observed_link_length_ratio:.2%}, retained={row.retained_mb:.1f} MB."
            )
    (reports / "stage0_readiness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
