"""Evaluate compact daily Stage0 products before expanding the experiment gate."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    quality_path = root / "reports" / "link_level_quality_summary.csv"
    link_quality = pd.read_csv(quality_path) if quality_path.exists() else pd.DataFrame()
    rows = []
    for day_dir in sorted((root / "matcher_comparison").glob("day=*")):
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
            "hmm_link_traversals": root / "hmm_link_traversals" / f"day={date}",
            "hmm_turn_movements": root / "hmm_turn_movements" / f"day={date}",
            "poi_behavior": root / "stage0_order_link_poi_behavior" / f"day={date}",
            "case_index": root / "case_traces" / f"day={date}" / "case_index.csv",
            "sensitivity": root / "reports" / "threshold_sensitivity" / f"day={date}.json",
        }
        artifact_complete = all(path.exists() for path in artifacts.values())
        fallback_ratio = float(comparison.fallback_used.mean())
        success_ratio = float(comparison.hmm_matching_success.mean())
        median_p90 = float(comparison.p90_match_dist.median())
        hmm_gap = float(comparison.topology_gap_count.mean())
        geo_gap = float(comparison.geo_topology_gap_count.mean())
        checks = {
            "artifact_complete": artifact_complete,
            "fallback_ratio_le_20pct": fallback_ratio <= 0.20,
            "hmm_success_ge_85pct": success_ratio >= 0.85,
            "median_order_p90_le_30m": median_p90 <= 30.0,
            "topology_not_degraded": hmm_gap <= geo_gap + 0.01,
            "observed_link_length_ge_70pct": observed_length_ratio >= 0.70,
            "low_traversal_quality_le_20pct": low_ratio <= 0.20,
        }
        row = {
            "date": date,
            "orders": len(comparison),
            "fallback_ratio": fallback_ratio,
            "hmm_success_ratio": success_ratio,
            "median_order_p90_match_dist_m": median_p90,
            "mean_hmm_topology_gaps": hmm_gap,
            "mean_geo_topology_gaps": geo_gap,
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
                f"HMM success={row.hmm_success_ratio:.2%}, median order P90="
                f"{row.median_order_p90_match_dist_m:.2f} m, observed traversal="
                f"{row.observed_traversal_ratio:.2%}, observed route length="
                f"{row.observed_link_length_ratio:.2%}."
            )
    (reports / "stage0_readiness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
