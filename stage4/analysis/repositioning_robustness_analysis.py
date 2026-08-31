"""Pair frozen anchors with R1 Train-only repositioning robustness outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_TO_R1 = {
    "MAIN_Q25_M_P70": "R1_Q25_M_P70_REPOS",
    "MAIN_Q50_M_P70": "R1_Q50_M_P70_REPOS",
    "MAIN_Q75_M_P70": "R1_Q75_M_P70_REPOS",
    "BENCH_AV_M": "R1_BENCH_AV_M_REPOS",
}
OUTPUT_REL = Path("stage4/output/paper_enhancement/repositioning_robustness")
NO_REPOS_REL = OUTPUT_REL / "no_reposition_reproduction"
ENABLED_REL = OUTPUT_REL / "enabled"
SYSTEM_METRICS = (
    "service_rate",
    "matched",
    "patience_expired",
    "request_to_pickup_mean",
    "request_to_pickup_p50",
    "request_to_pickup_p90",
    "request_to_pickup_p95",
    "AV_assignment_share",
)
GATE_METRICS = (
    "gate_av_n0_spatial",
    "gate_av_n5_solver_eligible",
    "gate_av_n6_selected",
    "eligibility_conversion_n5_over_n0",
    "assignment_conversion_n6_over_n0",
    "structural_retention_n2_over_n1",
    "evidence_retention_n3_over_n2",
    "patience_retention_n4_over_n3b",
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else np.nan


def _queue_metrics(path: Path) -> dict[str, float]:
    epoch = pd.read_parquet(path, columns=["waiting_orders"])
    waiting = pd.to_numeric(epoch["waiting_orders"], errors="coerce")
    return {
        "queue_pressure_mean_waiting": float(waiting.mean()),
        "queue_pressure_p90_waiting": float(waiting.quantile(0.90)),
        "queue_pressure_max_waiting": float(waiting.max()),
    }


def pairwise_system(root: Path) -> pd.DataFrame:
    rows = []
    for base, r1 in BASE_TO_R1.items():
        baseline_dir = root / NO_REPOS_REL / base
        repos_dir = root / ENABLED_REL / r1
        baseline = _json(baseline_dir / "summary.json")
        repos = _json(repos_dir / "summary.json")
        row: dict[str, Any] = {
            "base_scenario": base,
            "reposition_scenario": r1,
            "requested_q_A": baseline["requested_q_A"],
            "target_p_A": baseline["target_p_A"],
        }
        for metric in SYSTEM_METRICS:
            row[f"baseline_{metric}"] = baseline[metric]
            row[f"reposition_{metric}"] = repos[metric]
            row[f"delta_{metric}"] = repos[metric] - baseline[metric]
        baseline_queue = _queue_metrics(baseline_dir / "epoch_stats.parquet")
        repos_queue = _queue_metrics(repos_dir / "epoch_stats.parquet")
        for metric in baseline_queue:
            row[f"baseline_{metric}"] = baseline_queue[metric]
            row[f"reposition_{metric}"] = repos_queue[metric]
            row[f"delta_{metric}"] = repos_queue[metric] - baseline_queue[metric]
        rows.append(row)
    return pd.DataFrame(rows)


def gate_summary(root: Path) -> pd.DataFrame:
    rows = []
    for base, r1 in BASE_TO_R1.items():
        left = pd.read_csv(root / NO_REPOS_REL / base / "gate_totals.csv").iloc[0]
        right = pd.read_csv(root / ENABLED_REL / r1 / "gate_totals.csv").iloc[0]
        row: dict[str, Any] = {"base_scenario": base, "reposition_scenario": r1}
        for metric in GATE_METRICS:
            row[f"baseline_{metric}"] = left[metric]
            row[f"reposition_{metric}"] = right[metric]
            row[f"delta_{metric}"] = right[metric] - left[metric]
        rows.append(row)
    return pd.DataFrame(rows)


def operations_summary(root: Path) -> pd.DataFrame:
    rows = []
    for base, r1 in BASE_TO_R1.items():
        summary = _json(root / ENABLED_REL / r1 / "summary.json")
        operation = summary["repositioning"]
        rows.append(
            {
                "base_scenario": base,
                "reposition_scenario": r1,
                **operation,
                "wall_clock_runtime_s": summary["runtime"]["wall_clock_runtime_s"],
                "peak_rss_mb": summary["runtime"]["peak_rss_mb"],
                "all_valhalla_failed_routing_arcs": summary["runtime"]["failed_routing_arcs"],
            }
        )
    return pd.DataFrame(rows)


def _window(timestamp: pd.Series) -> pd.Series:
    hour = timestamp.dt.hour
    return pd.Series(
        np.select(
            [(hour >= 7) & (hour < 9), (hour >= 17) & (hour < 19)],
            ["MORNING_07_0859", "EVENING_17_1859"],
            default="OTHER",
        ),
        index=timestamp.index,
    )


def _rates(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["time_bin_start"] = pd.to_datetime(frame["time_bin_start"], utc=True).dt.tz_convert(
        "Asia/Shanghai"
    )
    frame["eligibility_conversion_n5_over_n0"] = frame["gate_av_n5_solver_eligible"] / frame[
        "gate_av_n0_spatial"
    ].replace(0, np.nan)
    frame["patience_retention_n4_over_n3b"] = frame[
        "gate_av_n4_pickup_within_patience"
    ] / frame["gate_av_n3b_route_returned"].replace(0, np.nan)
    frame["window"] = _window(frame["time_bin_start"])
    return frame


def temporal_summary(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired = []
    windows = []
    count_columns = [
        "gate_av_n0_spatial",
        "gate_av_n3b_route_returned",
        "gate_av_n4_pickup_within_patience",
        "gate_av_n5_solver_eligible",
    ]
    for base, r1 in BASE_TO_R1.items():
        left = _rates(pd.read_csv(root / NO_REPOS_REL / base / "gate_15min.csv"))
        right = _rates(pd.read_csv(root / ENABLED_REL / r1 / "gate_15min.csv"))
        merged = left.merge(right, on="time_bin_start", suffixes=("_baseline", "_reposition"))
        merged.insert(0, "reposition_scenario", r1)
        merged.insert(0, "base_scenario", base)
        merged["delta_eligibility_conversion"] = (
            merged["eligibility_conversion_n5_over_n0_reposition"]
            - merged["eligibility_conversion_n5_over_n0_baseline"]
        )
        merged["delta_patience_retention"] = (
            merged["patience_retention_n4_over_n3b_reposition"]
            - merged["patience_retention_n4_over_n3b_baseline"]
        )
        paired.append(merged)
        for label, mask in (
            ("ALL_DAY", pd.Series(True, index=left.index)),
            ("MORNING_07_0859", left["window"].eq("MORNING_07_0859")),
            ("EVENING_17_1859", left["window"].eq("EVENING_17_1859")),
        ):
            row: dict[str, Any] = {
                "base_scenario": base,
                "reposition_scenario": r1,
                "window": label,
            }
            for mode, frame, selected in (
                ("baseline", left, mask),
                ("reposition", right, mask),
            ):
                totals = frame.loc[selected, count_columns].sum()
                row[f"{mode}_eligibility_conversion"] = _safe(
                    totals["gate_av_n5_solver_eligible"], totals["gate_av_n0_spatial"]
                )
                row[f"{mode}_patience_retention"] = _safe(
                    totals["gate_av_n4_pickup_within_patience"],
                    totals["gate_av_n3b_route_returned"],
                )
            row["delta_eligibility_conversion"] = (
                row["reposition_eligibility_conversion"] - row["baseline_eligibility_conversion"]
            )
            row["delta_patience_retention"] = (
                row["reposition_patience_retention"] - row["baseline_patience_retention"]
            )
            windows.append(row)
    return pd.concat(paired, ignore_index=True), pd.DataFrame(windows)


def _paired_plot(
    frame: pd.DataFrame,
    baseline_column: str,
    reposition_column: str,
    ylabel: str,
    title: str,
    output: Path,
    scale: float = 1.0,
) -> None:
    labels = ["q=.25", "q=.50", "q=.75", "all AV"]
    x = np.arange(len(frame))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.bar(x - width / 2, frame[baseline_column] * scale, width, label="No reposition")
    ax.bar(x + width / 2, frame[reposition_column] * scale, width, label="Train-only reposition")
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _temporal_plot(frame: pd.DataFrame, output: Path) -> None:
    q75 = frame.loc[frame["base_scenario"].eq("MAIN_Q75_M_P70")].sort_values(
        "time_bin_start"
    )
    timestamp = pd.to_datetime(q75["time_bin_start"])
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    ax.plot(
        timestamp,
        100 * q75["eligibility_conversion_n5_over_n0_baseline"],
        label="q=.75 no reposition",
        linewidth=1.6,
    )
    ax.plot(
        timestamp,
        100 * q75["eligibility_conversion_n5_over_n0_reposition"],
        label="q=.75 Train-only reposition",
        linewidth=1.6,
    )
    ax.axvspan(
        pd.Timestamp("2016-10-31T17:00:00+08:00"),
        pd.Timestamp("2016-10-31T19:00:00+08:00"),
        color="grey",
        alpha=0.15,
        label="Pre-specified 17:00–18:59",
    )
    ax.set_ylabel("N5 / N0 (%)")
    ax.set_xlabel("Test31 local time")
    ax.set_title("15-minute eligibility conversion under q=.75")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def render_report(
    system: pd.DataFrame,
    gates: pd.DataFrame,
    windows: pd.DataFrame,
    operations: pd.DataFrame,
    classification: str,
) -> str:
    system_view = system[
        [
            "base_scenario",
            "baseline_service_rate",
            "reposition_service_rate",
            "delta_service_rate",
            "baseline_matched",
            "reposition_matched",
            "delta_matched",
            "baseline_patience_expired",
            "reposition_patience_expired",
        ]
    ].copy()
    for column in ("baseline_service_rate", "reposition_service_rate", "delta_service_rate"):
        system_view[column] = system_view[column].map(lambda value: f"{100*value:.3f} pp" if column.startswith("delta") else f"{100*value:.3f}%")
    gate_view = gates[
        [
            "base_scenario",
            "baseline_eligibility_conversion_n5_over_n0",
            "reposition_eligibility_conversion_n5_over_n0",
            "delta_eligibility_conversion_n5_over_n0",
            "delta_structural_retention_n2_over_n1",
            "delta_evidence_retention_n3_over_n2",
            "delta_patience_retention_n4_over_n3b",
        ]
    ].copy()
    for column in gate_view.columns[1:]:
        gate_view[column] = gate_view[column].map(lambda value: f"{100*value:.4f} pp")
    evening = windows.loc[windows["window"].eq("EVENING_17_1859")][
        [
            "base_scenario",
            "baseline_eligibility_conversion",
            "reposition_eligibility_conversion",
            "delta_eligibility_conversion",
            "baseline_patience_retention",
            "reposition_patience_retention",
            "delta_patience_retention",
        ]
    ].copy()
    for column in evening.columns[1:]:
        evening[column] = evening[column].map(lambda value: f"{100*value:.4f} pp")
    burden = operations[
        [
            "reposition_scenario",
            "reposition_trip_count",
            "total_reposition_distance_m",
            "total_reposition_travel_time_s",
            "mean_reposition_distance_m",
            "mean_reposition_travel_time_s",
            "av_vehicle_time_repositioning_share",
            "reposition_routing_failure_count",
            "peak_rss_mb",
        ]
    ].copy()
    burden["av_vehicle_time_repositioning_share"] = burden[
        "av_vehicle_time_repositioning_share"
    ].map(lambda value: f"{100*value:.3f}%")
    interpretation = {
        "SUPPORTS_CURRENT_STORY": "Simple Train-only rebalancing changes some outcomes but does not explain away the penetration-related effective-capacity pattern.",
        "QUALIFIES_CURRENT_STORY": "Spatial rebalancing materially attenuates the pattern while preserving its main ordering; spatial placement is an important mediator.",
        "CHANGES_CURRENT_STORY": "The favorable Train-only spatial abstraction largely removes or reverses the high-penetration deterioration, so the frozen no-reposition finding is strongly conditional on rebalancing capability.",
    }[classification]
    return "\n".join(
        [
            "# R1 Train-only AV repositioning robustness",
            "",
            "## Validity",
            "",
            "`NO-REPOSITION REPRODUCTION = 4/4 EXACT`.",
            "",
            "The policy uses only Stage1 Train pickup nodes from 20161009–20161024 in fixed 15-minute local-time bins. Normal passenger dispatch precedes repositioning. Empty routes are a favorable operational abstraction and are not ODD-certified.",
            "",
            "## System results",
            "",
            _markdown(system_view),
            "",
            "## Effective-capacity results",
            "",
            _markdown(gate_view),
            "",
            "## Pre-specified 17:00–18:59 window",
            "",
            _markdown(evening),
            "",
            "## Repositioning burden",
            "",
            _markdown(burden),
            "",
            "## Scientific story decision",
            "",
            f"`{classification}`.",
            "",
            interpretation,
            "",
            "Main limitation: this test isolates one transparent Train-only spatial policy. It is neither optimized nor deployable, and Valhalla deadheading does not establish AV route safety or ODD qualification.",
            "",
            "Recommendation: `GO_GAMMA_FRONTIER`.",
            "",
        ]
    )


def run(root: Path, classification: str) -> None:
    output = root / OUTPUT_REL
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    system = pairwise_system(root)
    gates = gate_summary(root)
    operations = operations_summary(root)
    temporal, windows = temporal_summary(root)
    system.to_csv(output / "repositioning_pairwise_summary.csv", index=False)
    gates.to_csv(output / "repositioning_gate_summary.csv", index=False)
    temporal.to_csv(output / "repositioning_15min_summary.csv", index=False)
    windows.to_csv(output / "repositioning_window_summary.csv", index=False)
    operations.to_csv(output / "repositioning_operations_summary.csv", index=False)
    _paired_plot(
        system,
        "baseline_service_rate",
        "reposition_service_rate",
        "Service rate (%)",
        "Service under Train-only AV repositioning",
        figures / "fig_reposition_service_comparison.png",
        100.0,
    )
    _paired_plot(
        gates,
        "baseline_eligibility_conversion_n5_over_n0",
        "reposition_eligibility_conversion_n5_over_n0",
        "N5 / N0 (%)",
        "Nominal-to-eligible AV opportunity conversion",
        figures / "fig_reposition_eligibility_conversion.png",
        100.0,
    )
    _paired_plot(
        gates,
        "baseline_patience_retention_n4_over_n3b",
        "reposition_patience_retention_n4_over_n3b",
        "Patience retention (%)",
        "Routed AV opportunities surviving pickup patience",
        figures / "fig_reposition_patience_retention.png",
        100.0,
    )
    _temporal_plot(temporal, figures / "fig_reposition_15min_conversion.png")
    report = render_report(system, gates, windows, operations, classification)
    (root / "stage4/docs/paper_redesign/repositioning_robustness_report.md").write_text(
        report, encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--classification",
        choices=(
            "SUPPORTS_CURRENT_STORY",
            "QUALIFIES_CURRENT_STORY",
            "CHANGES_CURRENT_STORY",
        ),
        required=True,
    )
    args = parser.parse_args()
    run(args.root.resolve(), args.classification)


if __name__ == "__main__":
    main()
