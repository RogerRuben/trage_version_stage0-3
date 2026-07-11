"""Compare estimated-time and actual-entry oracle route-conditioned results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


FULL_ABLATION = "static_rolling_dynamic_topology_route"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimated-eval-root", type=Path, default=Path("stage2/output/route_conditioned_eval/estimated_time"))
    parser.add_argument("--oracle-eval-root", type=Path, default=Path("stage2/output/route_conditioned_eval/oracle_actual_entry"))
    parser.add_argument("--estimated-dataset-root", type=Path, default=Path("stage2/output/route_conditioned_dataset/estimated_time_daily"))
    parser.add_argument("--oracle-dataset-root", type=Path, default=Path("stage2/output/actual_entry_oracle_causal_dataset"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/route_conditioned_eval/gap_audit"))
    return parser.parse_args()


def read_summary(root: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(root / "rolling_fair_summary.csv")
    frame["time_mode"] = label
    return frame


def daily_coverage(root: Path, dates: list[str], mode: str) -> pd.DataFrame:
    rows = []
    for date in dates:
        path = root / f"day={date}.parquet"
        if not path.exists():
            continue
        columns = ["order_id", "route_link_id", "route_conditioned_time_check"]
        fallback = ["order_id", "planned_link_id", "strict_availability_check"]
        try:
            frame = pd.read_parquet(path, columns=columns)
            check = "route_conditioned_time_check"
            link = "route_link_id"
        except Exception:
            frame = pd.read_parquet(path, columns=fallback)
            check = "strict_availability_check"
            link = "planned_link_id"
        state_columns = []
        sample = pd.read_parquet(path)
        state_columns = [column for column in sample.columns if column.endswith("recent_traversal_count_15m")]
        row = {
            "date": date,
            "time_mode": mode,
            "rows": len(frame),
            "orders": int(frame["order_id"].nunique()),
            "links": int(frame[link].nunique()),
            "strict_availability_ratio": float(frame[check].mean()),
        }
        for column in state_columns:
            row[f"{column}_coverage"] = float(sample[column].notna().mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    estimated = read_summary(args.estimated_eval_root, "estimated_time")
    oracle = read_summary(args.oracle_eval_root, "oracle_actual_entry")
    combined = pd.concat([estimated, oracle], ignore_index=True)
    combined.to_csv(args.output_root / "estimated_oracle_metric_summary.csv", index=False)
    key_cols = ["target", "ablation"]
    metric_cols = [
        "auc_mean", "ap_mean", "spearman_mean", "lift_top5_mean",
        "order_lift_top10_mean", "brier_mean", "ece_mean",
        "interval_coverage_mean", "interval_width_mean",
    ]
    gap = estimated[key_cols + metric_cols].merge(
        oracle[key_cols + metric_cols],
        on=key_cols,
        suffixes=("_estimated", "_oracle"),
        validate="one_to_one",
    )
    for metric in metric_cols:
        gap[f"{metric}_oracle_minus_estimated"] = gap[f"{metric}_oracle"] - gap[f"{metric}_estimated"]
    gap.to_csv(args.output_root / "estimated_oracle_metric_gap.csv", index=False)
    full_gap = gap[gap["ablation"] == FULL_ABLATION].copy()
    full_gap.to_csv(args.output_root / "estimated_oracle_full_branch_gap.csv", index=False)
    ap_gaps = full_gap["ap_mean_oracle_minus_estimated"].dropna()
    auc_gaps = full_gap["auc_mean_oracle_minus_estimated"].dropna()
    if not ap_gaps.empty and not auc_gaps.empty and (ap_gaps <= 0).all() and (auc_gaps <= 0).all():
        headline = (
            "In the current three-fold evaluation, actual-entry oracle state does not improve the "
            "full route-context branch. Entry-time uncertainty is therefore not the dominant observed "
            "Stage2 bottleneck under this route-conditioned setup."
        )
    elif not ap_gaps.empty and (ap_gaps > 0).any():
        headline = (
            "Actual-entry oracle state improves at least one target, indicating recoverable error from "
            "estimated link entry timing for those targets."
        )
    else:
        headline = "The estimated-vs-oracle gap is inconclusive and should be interpreted target by target."
    dates = sorted({path.stem.split("=", 1)[-1] for path in args.estimated_dataset_root.glob("day=*.parquet")})
    coverage = pd.concat([
        daily_coverage(args.estimated_dataset_root, dates, "estimated_time"),
        daily_coverage(args.oracle_dataset_root, dates, "oracle_actual_entry"),
    ], ignore_index=True)
    coverage.to_csv(args.output_root / "estimated_oracle_daily_coverage.csv", index=False)
    report_lines = [
        "# Estimated-time vs oracle-time gap audit",
        "",
        "The estimated-time product is the deployable route-conditioned Stage2 setting. "
        "The oracle-time product reattaches lagged state by actual link entry time and is used only as an upper-bound diagnostic.",
        "",
        "## Headline",
        "",
        headline,
        "",
        "## Full route-context branch gap",
        "",
        full_gap[[
            "target",
            "auc_mean_estimated", "auc_mean_oracle", "auc_mean_oracle_minus_estimated",
            "ap_mean_estimated", "ap_mean_oracle", "ap_mean_oracle_minus_estimated",
            "spearman_mean_estimated", "spearman_mean_oracle", "spearman_mean_oracle_minus_estimated",
            "lift_top5_mean_estimated", "lift_top5_mean_oracle", "lift_top5_mean_oracle_minus_estimated",
        ]].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Daily state coverage",
        "",
        coverage.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretation",
        "",
        "- Positive oracle-minus-estimated gaps indicate recoverable error from uncertain link entry timing.",
        "- Negative oracle-minus-estimated gaps mean actual-entry state did not improve that metric in this controlled setup; this can happen when route context and rolling profiles dominate, or when same-trip timing synchronization adds noise rather than useful pre-dispatch information.",
        "- Small or mixed gaps indicate that route context and lagged state are already robust to estimated entry time.",
        "- The oracle-time branch must not be used as a Stage3 input because actual link entry time is post-trip information.",
    ]
    (args.output_root / "estimated_oracle_gap_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    manifest = {
        "estimated_eval_root": str(args.estimated_eval_root),
        "oracle_eval_root": str(args.oracle_eval_root),
        "estimated_dataset_root": str(args.estimated_dataset_root),
        "oracle_dataset_root": str(args.oracle_dataset_root),
        "full_ablation": FULL_ABLATION,
    }
    (args.output_root / "estimated_oracle_gap_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
