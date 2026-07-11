"""Evaluate Stage2 Deep v3 predictions against the LightGBM benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lightgbm-root", type=Path, default=Path("stage2/output/route_conditioned_eval/estimated_time"))
    parser.add_argument("--deep-root", type=Path, default=Path("stage2/output/deep_v3/feasibility_100k/rc_mstnet"))
    parser.add_argument("--movement-root", type=Path, default=Path("stage2/output/deep_v3/feasibility_100k/rc_mstnet_movement"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/deep_v3/metrics"))
    return parser.parse_args()


def read_deep(root: Path) -> pd.DataFrame:
    path = root / "rc_mstnet_metrics_by_fold.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["model"] = "RC-MSTNet"
    frame["target"] = frame["target"].str.upper()
    return frame


def read_movement(root: Path) -> pd.DataFrame:
    path = root / "rc_mstnet_movement_metrics_by_fold.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["model"] = "RC-MSTNet-Movement"
    return frame


def read_lightgbm(root: Path) -> pd.DataFrame:
    path = root / "rolling_fair_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame = frame[frame["ablation"].eq("static_rolling_dynamic_topology_route")].copy()
    frame["model"] = "LightGBM route-context"
    frame["split"] = "test"
    frame["target"] = frame["target"].str.upper()
    rename = {
        "precision_top5": "precision_top5",
        "recall_top5": "recall_top5",
        "lift_top5": "lift_top5",
        "precision_top10": "precision_top10",
        "recall_top10": "recall_top10",
        "lift_top10": "lift_top10",
    }
    return frame.rename(columns=rename)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    frames = [read_lightgbm(args.lightgbm_root), read_deep(args.deep_root), read_movement(args.movement_root)]
    combined = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if any(not frame.empty for frame in frames) else pd.DataFrame()
    combined.to_csv(args.output_root / "deep_v3_model_comparison_by_fold.csv", index=False)
    if combined.empty:
        (args.output_root / "deep_v3_eval_report.md").write_text("# Stage2 Deep v3 evaluation\n\nNo model metrics found yet.\n", encoding="utf-8")
        return
    numeric = [column for column in ["auc", "ap", "spearman", "mae", "rmse", "lift_top5", "lift_top10", "order_lift_top10"] if column in combined.columns]
    summary = combined.groupby(["model", "split", "target"], as_index=False).agg({column: "mean" for column in numeric} | {"fold": "nunique"})
    summary = summary.rename(columns={"fold": "folds"})
    summary.to_csv(args.output_root / "deep_v3_model_comparison_summary.csv", index=False)
    test = summary[summary["split"].eq("test")].copy()
    gain_rows = []
    baseline = test[test["model"].eq("LightGBM route-context")].set_index("target")
    candidate = test[test["model"].eq("RC-MSTNet")].set_index("target")
    for target in sorted(set(baseline.index) & set(candidate.index)):
        row = {"target": target}
        for column in numeric:
            deep_value = candidate.at[target, column]
            baseline_value = baseline.at[target, column]
            row[f"rc_mstnet_{column}"] = deep_value
            row[f"lightgbm_{column}"] = baseline_value
            row[f"delta_{column}"] = deep_value - baseline_value
        gain_rows.append(row)
    gains = pd.DataFrame(gain_rows)
    gains.to_csv(args.output_root / "deep_v3_gain_vs_lightgbm.csv", index=False)
    lines = [
        "# Stage2 Deep v3 evaluation",
        "",
        "This report compares the route-conditioned LightGBM benchmark with Deep v3 predictions when available. "
        "A Deep run with fewer than the requested 100k train orders remains a feasibility/protocol run.",
        "",
        "## Test summary",
        "",
        test.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## RC-MSTNet gain over LightGBM",
        "",
        gains.to_markdown(index=False, floatfmt=".4f") if not gains.empty else "No aligned model targets found.",
        "",
        "## Current decision rule",
        "",
        "Deep can replace or hybridize with LightGBM only if gains are stable across folds in AP, Spearman, Lift@TopK, or order-level tail separation without worse calibration/uncertainty.",
    ]
    (args.output_root / "deep_v3_eval_report.md").write_text("\n".join(lines), encoding="utf-8")
    (args.output_root / "deep_v3_eval_manifest.json").write_text(json.dumps({"rows": len(combined), "models": sorted(combined["model"].dropna().unique())}, indent=2), encoding="utf-8")
    print(test.to_string(index=False))


if __name__ == "__main__":
    main()
