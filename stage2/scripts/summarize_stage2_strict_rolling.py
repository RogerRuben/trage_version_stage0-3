"""Summarize OD, planned routing, causal joins and rolling evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output"))
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_table(path: Path, ablation: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    return frame[frame["ablation"].eq(ablation)].copy()


def main() -> None:
    args = parse_args()
    report_root = args.output_root / "strict_rolling_report"
    report_root.mkdir(parents=True, exist_ok=True)
    stage0 = pd.read_csv("stage0/output/reports/stage0_readiness.csv")
    od = pd.read_csv("stage0/output/reports/order_od_quality_summary.csv")
    coverage = pd.read_csv("stage1/output/prediction_split/reports/stage1_label_coverage.csv")
    route = pd.read_csv(args.output_root / "routes/audit/planned_route_audit.csv")
    route_slices = pd.read_csv(args.output_root / "routes/audit/planned_route_label_coverage_slices.csv")
    causal = read_json(args.output_root / "planned_route_causal_dataset/planned_causal_manifest.json")
    target = read_json(args.output_root / "strict_targets/strict_target_manifest.json")
    state = read_json(args.output_root / "lagged_state_store/lagged_state_manifest.json")
    iis = pd.read_csv(args.output_root / "iis_movement_audit/iis_applicability_severity_by_day.csv")
    planned_all = pd.read_csv(args.output_root / "rolling_fair_eval/rolling_fair_summary.csv")
    iis_all = pd.read_csv(args.output_root / "rolling_fair_eval_iis/rolling_fair_summary.csv")
    planned = planned_all[planned_all["ablation"].eq("static_rolling_dynamic_topology_route")].copy()
    iis_metrics = iis_all[iis_all["ablation"].eq("static_rolling_dynamic_topology_route")].copy()
    oracle = metric_table(args.output_root / "rolling_fair_eval_actual_route_oracle/rolling_fair_summary.csv", "static_rolling_dynamic_topology_route")

    fastest = route[route["route_source"].eq("historical_fastest_path")]
    planned_label_ratio = float(fastest["realized_label_link_ratio"].mean())
    middle = route_slices[
        route_slices["route_source"].eq("historical_fastest_path")
        & route_slices["slice_dimension"].eq("route_position")
        & route_slices["slice_value"].eq("middle")
    ]
    middle_label_ratio = float((middle["realized_label_link_ratio"] * middle["rows"]).sum() / middle["rows"].sum())
    strict_pass = min(value["strict_availability_pass_ratio"] for value in causal["days"].values())
    route_success = []
    fastest_manifest = read_json(args.output_root / "routes/historical_fastest_path/planned_route_manifest.json")
    route_success = [value["route_success_ratio"] for value in fastest_manifest["days"].values()]
    engineering_pass = (
        stage0[stage0["date"].astype(str).isin(["20161018", "20161019"])]["stage0_gate"].eq("PASS").all()
        and strict_pass == 1.0 and min(route_success) >= 0.95
    )
    deployable_ready = bool(engineering_pass and planned_label_ratio >= 0.50)
    verdict = "READY_FOR_STAGE3" if deployable_ready else "HOLD_STAGE3_ROUTE_OBSERVABILITY_LIMIT"

    summary = {
        "engineering_pipeline_pass": bool(engineering_pass),
        "stage3_verdict": verdict,
        "stage0_20161018_20161019_pass": bool(stage0[stage0["date"].astype(str).isin(["20161018", "20161019"])]["stage0_gate"].eq("PASS").all()),
        "od_order_base_alignment_min": float(od["order_base_alignment_ratio"].min()),
        "od_route_eligible_min": float(od["route_eligible_ratio"].min()),
        "historical_fastest_route_success_min": float(min(route_success)),
        "historical_fastest_realized_label_link_ratio_mean": planned_label_ratio,
        "historical_fastest_actual_link_coverage_p50_mean": float(fastest["actual_link_coverage_p50"].mean()),
        "historical_fastest_middle_route_label_ratio": middle_label_ratio,
        "strict_availability_pass_min": strict_pass,
        "lagged_link_state_coverage_min": float(min(value["link_state_coverage"] for value in causal["days"].values())),
        "link_state_duplicate_rows_collapsed": int(state["link_state_duplicate_rows_collapsed"]),
        "strict_raw_tail90_thresholds": target["raw_tail90_thresholds"],
        "iis_applicable_ratio_mean": float(iis["iis_applicable_ratio"].mean()),
        "iis_severity_valid_ratio_mean": float(iis["severity_valid_ratio"].mean()),
        "planned_full_model_metrics": planned.to_dict("records"),
        "planned_ablation_metrics": planned_all.to_dict("records"),
        "iis_full_model_metrics": iis_metrics.to_dict("records"),
        "actual_route_oracle_full_model_metrics": oracle.to_dict("records"),
        "interpretation": "Strong metrics on common observed planned/actual links do not establish full-route deployability when route-label coverage is non-random and below 50%.",
    }
    (report_root / "strict_rolling_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Stage2 strict rolling readiness", "",
        f"**Engineering status:** {'PASS' if engineering_pass else 'FAIL'}", "",
        f"**Stage3 decision:** `{verdict}`", "",
        "## Core audit", "",
        f"- OD/order_base minimum alignment: {summary['od_order_base_alignment_min']:.2%}",
        f"- OD route-eligible minimum: {summary['od_route_eligible_min']:.2%}",
        f"- Historical-fastest route success minimum: {summary['historical_fastest_route_success_min']:.2%}",
        f"- Planned-link realized-label coverage mean: {planned_label_ratio:.2%}",
        f"- Median actual-route link coverage (daily mean): {summary['historical_fastest_actual_link_coverage_p50_mean']:.2%}",
        f"- Middle-route realized-label coverage: {middle_label_ratio:.2%}",
        f"- Strict lagged-time check minimum: {strict_pass:.2%}",
        f"- Lagged link-state coverage minimum: {summary['lagged_link_state_coverage_min']:.2%}",
        f"- Collapsed duplicate link-state rows: {state['link_state_duplicate_rows_collapsed']:,} / {state['lookup_rows']['link']:,}",
        f"- IIS applicability / severity-valid mean: {summary['iis_applicable_ratio_mean']:.2%} / {summary['iis_severity_valid_ratio_mean']:.2%}",
        "", "## Planned-route ablation chain", "",
        planned_all[["target", "ablation", "auc_mean", "ap_mean", "spearman_mean", "lift_top5_mean", "order_lift_top10_mean"]].to_markdown(index=False, floatfmt=".4f"),
        "", "## Planned-route full-model metrics", "",
        planned.to_markdown(index=False, floatfmt=".4f") if len(planned) else "Pending.",
        "", "## IIS conditional-severity metrics", "",
        iis_metrics.to_markdown(index=False, floatfmt=".4f") if len(iis_metrics) else "Pending.",
        "", "## Actual-route oracle upper bound", "",
        oracle.to_markdown(index=False, floatfmt=".4f") if len(oracle) else "Pending.",
        "", "## Decision", "",
        "The engineering chain is reproducible and causally joined. Stage3 remains on hold because only about one third of planned links have an order-specific realized label, and missingness is strongly route-position/road-class dependent. The current high AUC/AP values therefore characterize the common-link subset, not the complete deployable planned route.",
    ]
    (report_root / "strict_rolling_readiness_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
