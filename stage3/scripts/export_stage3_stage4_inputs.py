"""Export Stage3 rolling test predictions as Stage4 condition-vector inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TARGET_MAP = {
    "LCS": "pred_stop_go_stress",
    "PMIS": "pred_poi_mediated_stress",
    "RTS": "pred_reliability_stress",
    "OVERALL": "overall_high_stress_probability",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("stage3/results/rolling"))
    parser.add_argument("--order-feature-root", type=Path, default=Path("stage3/output/rolling_order_features"))
    parser.add_argument("--model-dir", default="core_deepsets")
    parser.add_argument("--warehouse-root", type=Path, default=Path("stage3/output/rolling_stage2_prediction_warehouse"))
    parser.add_argument("--od-root", type=Path, default=Path("stage0/output/order_od_audited"))
    parser.add_argument("--output-root", type=Path, default=Path("stage3/output/stage4_inputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifests = []
    for fold_dir in sorted(args.results_root.glob("fold_*")):
        fold = int(fold_dir.name.split("_")[-1])
        prediction_path = fold_dir / args.model_dir / "predictions.parquet"
        feature_path = args.order_feature_root / f"fold={fold}" / "split=test" / "order_features.parquet"
        if not prediction_path.exists() or not feature_path.exists():
            continue
        predictions = pd.read_parquet(prediction_path)
        predictions = predictions[predictions["split"].eq("test")].copy()
        wide = predictions.pivot(index="order_id", columns="target", values="pred_probability").reset_index()
        wide = wide.rename(columns=TARGET_MAP)
        features = pd.read_parquet(feature_path)
        link_parts = []
        link_base = args.warehouse_root / "link_predictions" / f"fold={fold}" / "split=test"
        for link_path in sorted(link_base.glob("day=*.parquet")):
            link_parts.append(pd.read_parquet(link_path, columns=["order_id", "estimated_link_entry_time", "route_link_seq"]))
        link_times = pd.concat(link_parts, ignore_index=True)
        link_times = link_times.sort_values(["order_id", "route_link_seq"]).groupby("order_id", as_index=False).agg(decision_time=("estimated_link_entry_time", "first"))
        od_parts = []
        for date in sorted(features["date"].astype(str).unique()):
            od_path = args.od_root / f"day={date}.parquet"
            if od_path.exists():
                od_parts.append(pd.read_parquet(od_path, columns=["order_id", "origin_lon", "origin_lat", "destination_lon", "destination_lat", "origin_timestamp"]))
        od = pd.concat(od_parts, ignore_index=True) if od_parts else pd.DataFrame({"order_id": []})
        keep = [
            "order_id",
            "date",
            "route_length_m",
            "link_count",
            "movement_count",
            "iis_applicability_mean",
            "iis_severity_q90",
            "iis_tail_prob_q90",
            "iis_prediction_available",
            "modality_coverage_score",
            "route_prediction_confidence",
            "rc_lcs_uncertainty_q90",
            "rc_pmis_uncertainty_q90",
            "rc_rts_uncertainty_q90",
            "rc_lcs_q90",
            "rc_pmis_q90",
            "rc_rts_q90",
        ]
        output = features[[column for column in keep if column in features.columns]].merge(wide, on="order_id", how="inner", validate="one_to_one")
        output = output.merge(link_times, on="order_id", how="left", validate="one_to_one")
        if not od.empty:
            output = output.merge(od, on="order_id", how="left", validate="one_to_one")
        output["decision_time"] = output["decision_time"].astype(str)
        output["request_time_proxy_definition"] = "first estimated service-route link entry time; falls back to OD origin timestamp when unavailable"
        output["route_id"] = output["order_id"].astype(str).map(lambda value: f"observed_matched_service_route_proxy:{value}")
        output["route_proxy_type"] = "observed_matched_service_route_proxy"
        output["lcs_expected"] = output["rc_lcs_q90"].astype("float32")
        output["lcs_tail_probability"] = output["pred_stop_go_stress"].astype("float32")
        output["lcs_uncertainty"] = output["rc_lcs_uncertainty_q90"].astype("float32")
        output["pmis_expected"] = output["rc_pmis_q90"].astype("float32")
        output["pmis_tail_probability"] = output["pred_poi_mediated_stress"].astype("float32")
        output["pmis_uncertainty"] = output["rc_pmis_uncertainty_q90"].astype("float32")
        output["rts_expected"] = output["rc_rts_q90"].astype("float32")
        output["rts_tail_probability"] = output["pred_reliability_stress"].astype("float32")
        output["rts_uncertainty"] = output["rc_rts_uncertainty_q90"].astype("float32")
        output["intersection_applicability"] = output.get("iis_applicability_mean", pd.Series(index=output.index, dtype=float))
        output["intersection_severity"] = output.get("iis_severity_q90", pd.Series(index=output.index, dtype=float))
        output["intersection_tail_probability"] = output.get("iis_tail_prob_q90", pd.Series(index=output.index, dtype=float))
        output["iis_availability"] = output.get("iis_prediction_available", pd.Series(False, index=output.index)).astype(bool)
        output["composite_expected"] = output[["lcs_expected", "pmis_expected", "rts_expected"]].mean(axis=1)
        output["core_overall_high_stress_probability"] = output["overall_high_stress_probability"].astype("float32")
        output["extended_overall_high_stress_probability"] = output[["core_overall_high_stress_probability", "intersection_tail_probability"]].max(axis=1)
        output["pred_intersection_stress"] = output["intersection_tail_probability"]
        output["pred_composite_operational_stress"] = output["composite_expected"]
        output["overall_uncertainty"] = output[["rc_lcs_uncertainty_q90", "rc_pmis_uncertainty_q90", "rc_rts_uncertainty_q90"]].mean(axis=1)
        output["iis_applicability"] = output["intersection_applicability"]
        output["iis_severity"] = output["intersection_severity"]
        output["iis_tail_probability"] = output["intersection_tail_probability"]
        output["iis_coverage_quality"] = output.get("iis_prediction_available", pd.Series(False, index=output.index)).astype(float)
        output["model_version"] = f"Stage3-{args.model_dir}-fold{fold}"
        output["prediction_cutoff_time"] = "pre_dispatch_estimated_entry_contract"
        fold_root = args.output_root / f"fold={fold}"
        fold_root.mkdir(parents=True, exist_ok=True)
        output.to_parquet(fold_root / "stage4_inputs.parquet", index=False, compression="zstd")
        manifests.append({"fold": fold, "rows": len(output), "orders": int(output["order_id"].nunique()), "model_dir": args.model_dir})
    manifest = {"status": "PASS" if manifests else "FAIL", "folds": manifests}
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if not manifests:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
