"""Build strictly held-out dispatch-time Stage 2 smoke predictions.

This is an engineering model for the canonical rebaseline, not a replacement
for the formal RC-MSTNet.  It deliberately uses only immutable route/network
features available at one order-level decision cutoff.  The upstream-only fit
day precedes every exported prediction day.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canonical_pipeline.manifest import load_manifest, require_canonical_input
from stage2.canonical.dispatch_time import audit_dispatch_features


TARGETS = ("lcs", "pmis", "rts")
FEATURES = (
    "route_link_seq", "route_link_count", "route_link_length_m", "route_length_m",
    "position_ratio", "distance_to_destination_ratio", "hour", "weekday", "is_weekend",
    "activity_intensity_index", "minor_road", "road_class_code", "speed_limit", "lane_num",
    "oneway_indicator", "bridge_indicator", "tunnel_indicator", "signal_indicator",
    "route_direction_valid", "transition_gap", "is_interpolated",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage0-manifest", type=Path, required=True)
    parser.add_argument("--stage1-manifest", type=Path, required=True)
    parser.add_argument("--fit-date", required=True)
    parser.add_argument("--train-date", required=True)
    parser.add_argument("--validation-date", required=True)
    parser.add_argument("--test-date", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--ensemble-size", type=int, default=4)
    parser.add_argument("--schema", type=Path, default=Path("config/artifact_manifest.schema.json"))
    return parser.parse_args()


def role_paths(manifest, workspace: Path) -> dict[str, Path]:
    return {item["role"]: workspace / item["path"] for item in manifest.data["files"]}


def truth_frame(path: Path) -> pd.DataFrame:
    columns = ["order_id", "link_id", "link_seq", "enter_time", "exit_time", "travel_time_sec"]
    columns += [f"{target}_raw" for target in TARGETS]
    frame = pd.read_parquet(path, columns=columns)
    frame["order_id"] = frame.order_id.astype(str)
    frame["route_link_id"] = frame.link_id.astype(str)
    frame["route_link_seq"] = frame.link_seq.astype(int)
    return frame.drop(columns=["link_id", "link_seq"])


def _binary(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool).astype(np.int8)


def route_features(
    route_path: Path,
    roads: pd.DataFrame,
    exposure: pd.DataFrame,
    road_classes: dict[str, int] | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    route = pd.read_parquet(route_path, columns=[
        "order_id", "link_id", "route_sequence", "timestamp", "is_interpolated",
        "transition_path_status", "route_direction_valid",
    ])
    route["order_id"] = route.order_id.astype(str)
    route["route_link_id"] = route.link_id.astype(str)
    route["route_link_seq"] = route.route_sequence.astype(int)
    route = route.drop(columns=["link_id", "route_sequence"])
    route["decision_time"] = pd.to_datetime(
        route.groupby("order_id").timestamp.transform("min"), unit="s", utc=True
    )
    road = roads.copy()
    road["route_link_id"] = road.link_id.astype(str)
    keep = [
        "route_link_id", "road_class", "length_m", "speed_limit", "lane_num", "oneway_code",
        "bridge", "tunnel", "signal",
    ]
    route = route.merge(road[keep], on="route_link_id", how="left", validate="many_to_one")
    expo = exposure.copy()
    expo["route_link_id"] = expo.link_id.astype(str)
    activity = "activity_intensity_index"
    if activity not in expo:
        poi = [column for column in expo if column.startswith("poi_density_100m_")]
        expo[activity] = expo[poi].sum(axis=1).rank(pct=True) if poi else 0.0
    route = route.merge(expo[["route_link_id", activity]].drop_duplicates("route_link_id"),
                        on="route_link_id", how="left", validate="many_to_one")
    route["route_link_length_m"] = pd.to_numeric(route.length_m, errors="coerce")
    group = route.groupby("order_id", sort=False)
    route["route_link_count"] = group.route_link_seq.transform("count").astype(int)
    route["route_length_m"] = group.route_link_length_m.transform("sum")
    denominator = (route.route_link_count - 1).clip(lower=1)
    route["position_ratio"] = route.route_link_seq / denominator
    route["distance_to_destination_ratio"] = 1.0 - route.position_ratio
    route["hour"] = route.decision_time.dt.hour.astype(int)
    route["weekday"] = route.decision_time.dt.weekday.astype(int)
    route["is_weekend"] = route.weekday.ge(5).astype(np.int8)
    classes = route.road_class.fillna("unknown").astype(str)
    if road_classes is None:
        road_classes = {value: index for index, value in enumerate(sorted(classes.unique()))}
    route["road_class_code"] = classes.map(road_classes).fillna(-1).astype(int)
    route["minor_road"] = classes.isin({"residential", "service", "living_street", "track"}).astype(np.int8)
    route["speed_limit"] = pd.to_numeric(route.speed_limit, errors="coerce").fillna(0)
    route["lane_num"] = pd.to_numeric(route.lane_num, errors="coerce").fillna(0)
    route["oneway_indicator"] = route.oneway_code.fillna("").astype(str).ne("").astype(np.int8)
    route["bridge_indicator"] = route.bridge.fillna("F").astype(str).str.upper().isin({"T", "TRUE", "1"}).astype(np.int8)
    route["tunnel_indicator"] = route.tunnel.fillna("F").astype(str).str.upper().isin({"T", "TRUE", "1"}).astype(np.int8)
    route["signal_indicator"] = _binary(route.signal)
    route["route_direction_valid"] = _binary(route.route_direction_valid)
    route["transition_gap"] = route.transition_path_status.eq("gap").astype(np.int8)
    route["is_interpolated"] = _binary(route.is_interpolated)
    route["feature_availability_timestamp"] = route.decision_time
    route["prediction_mode"] = "dispatch_time"
    route["information_cutoff"] = route.decision_time
    return route, road_classes


def matrix(frame: pd.DataFrame, medians: dict[str, float] | None = None) -> tuple[np.ndarray, dict[str, float]]:
    numeric = frame.loc[:, FEATURES].apply(pd.to_numeric, errors="coerce")
    if medians is None:
        medians = {column: float(numeric[column].median()) if numeric[column].notna().any() else 0.0 for column in FEATURES}
    return numeric.fillna(medians).to_numpy(dtype=np.float32), medians


def fit_ensemble(x: np.ndarray, y: np.ndarray, seeds: list[int], classifier: bool):
    models = []
    for seed in seeds:
        cls = HistGradientBoostingClassifier if classifier else HistGradientBoostingRegressor
        model = cls(
            max_iter=70, learning_rate=0.07, max_leaf_nodes=15, min_samples_leaf=30,
            l2_regularization=1.0, random_state=seed,
        )
        model.fit(x, y)
        models.append(model)
    return models


def predict_ensemble(models, x: np.ndarray, classifier: bool) -> tuple[np.ndarray, np.ndarray]:
    values = np.column_stack([
        model.predict_proba(x)[:, 1] if classifier else model.predict(x)
        for model in models
    ])
    return values.mean(axis=1), values.std(axis=1, ddof=0)


def safe_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    return float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None


def main() -> None:
    args = arguments()
    workspace = Path.cwd().resolve()
    stage0 = load_manifest(args.stage0_manifest, args.schema, workspace)
    stage1 = load_manifest(args.stage1_manifest, args.schema, workspace)
    require_canonical_input(stage0); require_canonical_input(stage1)
    if stage0.data["stage"] != "stage0" or stage1.data["stage"] != "stage1":
        raise ValueError("explicit Stage0 and Stage1 canonical manifests are required")
    s0 = role_paths(stage0, workspace); s1 = role_paths(stage1, workspace)
    dates = [args.fit_date, args.train_date, args.validation_date, args.test_date]
    if not (args.fit_date < args.train_date < args.validation_date < args.test_date):
        raise ValueError("fit < train < validation < test is required")
    roads = pd.read_parquet(s0["roads"], columns=[
        "link_id", "road_class", "length_m", "speed_limit", "lane_num", "oneway_code",
        "bridge", "tunnel", "signal",
    ])
    exposure = pd.read_parquet(s0["poi_exposure"])
    features: dict[str, pd.DataFrame] = {}
    truths: dict[str, pd.DataFrame] = {}
    road_classes = None
    for date in dates:
        features[date], road_classes = route_features(s0[f"routes_{date}"], roads, exposure, road_classes)
        truths[date] = truth_frame(s1[f"link_labels_{date}"])
        features[date] = features[date].merge(
            truths[date], on=["order_id", "route_link_id", "route_link_seq"], how="left",
            validate="one_to_one",
        )
        if features[date][[f"{target}_raw" for target in TARGETS]].isna().all(axis=1).any():
            raise ValueError(f"{date}: planned route rows missing all Stage1 targets")

    fit = features[args.fit_date]
    x_fit, medians = matrix(fit)
    seeds = [args.seed + 1009 * index for index in range(args.ensemble_size)]
    models: dict[str, object] = {"feature_columns": FEATURES, "medians": medians, "road_classes": road_classes}
    thresholds = {}
    residual_scales = {}
    for target in TARGETS:
        valid = fit[f"{target}_raw"].notna().to_numpy()
        y = fit.loc[valid, f"{target}_raw"].to_numpy(dtype=float)
        threshold = float(np.quantile(y, 0.90)); thresholds[target] = threshold
        regressors = fit_ensemble(x_fit[valid], y, seeds, classifier=False)
        fitted, _ = predict_ensemble(regressors, x_fit[valid], classifier=False)
        residual_scales[target] = float(np.sqrt(np.mean(np.square(y - fitted))))
        classifiers = fit_ensemble(x_fit[valid], (y >= threshold).astype(int), seeds, classifier=True)
        models[f"{target}_regressors"] = regressors
        models[f"{target}_classifiers"] = classifiers

    order_fit = fit.groupby("order_id", sort=False).agg(
        route_link_count=("route_link_seq", "count"),
        route_length_m=("route_link_length_m", "sum"),
        hour=("hour", "first"), weekday=("weekday", "first"),
        route_direction_valid=("route_direction_valid", "first"),
        actual_service_time_sec=("travel_time_sec", "sum"),
    ).reset_index()
    order_features = ("route_link_count", "route_length_m", "hour", "weekday", "route_direction_valid")
    service_medians = {column: float(order_fit[column].median()) for column in order_features}
    x_service = order_fit.loc[:, order_features].fillna(service_medians).to_numpy(dtype=np.float32)
    service_models = fit_ensemble(x_service, order_fit.actual_service_time_sec.to_numpy(dtype=float), seeds, classifier=False)
    service_fitted, _ = predict_ensemble(service_models, x_service, classifier=False)
    service_residual = float(np.sqrt(np.mean(np.square(order_fit.actual_service_time_sec - service_fitted))))
    models.update({
        "service_models": service_models, "service_feature_columns": order_features,
        "service_medians": service_medians, "thresholds": thresholds,
        "residual_scales": residual_scales, "service_residual_scale": service_residual,
    })
    args.output_root.mkdir(parents=True, exist_ok=True)
    model_path = args.output_root / "models" / "stage2_dispatch_smoke.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, model_path, compress=3)

    fit_cutoff = pd.to_datetime(fit.exit_time.max(), unit="s", utc=True)
    metrics = []
    files = [{"role": "stage2_dispatch_model", "path": model_path.as_posix()}]
    fold_by_date = {args.train_date: "train", args.validation_date: "validation", args.test_date: "test"}
    for date in (args.train_date, args.validation_date, args.test_date):
        frame = features[date].copy()
        x, _ = matrix(frame, medians)
        output = frame[[
            "order_id", "route_link_id", "route_link_seq", "route_link_count", "route_link_length_m",
            "route_length_m", "position_ratio", "route_direction_valid", "transition_gap", "decision_time",
            "feature_availability_timestamp",
        ]].copy()
        output["date"] = date
        output["fold"] = fold_by_date[date]
        output["prediction_mode"] = "dispatch_time"
        output["heldout_prediction"] = True
        output["prediction_cutoff_time"] = output.decision_time
        output["model_training_cutoff"] = fit_cutoff
        output["route_source"] = "matched_route_as_assigned_route_proxy"
        output["decision_time_source"] = "first_matched_route_timestamp_proxy"
        output["feature_schema_version"] = "stage2_dispatch_static_route_v2"
        output["requested_level_support_count"] = 0
        output["fallback_level"] = "static_route_model"
        output["fallback_support_count"] = int(len(fit))
        output["fallback_value_source"] = f"fit_only:{args.fit_date}:static_route_model"
        for target in TARGETS:
            expected, spread = predict_ensemble(models[f"{target}_regressors"], x, classifier=False)
            probability, probability_spread = predict_ensemble(models[f"{target}_classifiers"], x, classifier=True)
            output[f"{target}_expected_raw"] = np.clip(expected, 0.0, 1.0)
            output[f"{target}_tail_probability"] = np.clip(probability, 0.0, 1.0)
            output[f"{target}_uncertainty"] = np.sqrt(np.square(spread) + residual_scales[target] ** 2)
            valid = frame[f"{target}_raw"].notna().to_numpy()
            true = frame.loc[valid, f"{target}_raw"].to_numpy(dtype=float)
            binary = (true >= thresholds[target]).astype(int)
            pred = output.loc[valid, f"{target}_expected_raw"].to_numpy(dtype=float)
            prob = output.loc[valid, f"{target}_tail_probability"].to_numpy(dtype=float)
            metrics.append({
                "date": date, "fold": fold_by_date[date], "target": target,
                "rows": int(valid.sum()), "tail_threshold_fit_only": thresholds[target],
                "rmse": float(math.sqrt(mean_squared_error(true, pred))),
                "spearman": float(pd.Series(true).corr(pd.Series(pred), method="spearman")),
                "auc": safe_auc(binary, prob),
                "average_precision": float(average_precision_score(binary, prob)),
            })
        link_path = args.output_root / "link_predictions" / f"day={date}.parquet"
        link_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_parquet(link_path, index=False, compression="zstd")
        files.append({"role": f"link_predictions_{date}", "path": link_path.as_posix()})

        order = frame.groupby("order_id", sort=False).agg(
            decision_time=("decision_time", "first"), route_link_count=("route_link_seq", "count"),
            route_length_m=("route_link_length_m", "sum"), hour=("hour", "first"),
            weekday=("weekday", "first"), route_direction_valid=("route_direction_valid", "first"),
            actual_service_time_sec=("travel_time_sec", "sum"),
        ).reset_index()
        x_order = order.loc[:, order_features].fillna(service_medians).to_numpy(dtype=np.float32)
        expected, spread = predict_ensemble(service_models, x_order, classifier=False)
        order_output = order.drop(columns="actual_service_time_sec")
        order_output["date"] = date; order_output["fold"] = fold_by_date[date]
        order_output["predicted_service_time_sec"] = np.maximum(expected, 30.0)
        order_output["service_time_uncertainty_sec"] = np.sqrt(np.square(spread) + service_residual ** 2)
        order_output["service_time_distribution"] = "normal_truncated_positive_engineering_smoke"
        order_output["prediction_cutoff_time"] = order_output.decision_time
        order_output["model_training_cutoff"] = fit_cutoff
        order_output["heldout_prediction"] = True
        order_output["realized_duration_read_allowed"] = False
        service_path = args.output_root / "service_time_predictions" / f"day={date}.parquet"
        service_path.parent.mkdir(parents=True, exist_ok=True)
        order_output.to_parquet(service_path, index=False, compression="zstd")
        files.append({"role": f"service_time_predictions_{date}", "path": service_path.as_posix()})
        metrics.append({
            "date": date, "fold": fold_by_date[date], "target": "service_time_sec", "rows": int(len(order)),
            "mae": float(mean_absolute_error(order.actual_service_time_sec, order_output.predicted_service_time_sec)),
            "rmse": float(math.sqrt(mean_squared_error(order.actual_service_time_sec, order_output.predicted_service_time_sec))),
        })

    metrics_path = args.output_root / "metrics.csv"
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    files.append({"role": "stage2_metrics", "path": metrics_path.as_posix()})
    metadata = {
        "status": "PASS", "schema_version": "stage2_dispatch_prediction_v2",
        "input_stage0_artifact_id": stage0.artifact_id, "input_stage1_artifact_id": stage1.artifact_id,
        "fit_date": args.fit_date, "prediction_dates": [args.train_date, args.validation_date, args.test_date],
        "model_training_cutoff": fit_cutoff.isoformat(), "feature_columns": list(FEATURES),
        "forbidden_feature_columns": [], "tail_thresholds": thresholds,
        "model_role": "engineering_smoke_not_formal_rc_mstnet", "files": files,
    }
    (args.output_root / "stage2_summary.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
