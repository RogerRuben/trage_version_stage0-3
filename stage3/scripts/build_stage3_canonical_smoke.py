"""Train and export the canonical Stage 3 engineering smoke vector.

Only held-out Stage 2 dispatch predictions are model features. Stage 1 order
labels are loaded into a separate target frame and never joined to the feature
matrix until after feature construction has been frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, mean_squared_error, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canonical_pipeline.manifest import load_manifest, require_canonical_input


TARGETS = ("lcs", "pmis", "rts")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2-manifest", type=Path, required=True)
    parser.add_argument("--stage1-target-manifest", type=Path, required=True)
    parser.add_argument("--train-date", required=True)
    parser.add_argument("--validation-date", required=True)
    parser.add_argument("--test-date", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--ensemble-size", type=int, default=8)
    parser.add_argument("--schema", type=Path, default=Path("config/artifact_manifest.schema.json"))
    return parser.parse_args()


def roles(manifest, workspace: Path) -> dict[str, Path]:
    return {item["role"]: workspace / item["path"] for item in manifest.data["files"]}


def route_hash(values: pd.Series) -> str:
    return hashlib.sha256("|".join(values.astype(str)).encode("utf-8")).hexdigest()[:20]


def aggregate_stage2(link_path: Path, service_path: Path) -> pd.DataFrame:
    link = pd.read_parquet(link_path).sort_values(["order_id", "route_link_seq"], kind="mergesort")
    if not link.heldout_prediction.astype(bool).all():
        raise ValueError("Stage3 refuses non-held-out Stage2 predictions")
    aggregations: dict[str, tuple[str, str | callable]] = {
        "decision_time": ("decision_time", "first"),
        "prediction_cutoff_time": ("prediction_cutoff_time", "max"),
        "link_count": ("route_link_seq", "count"),
        "route_length_m": ("route_link_length_m", "sum"),
        "route_direction_valid": ("route_direction_valid", "min"),
        "topology_gap_share": ("transition_gap", "mean"),
        "route_id": ("route_link_id", route_hash),
    }
    for target in TARGETS:
        aggregations.update({
            f"{target}_link_expected_mean": (f"{target}_expected_raw", "mean"),
            f"{target}_link_expected_max": (f"{target}_expected_raw", "max"),
            f"{target}_link_expected_q90": (f"{target}_expected_raw", lambda x: x.quantile(.9)),
            f"{target}_link_tail_mean": (f"{target}_tail_probability", "mean"),
            f"{target}_link_tail_max": (f"{target}_tail_probability", "max"),
            f"{target}_link_tail_q90": (f"{target}_tail_probability", lambda x: x.quantile(.9)),
            f"{target}_link_uncertainty_mean": (f"{target}_uncertainty", "mean"),
            f"{target}_link_uncertainty_max": (f"{target}_uncertainty", "max"),
        })
    order = link.groupby("order_id", sort=False).agg(**aggregations).reset_index()
    service = pd.read_parquet(service_path, columns=[
        "order_id", "predicted_service_time_sec", "service_time_uncertainty_sec",
        "service_time_distribution", "heldout_prediction", "realized_duration_read_allowed",
    ])
    if not service.heldout_prediction.astype(bool).all() or service.realized_duration_read_allowed.astype(bool).any():
        raise ValueError("invalid Stage2 service-time lineage")
    return order.merge(service.drop(columns=["heldout_prediction", "realized_duration_read_allowed"]),
                       on="order_id", how="inner", validate="one_to_one")


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    boundaries = np.linspace(0, 1, bins + 1)
    total = len(y); result = 0.0
    for low, high in zip(boundaries[:-1], boundaries[1:]):
        mask = (p >= low) & (p < high if high < 1 else p <= high)
        if mask.any():
            result += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(result) if total else float("nan")


def fit_bootstrap_models(x: np.ndarray, y: np.ndarray, *, classifier: bool, count: int, seed: int):
    rng = np.random.default_rng(seed)
    models = []
    for _ in range(count):
        indexes = rng.integers(0, len(y), len(y))
        if classifier and np.unique(y[indexes]).size < 2:
            indexes = np.arange(len(y))
        model = LogisticRegression(C=0.5, max_iter=1000) if classifier else Ridge(alpha=10.0)
        model.fit(x[indexes], y[indexes]); models.append(model)
    return models


def predict_bootstrap(models, x: np.ndarray, classifier: bool) -> tuple[np.ndarray, np.ndarray]:
    values = np.column_stack([
        model.predict_proba(x)[:, 1] if classifier else model.predict(x) for model in models
    ])
    return values.mean(axis=1), values.std(axis=1, ddof=0)


def choose_calibrator(y: np.ndarray, raw: np.ndarray):
    candidates: dict[str, tuple[object | None, np.ndarray]] = {"raw": (None, raw)}
    if np.unique(y).size == 2:
        clipped = np.clip(raw, 1e-6, 1 - 1e-6)
        logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        platt = LogisticRegression(C=1e6, max_iter=1000).fit(logits, y)
        candidates["platt"] = (platt, platt.predict_proba(logits)[:, 1])
        isotonic = IsotonicRegression(out_of_bounds="clip").fit(raw, y)
        candidates["isotonic"] = (isotonic, isotonic.predict(raw))
    scores = {name: float(brier_score_loss(y, values)) for name, (_, values) in candidates.items()}
    selected = min(scores, key=scores.get)
    return selected, candidates[selected][0], scores


def apply_calibrator(name: str, calibrator, raw: np.ndarray) -> np.ndarray:
    if name == "raw": return raw
    if name == "isotonic": return calibrator.predict(raw)
    clipped = np.clip(raw, 1e-6, 1 - 1e-6)
    return calibrator.predict_proba(np.log(clipped / (1 - clipped)).reshape(-1, 1))[:, 1]


def safe_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    return float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None


def main() -> None:
    args = arguments(); workspace = Path.cwd().resolve()
    if not (args.train_date < args.validation_date < args.test_date):
        raise ValueError("Stage3 train < validation < test is required")
    stage2 = load_manifest(args.stage2_manifest, args.schema, workspace)
    targets_manifest = load_manifest(args.stage1_target_manifest, args.schema, workspace)
    require_canonical_input(stage2); require_canonical_input(targets_manifest)
    if stage2.data["stage"] != "stage2" or targets_manifest.data["stage"] != "stage1":
        raise ValueError("explicit Stage2 feature and Stage1 target manifests are required")
    s2 = roles(stage2, workspace); s1 = roles(targets_manifest, workspace)
    date_list = [args.train_date, args.validation_date, args.test_date]
    feature_frames = {
        date: aggregate_stage2(s2[f"link_predictions_{date}"], s2[f"service_time_predictions_{date}"])
        for date in date_list
    }
    target_frames = {
        date: pd.read_parquet(s1[f"order_labels_{date}"]) for date in date_list
    }
    feature_columns = [column for column in feature_frames[args.train_date].columns if column not in {
        "order_id", "decision_time", "prediction_cutoff_time", "route_id", "service_time_distribution"
    }]
    if any(column.endswith(("_mean", "_tail")) and "link_" not in column for column in feature_columns):
        raise ValueError("realized Stage1 label leaked into Stage3 features")
    train_features = feature_frames[args.train_date]
    medians = train_features[feature_columns].apply(pd.to_numeric, errors="coerce").median().fillna(0)
    scaler = StandardScaler().fit(train_features[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(medians))
    x = {
        date: scaler.transform(feature_frames[date][feature_columns].apply(pd.to_numeric, errors="coerce").fillna(medians))
        for date in date_list
    }
    aligned_targets = {}
    for date in date_list:
        aligned_targets[date] = feature_frames[date][["order_id"]].merge(
            target_frames[date], on="order_id", how="left", validate="one_to_one"
        )
        if len(aligned_targets[date]) != len(feature_frames[date]):
            raise ValueError("target alignment changed order universe")

    thresholds = {
        target: float(aligned_targets[args.train_date][f"{target}_mean"].quantile(.90)) for target in TARGETS
    }
    y_binary = {
        date: {
            target: (aligned_targets[date][f"{target}_mean"].to_numpy(dtype=float) >= thresholds[target]).astype(int)
            for target in TARGETS
        } for date in date_list
    }
    for date in date_list:
        y_binary[date]["core_overall"] = np.logical_or.reduce([y_binary[date][target] for target in TARGETS]).astype(int)
    target_available = {
        date: {
            target: aligned_targets[date][f"{target}_mean"].notna().to_numpy()
            for target in TARGETS
        } for date in date_list
    }
    for date in date_list:
        target_available[date]["core_overall"] = np.ones(len(aligned_targets[date]), dtype=bool)

    model_bundle = {"feature_columns": feature_columns, "medians": medians.to_dict(), "scaler": scaler,
                    "target_thresholds": thresholds, "models": {}, "calibrators": {}}
    raw_probabilities: dict[str, dict[str, np.ndarray]] = {date: {} for date in date_list}
    calibrated_probabilities: dict[str, dict[str, np.ndarray]] = {date: {} for date in date_list}
    classifier_uncertainty: dict[str, dict[str, np.ndarray]] = {date: {} for date in date_list}
    continuous_predictions: dict[str, dict[str, np.ndarray]] = {date: {} for date in date_list}
    continuous_uncertainty: dict[str, dict[str, np.ndarray]] = {date: {} for date in date_list}
    calibration_rows = []
    for index, target in enumerate((*TARGETS, "core_overall")):
        train_valid = target_available[args.train_date][target]
        classifiers = fit_bootstrap_models(x[args.train_date][train_valid], y_binary[args.train_date][target][train_valid],
                                           classifier=True, count=args.ensemble_size, seed=args.seed + index)
        model_bundle["models"][f"{target}_classifiers"] = classifiers
        for date in date_list:
            raw_probabilities[date][target], classifier_uncertainty[date][target] = predict_bootstrap(
                classifiers, x[date], classifier=True
            )
        validation_valid = target_available[args.validation_date][target]
        method, calibrator, scores = choose_calibrator(
            y_binary[args.validation_date][target][validation_valid],
            raw_probabilities[args.validation_date][target][validation_valid]
        )
        model_bundle["calibrators"][target] = {"method": method, "object": calibrator,
                                               "fit_date": args.validation_date, "scores": scores}
        for date in date_list:
            calibrated_probabilities[date][target] = np.clip(
                apply_calibrator(method, calibrator, raw_probabilities[date][target]), 0, 1
            )
        calibration_rows.append({"target": target, "selected_method": method, "fit_date": args.validation_date,
                                 **{f"validation_brier_{name}": value for name, value in scores.items()}})
        if target in TARGETS:
            y_continuous = aligned_targets[args.train_date].loc[train_valid, f"{target}_mean"].to_numpy(dtype=float)
            regressors = fit_bootstrap_models(x[args.train_date][train_valid], y_continuous, classifier=False,
                                              count=args.ensemble_size, seed=args.seed + 100 + index)
            model_bundle["models"][f"{target}_regressors"] = regressors
            for date in date_list:
                continuous_predictions[date][target], continuous_uncertainty[date][target] = predict_bootstrap(
                    regressors, x[date], classifier=False
                )

    args.output_root.mkdir(parents=True, exist_ok=True)
    model_path = args.output_root / "models" / "stage3_smoke.joblib"; model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, model_path, compress=3)
    calibration_path = args.output_root / "calibration_selection.csv"
    pd.DataFrame(calibration_rows).to_csv(calibration_path, index=False)
    files = [{"role": "stage3_model", "path": model_path.as_posix()},
             {"role": "stage3_calibration_selection", "path": calibration_path.as_posix()}]
    metric_rows = []
    for date in date_list:
        source = feature_frames[date]
        output = source[[
            "order_id", "decision_time", "prediction_cutoff_time", "route_id", "route_length_m", "link_count",
            "route_direction_valid", "topology_gap_share", "predicted_service_time_sec",
            "service_time_uncertainty_sec", "service_time_distribution",
        ]].copy()
        output["date"] = date
        output["fold"] = {args.train_date: "train", args.validation_date: "validation", args.test_date: "test"}[date]
        for target in TARGETS:
            output[f"{target}_expected"] = np.clip(continuous_predictions[date][target], 0, 1)
            output[f"{target}_tail_probability"] = calibrated_probabilities[date][target]
            stage2_uncertainty = source[f"{target}_link_uncertainty_mean"].to_numpy(dtype=float)
            output[f"{target}_uncertainty"] = np.sqrt(
                np.square(stage2_uncertainty) + np.square(continuous_uncertainty[date][target])
            )
            valid = target_available[date][target]
            truth = aligned_targets[date].loc[valid, f"{target}_mean"].to_numpy(dtype=float)
            binary = y_binary[date][target][valid]
            probability = output.loc[valid, f"{target}_tail_probability"].to_numpy(dtype=float)
            metric_rows.append({
                "date": date, "fold": output.fold.iloc[0], "target": target,
                "auc": safe_auc(binary, probability),
                "average_precision": float(average_precision_score(binary, probability)),
                "brier": float(brier_score_loss(binary, probability)), "ece": ece(binary, probability),
                "continuous_rmse": float(np.sqrt(mean_squared_error(truth, output.loc[valid, f"{target}_expected"]))),
                "calibration_method": model_bundle["calibrators"][target]["method"],
            })
        output["core_overall_high_stress_probability"] = calibrated_probabilities[date]["core_overall"]
        output["core_overall_uncertainty"] = classifier_uncertainty[date]["core_overall"]
        output["extended_overall_high_stress_probability"] = np.nan
        output["extended_overall_status"] = "unavailable_no_dispatch_iis_model"
        output["iis_availability"] = False
        output["iis_applicability_probability"] = np.nan
        output["iis_conditional_severity"] = np.nan
        output["iis_tail_probability"] = np.nan
        output["overall_uncertainty"] = output[[f"{target}_uncertainty" for target in TARGETS]].mean(axis=1)
        output["modality_coverage_score"] = 0.75
        output["route_prediction_confidence"] = np.exp(-output.overall_uncertainty.clip(lower=0)) * (
            0.5 + 0.5 * output.route_direction_valid.astype(float)
        )
        output["condition_available"] = True
        output["model_version"] = "stage3_condition_vector_v2_engineering_smoke"
        output["feature_lineage"] = stage2.artifact_id
        output["calibration_fit_date"] = args.validation_date
        output["realized_stage1_feature_count"] = 0
        vector_path = args.output_root / "condition_vectors" / f"day={date}.parquet"
        vector_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_parquet(vector_path, index=False, compression="zstd")
        files.append({"role": f"condition_vector_{date}", "path": vector_path.as_posix()})
        binary = y_binary[date]["core_overall"]
        probability = output.core_overall_high_stress_probability.to_numpy(dtype=float)
        metric_rows.append({
            "date": date, "fold": output.fold.iloc[0], "target": "core_overall",
            "auc": safe_auc(binary, probability),
            "average_precision": float(average_precision_score(binary, probability)),
            "brier": float(brier_score_loss(binary, probability)), "ece": ece(binary, probability),
            "continuous_rmse": None, "calibration_method": model_bundle["calibrators"]["core_overall"]["method"],
        })
    metrics_path = args.output_root / "metrics.csv"; pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    files.append({"role": "stage3_metrics", "path": metrics_path.as_posix()})
    summary = {
        "status": "PASS", "schema_version": "stage3_condition_vector_v2",
        "input_stage2_artifact_id": stage2.artifact_id,
        "stage1_target_artifact_id": targets_manifest.artifact_id,
        "train_date": args.train_date, "validation_date": args.validation_date, "test_date": args.test_date,
        "feature_columns": feature_columns, "stage1_realized_feature_columns": [],
        "calibration_fit_date": args.validation_date, "extended_probability_available": False,
        "iis_policy": "availability_gated_no_zero_imputation", "files": files,
    }
    (args.output_root / "stage3_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
