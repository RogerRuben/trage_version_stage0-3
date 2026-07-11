"""Run three rolling folds with common rows and causal feature ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score


TARGETS = ["lcs", "iis", "rts", "pmis"]
FOLDS = [
    {"fold": 1, "train": ["20161009", "20161010", "20161011", "20161012", "20161013", "20161014", "20161015"], "validation": ["20161016"], "test": ["20161017"]},
    {"fold": 2, "train": ["20161010", "20161011", "20161012", "20161013", "20161014", "20161015", "20161016"], "validation": ["20161017"], "test": ["20161018"]},
    {"fold": 3, "train": ["20161011", "20161012", "20161013", "20161014", "20161015", "20161016", "20161017"], "validation": ["20161018"], "test": ["20161019"]},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("stage2/output/planned_route_causal_dataset"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/rolling_fair_eval"))
    parser.add_argument("--fold-config", type=Path, default=Path("rolling_threefold_config.json"))
    parser.add_argument("--targets", nargs="+", default=TARGETS, choices=TARGETS)
    parser.add_argument("--max-train-rows", type=int, default=500_000)
    parser.add_argument("--num-boost-round", type=int, default=350)
    parser.add_argument("--bootstrap-reps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_dates(root: Path, dates: list[str]) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(root / f"day={date}.parquet") for date in dates], ignore_index=True)


def add_temporal(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    time_column = "prediction_link_entry_time" if "prediction_link_entry_time" in frame.columns else "estimated_link_entry_time"
    local = pd.to_datetime(frame[time_column], utc=True).dt.tz_convert("Asia/Shanghai")
    frame["estimated_hour"] = local.dt.hour
    frame["estimated_weekday"] = local.dt.dayofweek
    frame["estimated_is_weekend"] = local.dt.dayofweek.ge(5).astype("int8")
    return frame


def feature_sets(columns: list[str], target: str) -> dict[str, list[str]]:
    poi = [column for column in columns if column.startswith("poi_density_100m_")]
    static = [column for column in [
        "road_class", "area_grid", "planned_link_length_m", "endpoint_degree", "link_fragmentation", "minor_road",
        "activity_intensity_index", "estimated_hour", "estimated_weekday", "estimated_is_weekend",
    ] + poi if column in columns]
    rolling = [column for column in [
        f"rolling_{target}_raw_mean", f"rolling_{target}_raw_std", f"rolling_{target}_history_count",
    ] if column in columns]
    local_state = [
        column for column in columns
        if any(column.startswith(prefix) for prefix in ["link_recent_", "area_recent_", "network_recent_"])
        and "timestamp" not in column
    ]
    topology = [column for column in columns if column.startswith("upstream_recent_") or column.startswith("downstream_recent_") or column.endswith("neighbor_count")]
    route = [column for column in [
        "planned_link_seq", "planned_route_link_count", "position_ratio", "distance_to_destination_ratio",
        "origin_snap_distance_m", "destination_snap_distance_m", "estimated_link_travel_time_sec",
        "route_link_seq", "route_link_count", "route_link_length_m", "prediction_time_bin",
        "prediction_hour", "prediction_weekday", "prediction_is_weekend",
    ] if column in columns]
    return {
        "static": static,
        "static_rolling": static + rolling,
        "static_rolling_dynamic": static + rolling + local_state,
        "static_rolling_dynamic_topology": static + rolling + local_state + topology,
        "static_rolling_dynamic_topology_route": static + rolling + local_state + topology + route,
    }


def prepare_features(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, features: list[str]):
    train_x, validation_x, test_x = train[features].copy(), validation[features].copy(), test[features].copy()
    for column in features:
        if train_x[column].dtype == object or str(train_x[column].dtype).startswith("string"):
            categories = pd.Index(train_x[column].dropna().astype(str).unique())
            mapping = pd.Series(np.arange(len(categories), dtype="int32"), index=categories)
            for frame in [train_x, validation_x, test_x]:
                frame[column] = frame[column].astype(str).map(mapping).fillna(-1).astype("int32")
        else:
            for frame in [train_x, validation_x, test_x]:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float32")
    return train_x, validation_x, test_x


def ece(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (probability >= left) & ((probability <= right) if right == 1 else (probability < right))
        if mask.any():
            total += mask.mean() * abs(y[mask].mean() - probability[mask].mean())
    return float(total)


def tail_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    result = {"auc": roc_auc_score(y, score), "ap": average_precision_score(y, score)} if len(np.unique(y)) == 2 else {"auc": np.nan, "ap": np.nan}
    base = y.mean()
    for share in [0.05, 0.10]:
        count = max(1, int(len(y) * share))
        top = y[np.argsort(score)[-count:]]
        result[f"precision_top{int(share*100)}"] = float(top.mean())
        result[f"recall_top{int(share*100)}"] = float(top.sum() / max(y.sum(), 1))
        result[f"lift_top{int(share*100)}"] = float(top.mean() / base) if base > 0 else np.nan
    return result


def order_metrics(frame: pd.DataFrame, raw: str, tail: str, prediction: np.ndarray) -> dict:
    data = frame[["order_id", raw, tail]].copy()
    data["prediction"] = prediction
    orders = data.groupby("order_id").agg(
        true_q90=(raw, lambda value: value.quantile(0.90)),
        true_tail=(tail, "max"), pred_q90=("prediction", lambda value: value.quantile(0.90)),
    )
    metrics = tail_metrics(orders["true_tail"].astype(int).to_numpy(), orders["pred_q90"].to_numpy())
    return {f"order_{key}": value for key, value in metrics.items()} | {"orders": len(orders)}


def block_bootstrap(predictions: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for (target, ablation), frame in predictions.groupby(["target", "ablation"]):
        blocks = frame["date"].astype(str) + "|" + frame["order_id"].astype(str)
        groups = {key: index.to_numpy() for key, index in frame.groupby(blocks).groups.items()}
        keys = np.array(list(groups), dtype=object)
        estimates = {"auc": [], "ap": [], "lift_top5": [], "lift_top10": []}
        for _ in range(reps):
            sampled = rng.choice(keys, size=len(keys), replace=True)
            index = np.concatenate([groups[key] for key in sampled])
            y = frame.loc[index, "true_tail"].astype(int).to_numpy()
            score = frame.loc[index, "pred_tail_probability"].to_numpy(dtype=float)
            metric = tail_metrics(y, score)
            for name in estimates:
                estimates[name].append(metric[name])
        for name, values in estimates.items():
            rows.append({
                "target": target, "ablation": ablation, "metric": name, "bootstrap_reps": reps,
                "estimate_mean": float(np.nanmean(values)),
                "ci95_low": float(np.nanquantile(values, 0.025)),
                "ci95_high": float(np.nanquantile(values, 0.975)),
                "block": "date_order_id",
            })
    return pd.DataFrame(rows)


def calibrated_probability(validation_tail: pd.Series, validation_score: np.ndarray, test_score: np.ndarray) -> np.ndarray:
    if validation_tail.nunique() != 2:
        return np.full(len(test_score), float(validation_tail.mean()))
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(validation_score, validation_tail.astype(int))
    return np.clip(calibrator.predict(test_score), 0, 1)


def evaluate_variant(
    fold: dict, target: str, ablation: str, train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame,
    pred_validation: np.ndarray, pred_test: np.ndarray, test_probability: np.ndarray,
    feature_count: int,
) -> tuple[dict, np.ndarray]:
    raw = f"target_{target}_raw"
    tail = f"target_{target}_tail90_raw"
    y = test[raw].to_numpy(dtype=float)
    high = test[tail].astype(int).to_numpy()
    scale_column = f"rolling_{target}_raw_std"
    scale_validation = validation.get(scale_column, pd.Series(1.0, index=validation.index)).fillna(1.0).clip(lower=0.02).to_numpy()
    scale_test = test.get(scale_column, pd.Series(1.0, index=test.index)).fillna(1.0).clip(lower=0.02).to_numpy()
    normalized_residual = np.abs(validation[raw].to_numpy(dtype=float) - pred_validation) / scale_validation
    conformal_q90 = float(np.quantile(normalized_residual, 0.90))
    half_width = conformal_q90 * scale_test
    metrics = {
        "fold": fold["fold"], "target": target, "ablation": ablation,
        "train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test), "features": feature_count,
        "mae": mean_absolute_error(y, pred_test), "rmse": mean_squared_error(y, pred_test, squared=False),
        "spearman": spearmanr(y, pred_test).statistic,
        "brier_calibrated": brier_score_loss(high, test_probability), "ece_calibrated": ece(high, test_probability),
        "conformal_q90": conformal_q90,
        "interval_90_coverage": float(((y >= pred_test - half_width) & (y <= pred_test + half_width)).mean()),
        "interval_mean_width": float((2 * half_width).mean()),
        **tail_metrics(high, test_probability),
        **order_metrics(test, raw, tail, pred_test),
    }
    return metrics, half_width


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.fold_config.exists():
        configured = json.loads(args.fold_config.read_text(encoding="utf-8"))["folds"]
        folds = [{
            "fold": value["fold"], "train": value["train_dates"],
            "validation": [value["validation_date"]], "test": [value["test_date"]],
        } for value in configured]
    else:
        folds = FOLDS
    metric_rows, prediction_parts = [], []
    for fold in folds:
        train_all = add_temporal(load_dates(args.dataset_root, fold["train"]))
        validation_all = add_temporal(load_dates(args.dataset_root, fold["validation"]))
        test_all = add_temporal(load_dates(args.dataset_root, fold["test"]))
        for target in args.targets:
            raw = f"target_{target}_raw"
            tail = f"target_{target}_tail90_raw"
            valid_train = train_all[raw].notna() & train_all[tail].notna()
            valid_validation = validation_all[raw].notna() & validation_all[tail].notna()
            valid_test = test_all[raw].notna() & test_all[tail].notna()
            train = train_all.loc[valid_train].copy()
            validation = validation_all.loc[valid_validation].copy()
            test = test_all.loc[valid_test].copy()
            if len(train) > args.max_train_rows:
                train = train.sample(n=args.max_train_rows, random_state=args.seed + fold["fold"])
            if min(len(train), len(validation), len(test)) < 100:
                continue
            sets = feature_sets(list(train.columns), target)
            prediction_cache = {}
            for ablation, features in sets.items():
                train_x, validation_x, test_x = prepare_features(train, validation, test, features)
                regressor = lgb.LGBMRegressor(
                    objective="regression", n_estimators=args.num_boost_round, learning_rate=0.035,
                    num_leaves=64, min_child_samples=100, subsample=0.9, colsample_bytree=0.9,
                    random_state=args.seed, n_jobs=-1, verbosity=-1,
                )
                weights = np.where(train[tail].astype(bool), 4.0, 1.0)
                regressor.fit(
                    train_x, train[raw].astype(float), sample_weight=weights,
                    eval_set=[(validation_x, validation[raw].astype(float))], eval_metric="l2",
                    callbacks=[lgb.early_stopping(30, verbose=False)],
                )
                pred_validation = np.clip(regressor.predict(validation_x), 0, 1)
                pred_test = np.clip(regressor.predict(test_x), 0, 1)
                classifier = lgb.LGBMClassifier(
                    objective="binary", n_estimators=args.num_boost_round, learning_rate=0.035,
                    num_leaves=48, min_child_samples=100, subsample=0.9, colsample_bytree=0.9,
                    random_state=args.seed, n_jobs=-1, verbosity=-1,
                )
                classifier.fit(
                    train_x, train[tail].astype(int),
                    eval_set=[(validation_x, validation[tail].astype(int))], eval_metric="binary_logloss",
                    callbacks=[lgb.early_stopping(30, verbose=False)],
                )
                val_probability = classifier.predict_proba(validation_x)[:, 1]
                test_probability = classifier.predict_proba(test_x)[:, 1]
                if validation[tail].nunique() == 2:
                    calibrator = IsotonicRegression(out_of_bounds="clip").fit(val_probability, validation[tail].astype(int))
                    test_probability = calibrator.predict(test_probability)
                metrics, half_width = evaluate_variant(
                    fold, target, ablation, train, validation, test,
                    pred_validation, pred_test, test_probability, len(features),
                )
                metric_rows.append(metrics)
                prediction_cache[ablation] = {"validation": pred_validation, "test": pred_test}
                if ablation == "static_rolling_dynamic_topology_route":
                    link_column = "planned_link_id" if "planned_link_id" in test.columns else "route_link_id"
                    seq_column = "planned_link_seq" if "planned_link_seq" in test.columns else "route_link_seq"
                    part = test[["order_id", "date", link_column, seq_column]].copy()
                    part = part.rename(columns={link_column: "planned_link_id", seq_column: "planned_link_seq"})
                    part["fold"] = fold["fold"]
                    part["target"] = target
                    part["ablation"] = ablation
                    part["true_raw"] = test[raw].to_numpy(dtype=float)
                    part["true_tail"] = test[tail].astype(int).to_numpy()
                    part["pred_raw"] = pred_test
                    part["pred_tail_probability"] = test_probability
                    part["pred_uncertainty_half_width_90"] = half_width
                    prediction_parts.append(part)
                print(f"fold={fold['fold']} target={target} ablation={ablation} AP={metrics['ap']:.4f}", flush=True)

            # RTS is dominated by temporal/path propagation.  This branch learns
            # only the deviation from the strictly historical raw profile, then
            # validation-selects a convex fusion with the full causal tabular model.
            if target == "rts" and "static_rolling_dynamic_topology_route" in prediction_cache:
                base_column = "rolling_rts_raw_mean"
                structural = [
                    column for column in sets["static_rolling_dynamic_topology_route"]
                    if column not in sets["static_rolling"]
                ]
                if base_column in train and structural:
                    train_x, validation_x, test_x = prepare_features(train, validation, test, structural)
                    fallback = float(train[raw].mean())
                    base_train = train[base_column].fillna(fallback).to_numpy(dtype=float)
                    base_validation = validation[base_column].fillna(fallback).to_numpy(dtype=float)
                    base_test = test[base_column].fillna(fallback).to_numpy(dtype=float)
                    residual_model = lgb.LGBMRegressor(
                        objective="huber", n_estimators=args.num_boost_round, learning_rate=0.035,
                        num_leaves=48, min_child_samples=100, subsample=0.9, colsample_bytree=0.9,
                        random_state=args.seed + 91, n_jobs=-1, verbosity=-1,
                    )
                    weights = np.where(train[tail].astype(bool), 4.0, 1.0)
                    residual_model.fit(
                        train_x, train[raw].to_numpy(dtype=float) - base_train, sample_weight=weights,
                        eval_set=[(validation_x, validation[raw].to_numpy(dtype=float) - base_validation)],
                        eval_metric="l2", callbacks=[lgb.early_stopping(30, verbose=False)],
                    )
                    residual_validation = np.clip(base_validation + residual_model.predict(validation_x), 0, 1)
                    residual_test = np.clip(base_test + residual_model.predict(test_x), 0, 1)
                    residual_probability = calibrated_probability(validation[tail], residual_validation, residual_test)
                    metrics, half_width = evaluate_variant(
                        fold, target, "rts_profile_residual", train, validation, test,
                        residual_validation, residual_test, residual_probability, len(structural) + 1,
                    )
                    metric_rows.append(metrics)
                    print(f"fold={fold['fold']} target=rts ablation=rts_profile_residual AP={metrics['ap']:.4f}", flush=True)

                    full_validation = prediction_cache["static_rolling_dynamic_topology_route"]["validation"]
                    full_test = prediction_cache["static_rolling_dynamic_topology_route"]["test"]
                    alphas = np.linspace(0, 1, 21)
                    val_y = validation[tail].astype(int).to_numpy()
                    scores = []
                    for alpha in alphas:
                        fused = alpha * full_validation + (1 - alpha) * residual_validation
                        scores.append(average_precision_score(val_y, fused) if len(np.unique(val_y)) == 2 else -mean_absolute_error(validation[raw], fused))
                    alpha = float(alphas[int(np.nanargmax(scores))])
                    fused_validation = np.clip(alpha * full_validation + (1 - alpha) * residual_validation, 0, 1)
                    fused_test = np.clip(alpha * full_test + (1 - alpha) * residual_test, 0, 1)
                    fused_probability = calibrated_probability(validation[tail], fused_validation, fused_test)
                    metrics, half_width = evaluate_variant(
                        fold, target, "rts_validation_selected_hybrid", train, validation, test,
                        fused_validation, fused_test, fused_probability,
                        len(sets["static_rolling_dynamic_topology_route"]) + len(structural) + 1,
                    )
                    metrics["fusion_tabular_alpha"] = alpha
                    metric_rows.append(metrics)
                    link_column = "planned_link_id" if "planned_link_id" in test.columns else "route_link_id"
                    seq_column = "planned_link_seq" if "planned_link_seq" in test.columns else "route_link_seq"
                    part = test[["order_id", "date", link_column, seq_column]].copy()
                    part = part.rename(columns={link_column: "planned_link_id", seq_column: "planned_link_seq"})
                    part["fold"] = fold["fold"]
                    part["target"] = target
                    part["ablation"] = "rts_validation_selected_hybrid"
                    part["true_raw"] = test[raw].to_numpy(dtype=float)
                    part["true_tail"] = test[tail].astype(int).to_numpy()
                    part["pred_raw"] = fused_test
                    part["pred_tail_probability"] = fused_probability
                    part["pred_uncertainty_half_width_90"] = half_width
                    prediction_parts.append(part)
                    print(f"fold={fold['fold']} target=rts ablation=hybrid alpha={alpha:.2f} AP={metrics['ap']:.4f}", flush=True)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output_root / "rolling_fair_metrics.csv", index=False)
    summary = metrics.groupby(["target", "ablation"], as_index=False).agg(
        folds=("fold", "nunique"), auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        ap_mean=("ap", "mean"), ap_std=("ap", "std"), spearman_mean=("spearman", "mean"),
        lift_top5_mean=("lift_top5", "mean"), order_lift_top10_mean=("order_lift_top10", "mean"),
        brier_mean=("brier_calibrated", "mean"), ece_mean=("ece_calibrated", "mean"),
        interval_coverage_mean=("interval_90_coverage", "mean"), interval_width_mean=("interval_mean_width", "mean"),
    )
    summary.to_csv(args.output_root / "rolling_fair_summary.csv", index=False)
    if prediction_parts:
        predictions = pd.concat(prediction_parts, ignore_index=True)
        predictions.to_parquet(args.output_root / "rolling_oof_predictions.parquet", index=False, compression="zstd")
        block_bootstrap(predictions, args.bootstrap_reps, args.seed).to_csv(args.output_root / "rolling_fair_bootstrap_ci.csv", index=False)
    lines = ["# Stage2 three-fold rolling fair evaluation", "", summary.to_markdown(index=False, floatfmt=".4f"), "",
             "All ablations use the same realized-label-available planned-route rows within each fold. Tail probabilities are isotonic-calibrated on the validation day; uncertainty intervals use validation-normalized conformal residuals."]
    (args.output_root / "rolling_fair_report.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {"folds": folds, "targets": args.targets, "max_train_rows": args.max_train_rows, "bootstrap_reps": args.bootstrap_reps, "complete": True}
    (args.output_root / "rolling_fair_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
