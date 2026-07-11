"""Train Stage2 single-target LightGBM baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score


TARGETS = {
    "LCS": ("target_lcs_pct", "lcs_valid", "target_high_lcs_90"),
    "IIS": ("target_iis_pct", "iis_valid", "target_high_iis_90"),
    "RTS": ("target_rts_pct", "rts_valid", "target_high_rts_90"),
    "PMIS": ("target_pmis_pct", "pmis_valid", "target_high_pmis_90"),
}

ID_COLUMNS = ["order_id", "driver_id", "date", "link_id", "link_seq"]

FEATURE_COLUMNS = [
    "time_bin", "hour", "weekday_type", "peak_offpeak", "is_weekend",
    "road_class", "link_length_m", "curvature_deg_per_km_link", "minor_road",
    "endpoint_degree", "link_fragmentation", "area_grid", "gns_pct_link",
    "activity_intensity_index",
    "poi_density_100m_school", "poi_density_100m_hospital", "poi_density_100m_commercial",
    "poi_density_100m_restaurant", "poi_density_100m_transit", "poi_density_100m_bus_stop",
    "poi_density_100m_residential", "poi_density_100m_office", "poi_density_100m_scenic",
    "poi_density_100m_parking",
    "link_seq", "route_link_count", "position_ratio", "distance_to_destination_ratio",
]

CATEGORICAL_COLUMNS = ["weekday_type", "peak_offpeak", "road_class", "area_grid"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("stage2/output/link_dataset"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/baselines"))
    parser.add_argument("--max-train-rows", type=int, default=1_000_000)
    parser.add_argument("--max-eval-rank-rows", type=int, default=1_000_000)
    parser.add_argument("--num-boost-round", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def available_columns(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema_arrow.names


def selected_columns(path: Path, target: str, mask: str, high: str, include_ids: bool = False) -> list[str]:
    desired = FEATURE_COLUMNS + [target, mask, high]
    if include_ids:
        desired = ID_COLUMNS + desired
    available = set(available_columns(path))
    result = []
    for column in desired:
        if column in available and column not in result:
            result.append(column)
    return result


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame[[column for column in FEATURE_COLUMNS if column in frame.columns]].copy()
    for column in CATEGORICAL_COLUMNS:
        if column in features.columns:
            features[column] = features[column].astype("category")
    for column in features.columns:
        if column not in CATEGORICAL_COLUMNS:
            if features[column].dtype == "bool":
                features[column] = features[column].astype("int8")
            else:
                features[column] = pd.to_numeric(features[column], errors="coerce").astype("float32")
    return features


def read_training_sample(path: Path, target: str, mask: str, high: str, max_rows: int, seed: int) -> pd.DataFrame:
    parquet = pq.ParquetFile(path)
    columns = selected_columns(path, target, mask, high, include_ids=False)
    per_group = max(1, int(np.ceil(max_rows / max(parquet.metadata.num_row_groups, 1))))
    parts = []
    for group_no in range(parquet.metadata.num_row_groups):
        frame = parquet.read_row_group(group_no, columns=columns).to_pandas()
        frame = frame[frame[mask].fillna(False) & frame[target].notna()]
        if frame.empty:
            continue
        take = min(per_group, len(frame))
        parts.append(frame.sample(n=take, random_state=seed + group_no) if take < len(frame) else frame)
    sample = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
    if len(sample) > max_rows:
        sample = sample.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    return sample


def read_eval_split(path: Path, target: str, mask: str, high: str) -> pd.DataFrame:
    return pd.read_parquet(path, columns=selected_columns(path, target, mask, high, include_ids=True))


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = len(y_true)
    if total == 0:
        return float("nan")
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        if high == 1:
            mask = (y_prob >= low) & (y_prob <= high)
        else:
            mask = (y_prob >= low) & (y_prob < high)
        if mask.any():
            ece += mask.mean() * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return float(ece)


def rank_metrics(y: np.ndarray, pred: np.ndarray, rank_rows: int | None, seed: int) -> dict[str, float]:
    frame = pd.DataFrame({"y": y, "pred": pred}).dropna()
    if rank_rows and len(frame) > rank_rows:
        frame = frame.sample(n=rank_rows, random_state=seed)
    high = frame.y.ge(0.90)
    base_rate = float(high.mean())
    result: dict[str, float] = {"high_rate": base_rate}
    for share, label in [(0.10, "top10"), (0.05, "top5")]:
        n = max(1, int(len(frame) * share))
        top = frame.nlargest(n, "pred")
        precision = float(top.y.ge(0.90).mean())
        recall = float(top.y.ge(0.90).sum() / max(high.sum(), 1))
        result[f"precision_at_{label}pct"] = precision
        result[f"recall_at_{label}pct"] = recall
        result[f"{label}_lift"] = precision / base_rate if base_rate > 0 else float("nan")
    return result


def evaluate(y: np.ndarray, pred: np.ndarray, high: np.ndarray, rank_rows: int | None, seed: int) -> dict[str, float]:
    valid = np.isfinite(y) & np.isfinite(pred)
    y = y[valid]
    pred = np.clip(pred[valid], 0, 1)
    high = high[valid].astype(bool)
    metrics = {
        "rows": int(len(y)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred, squared=False)),
        "pearson": float(pd.Series(y).corr(pd.Series(pred), method="pearson")),
        "spearman": float(pd.Series(y).corr(pd.Series(pred), method="spearman")),
        **rank_metrics(y, pred, rank_rows, seed),
    }
    if high.any() and (~high).any():
        metrics["auc"] = float(roc_auc_score(high, pred))
        metrics["ap"] = float(average_precision_score(high, pred))
        metrics["brier"] = float(brier_score_loss(high, pred))
        metrics["ece"] = ece_score(high.astype(float), pred)
    else:
        metrics.update({"auc": float("nan"), "ap": float("nan"), "brier": float("nan"), "ece": float("nan")})
    return metrics


def prediction_frame(split_frame: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in ID_COLUMNS if column in split_frame.columns]
    return split_frame[columns].copy()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    metrics_rows = []
    metrics_json: dict[str, object] = {
        "dataset_root": str(args.dataset_root),
        "max_train_rows": args.max_train_rows,
        "model": "lightgbm_lgbmregressor_single_target",
        "targets": {},
        "feature_columns": FEATURE_COLUMNS,
        "forbidden_leakage_excluded": [
            "travel_time_sec", "observed_distance_m", "reference_travel_time_sec", "excess_time_ratio",
            "tail_delay_ratio", "low_speed_ratio_on_poi_link", "stop_time_on_poi_link", "delay_on_poi_link",
            "traversal_quality", "observed_or_inferred", "low_quality_flag",
        ],
    }

    prediction_outputs: dict[str, pd.DataFrame | None] = {"validation": None, "test": None}

    for target_name, (target_col, mask_col, high_col) in TARGETS.items():
        print(f"training {target_name}", flush=True)
        train_path = args.dataset_root / "train.parquet"
        train = read_training_sample(train_path, target_col, mask_col, high_col, args.max_train_rows, args.seed)
        x_train = prepare_features(train)
        y_train = train[target_col].astype(float).clip(0, 1)

        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=args.num_boost_round,
            learning_rate=args.learning_rate,
            num_leaves=64,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=100,
            random_state=args.seed,
            n_jobs=-1,
            verbosity=-1,
        )

        validation = read_eval_split(args.dataset_root / "validation.parquet", target_col, mask_col, high_col)
        val_valid = validation[mask_col].fillna(False) & validation[target_col].notna()
        x_val = prepare_features(validation.loc[val_valid])
        y_val = validation.loc[val_valid, target_col].astype(float).clip(0, 1)
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )

        target_metrics = {"train_sample_rows": int(len(train)), "splits": {}}
        importances = pd.DataFrame({
            "feature": x_train.columns,
            "importance_split": model.booster_.feature_importance(importance_type="split"),
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
        }).sort_values("importance_gain", ascending=False)
        importances.to_csv(args.output_root / f"feature_importance_{target_name}.csv", index=False)

        for split in ["validation", "test"]:
            split_frame = validation if split == "validation" else read_eval_split(args.dataset_root / f"{split}.parquet", target_col, mask_col, high_col)
            valid = split_frame[mask_col].fillna(False) & split_frame[target_col].notna()
            preds = np.full(len(split_frame), np.nan, dtype="float32")
            preds[:] = np.clip(model.predict(prepare_features(split_frame), num_iteration=model.best_iteration_), 0, 1)
            metrics = evaluate(
                split_frame.loc[valid, target_col].to_numpy(dtype=float),
                preds[valid.to_numpy()],
                split_frame.loc[valid, high_col].to_numpy(dtype=bool),
                args.max_eval_rank_rows,
                args.seed,
            )
            metrics.update({"target": target_name, "split": split, "train_sample_rows": int(len(train))})
            metrics_rows.append(metrics)
            target_metrics["splits"][split] = metrics

            if prediction_outputs[split] is None:
                prediction_outputs[split] = prediction_frame(split_frame)
            prediction_outputs[split][f"pred_{target_name.lower()}"] = preds
            prediction_outputs[split][f"target_{target_name.lower()}"] = split_frame[target_col].to_numpy()
            prediction_outputs[split][f"{target_name.lower()}_valid"] = split_frame[mask_col].to_numpy()

        metrics_json["targets"][target_name] = target_metrics

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(args.output_root / "baseline_metrics_by_target.csv", index=False)
    (args.output_root / "baseline_metrics.json").write_text(
        json.dumps(metrics_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for split, frame in prediction_outputs.items():
        if frame is not None:
            frame.to_parquet(args.output_root / f"prediction_sample_{split}.parquet", index=False, compression="zstd")
            frame.to_parquet(args.output_root / f"predictions_{split}.parquet", index=False, compression="zstd")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
