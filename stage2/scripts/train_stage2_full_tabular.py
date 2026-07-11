"""Train strong full/large-sample tabular Stage2 baselines.

This script is intended as an upper-bound tabular probe relative to the 1M-row
single-target baseline. It keeps the Stage2 data contract feature whitelist,
adds train-only historical profile features, and supports target-specific
tail-weighted LightGBM training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage2_model_utils import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    FORBIDDEN_LEAKAGE_COLUMNS,
    ID_COLUMNS,
    TARGETS,
    TARGET_ORDER,
    evaluate_predictions,
    prepare_tabular_features,
    safe_json_float,
    unique_existing_columns,
)


PROFILE_KEYS = ["link_id", "time_bin"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("stage2/output/link_dataset"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/deep_baselines/full_tabular"))
    parser.add_argument("--prediction-root", type=Path, default=Path("stage2/output/deep_predictions"))
    parser.add_argument("--max-train-rows", default="3000000", help="'all' or integer rows per target")
    parser.add_argument("--max-train-orders", type=int, default=None, help="optional deterministic order budget shared by all targets")
    parser.add_argument(
        "--profile-scope",
        choices=["all_train", "sampled_train_orders"],
        default="all_train",
        help="whether train-only historical profiles use all train rows or only the sampled order budget",
    )
    parser.add_argument("--num-boost-round", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--tail-weight", type=float, default=4.0)
    parser.add_argument("--tail-cutoff", type=float, default=0.90)
    parser.add_argument("--min-profile-count", type=int, default=20)
    parser.add_argument("--max-eval-rank-rows", type=int, default=1_000_000)
    parser.add_argument("--targets", nargs="+", default=TARGET_ORDER, choices=TARGET_ORDER)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def parse_max_rows(value: str) -> int | None:
    return None if value.lower() == "all" else int(value)


def collect_order_budget(path: Path, max_orders: int | None, seed: int) -> set | None:
    if max_orders is None:
        return None
    parquet = pq.ParquetFile(path)
    rng = np.random.default_rng(seed)
    selected: list = []
    selected_lookup: set = set()
    # Reservoir sample unique order ids without loading the whole split.
    seen = 0
    for group_no in range(parquet.metadata.num_row_groups):
        frame = parquet.read_row_group(group_no, columns=["order_id"]).to_pandas()
        for order_id in pd.unique(frame["order_id"].dropna()):
            if order_id in selected_lookup:
                continue
            seen += 1
            if len(selected) < max_orders:
                selected.append(order_id)
                selected_lookup.add(order_id)
                continue
            replace_at = int(rng.integers(0, seen))
            if replace_at < max_orders:
                selected_lookup.remove(selected[replace_at])
                selected[replace_at] = order_id
                selected_lookup.add(order_id)
    return selected_lookup


def selected_columns(path: Path, target: str, mask: str, high: str, include_ids: bool = False) -> list[str]:
    desired = FEATURE_COLUMNS + PROFILE_KEYS + [target, mask, high]
    if include_ids:
        desired = ID_COLUMNS + desired
    return unique_existing_columns(path, desired)


def fit_profiles(path: Path, output_root: Path, min_count: int, order_filter: set | None = None) -> pd.DataFrame:
    columns = PROFILE_KEYS[:]
    if order_filter is not None:
        columns.append("order_id")
    for target_name in TARGET_ORDER:
        target_col, mask_col, _ = TARGETS[target_name]
        columns.extend([target_col, mask_col])
    columns = unique_existing_columns(path, columns)
    parquet = pq.ParquetFile(path)
    accumulators: dict[str, pd.DataFrame | None] = {target_name: None for target_name in TARGET_ORDER}
    for group_no in range(parquet.metadata.num_row_groups):
        frame = parquet.read_row_group(group_no, columns=columns).to_pandas()
        if order_filter is not None:
            frame = frame[frame["order_id"].isin(order_filter)]
            if frame.empty:
                continue
        for target_name in TARGET_ORDER:
            target_col, mask_col, _ = TARGETS[target_name]
            valid = frame[mask_col].fillna(False) & frame[target_col].notna()
            if not valid.any():
                continue
            summary = frame.loc[valid, PROFILE_KEYS + [target_col]].copy()
            summary["sum"] = summary[target_col].astype(float)
            summary["count"] = 1
            summary = summary.groupby(PROFILE_KEYS, as_index=False)[["sum", "count"]].sum()
            if accumulators[target_name] is None:
                accumulators[target_name] = summary
            else:
                accumulators[target_name] = pd.concat([accumulators[target_name], summary], ignore_index=True).groupby(
                    PROFILE_KEYS, as_index=False
                )[["sum", "count"]].sum()
    parts = []
    for target_name, summary in accumulators.items():
        if summary is None or summary.empty:
            continue
        summary = summary[summary["count"] >= min_count].copy()
        summary[f"profile_{target_name.lower()}"] = summary["sum"] / summary["count"]
        summary[f"profile_{target_name.lower()}_count"] = summary["count"]
        parts.append(summary[PROFILE_KEYS + [f"profile_{target_name.lower()}", f"profile_{target_name.lower()}_count"]])
    if not parts:
        return pd.DataFrame()
    result = parts[0]
    for part in parts[1:]:
        result = result.merge(part, on=PROFILE_KEYS, how="outer")
    output_root.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_root / "train_historical_profiles.parquet", index=False, compression="zstd")
    return result


def add_profiles(frame: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    if profiles.empty:
        return frame
    return frame.merge(profiles, on=PROFILE_KEYS, how="left")


def profile_feature_columns(profiles: pd.DataFrame) -> list[str]:
    return [column for column in profiles.columns if column not in PROFILE_KEYS]


def read_training_sample(
    path: Path,
    target_name: str,
    profiles: pd.DataFrame,
    max_rows: int | None,
    seed: int,
    order_filter: set | None = None,
) -> pd.DataFrame:
    target_col, mask_col, high_col = TARGETS[target_name]
    parquet = pq.ParquetFile(path)
    columns = selected_columns(path, target_col, mask_col, high_col, include_ids=order_filter is not None)
    if max_rows is None:
        parts = []
        for group_no in range(parquet.metadata.num_row_groups):
            frame = parquet.read_row_group(group_no, columns=columns).to_pandas()
            if order_filter is not None:
                frame = frame[frame["order_id"].isin(order_filter)]
            frame = frame[frame[mask_col].fillna(False) & frame[target_col].notna()]
            if not frame.empty:
                parts.append(add_profiles(frame, profiles))
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)

    per_group = max(1, int(np.ceil(max_rows / max(parquet.metadata.num_row_groups, 1))))
    parts = []
    for group_no in range(parquet.metadata.num_row_groups):
        frame = parquet.read_row_group(group_no, columns=columns).to_pandas()
        if order_filter is not None:
            frame = frame[frame["order_id"].isin(order_filter)]
        frame = frame[frame[mask_col].fillna(False) & frame[target_col].notna()]
        if frame.empty:
            continue
        take = min(per_group, len(frame))
        frame = frame.sample(n=take, random_state=seed + group_no) if take < len(frame) else frame
        parts.append(add_profiles(frame, profiles))
    sample = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
    if len(sample) > max_rows:
        sample = sample.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    return sample


def read_eval_split(path: Path, target_name: str, profiles: pd.DataFrame) -> pd.DataFrame:
    target_col, mask_col, high_col = TARGETS[target_name]
    frame = pd.read_parquet(path, columns=selected_columns(path, target_col, mask_col, high_col, include_ids=True))
    return add_profiles(frame, profiles)


def target_params(target_name: str, args: argparse.Namespace) -> dict:
    base = {
        "objective": "regression",
        "n_estimators": args.num_boost_round,
        "learning_rate": args.learning_rate,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "min_child_samples": 150,
        "random_state": args.seed,
        "n_jobs": -1,
        "verbosity": -1,
    }
    if target_name == "IIS":
        base.update({"num_leaves": 96, "min_child_samples": 80})
    elif target_name == "RTS":
        base.update({"num_leaves": 48, "learning_rate": min(args.learning_rate, 0.03), "min_child_samples": 200})
    else:
        base.update({"num_leaves": 80})
    return base


def prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[[column for column in ID_COLUMNS if column in frame.columns]].copy()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.prediction_root.mkdir(parents=True, exist_ok=True)
    max_rows = parse_max_rows(args.max_train_rows)
    train_path = args.dataset_root / "train.parquet"
    order_budget = collect_order_budget(train_path, args.max_train_orders, args.seed)

    profile_order_filter = order_budget if args.profile_scope == "sampled_train_orders" else None
    profiles = fit_profiles(train_path, args.output_root, args.min_profile_count, profile_order_filter)
    features = FEATURE_COLUMNS + profile_feature_columns(profiles)
    metrics_rows = []
    manifest: dict[str, object] = {
        "model": "lightgbm_full_tabular_tail_weighted",
        "max_train_rows": args.max_train_rows,
        "max_train_orders": args.max_train_orders,
        "profile_scope": args.profile_scope,
        "tail_weight": args.tail_weight,
        "tail_cutoff": args.tail_cutoff,
        "features": features,
        "historical_profiles": "train_only_link_id_time_bin",
        "forbidden_leakage_excluded": FORBIDDEN_LEAKAGE_COLUMNS,
        "requested_targets": args.targets,
        "targets": {},
    }
    predictions: dict[str, pd.DataFrame | None] = {"validation": None, "test": None}

    for target_name in args.targets:
        target_col, mask_col, high_col = TARGETS[target_name]
        print(f"full tabular training {target_name}", flush=True)
        train = read_training_sample(train_path, target_name, profiles, max_rows, args.seed, order_budget)
        x_train = prepare_tabular_features(train, features)
        y_train = train[target_col].astype(float).clip(0, 1)
        weights = np.ones(len(train), dtype="float32")
        weights[y_train.to_numpy() >= args.tail_cutoff] = args.tail_weight

        validation = read_eval_split(args.dataset_root / "validation.parquet", target_name, profiles)
        val_valid = validation[mask_col].fillna(False) & validation[target_col].notna()
        model = lgb.LGBMRegressor(**target_params(target_name, args))
        model.fit(
            x_train,
            y_train,
            sample_weight=weights,
            eval_set=[(prepare_tabular_features(validation.loc[val_valid], features), validation.loc[val_valid, target_col].astype(float).clip(0, 1))],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(40, verbose=False)],
        )

        importances = pd.DataFrame({
            "feature": x_train.columns,
            "importance_split": model.booster_.feature_importance(importance_type="split"),
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
        }).sort_values("importance_gain", ascending=False)
        importances.to_csv(args.output_root / f"feature_importance_{target_name}.csv", index=False)

        target_manifest = {"train_rows": int(len(train)), "splits": {}, "best_iteration": int(model.best_iteration_ or args.num_boost_round)}
        for split in ["validation", "test"]:
            frame = validation if split == "validation" else read_eval_split(args.dataset_root / f"{split}.parquet", target_name, profiles)
            valid = frame[mask_col].fillna(False) & frame[target_col].notna()
            pred = np.clip(model.predict(prepare_tabular_features(frame, features), num_iteration=model.best_iteration_), 0, 1).astype("float32")
            metrics = evaluate_predictions(
                frame.loc[valid, target_col].to_numpy(dtype=float),
                pred[valid.to_numpy()],
                frame.loc[valid, high_col].to_numpy(dtype=bool),
                args.max_eval_rank_rows,
                args.seed,
            )
            metrics.update({"target": target_name, "split": split, "train_rows": int(len(train))})
            metrics_rows.append(metrics)
            target_manifest["splits"][split] = metrics
            if predictions[split] is None:
                predictions[split] = prediction_frame(frame)
            predictions[split][f"pred_{target_name.lower()}"] = pred
            predictions[split][f"target_{target_name.lower()}"] = frame[target_col].to_numpy()
            predictions[split][f"{target_name.lower()}_valid"] = frame[mask_col].to_numpy()
        manifest["targets"][target_name] = target_manifest

    metrics_table = pd.DataFrame(metrics_rows)
    metrics_table.to_csv(args.output_root / "full_tabular_metrics_by_target.csv", index=False)
    (args.output_root / "full_tabular_metrics.json").write_text(
        json.dumps(safe_json_float(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for split, frame in predictions.items():
        if frame is not None:
            frame.to_parquet(args.prediction_root / f"full_tabular_{split}.parquet", index=False, compression="zstd")
    print(metrics_table.to_string(index=False))


if __name__ == "__main__":
    main()
