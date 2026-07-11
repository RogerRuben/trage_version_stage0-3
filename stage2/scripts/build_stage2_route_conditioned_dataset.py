"""Build the route-conditioned Stage2 datasets.

The input is the actual matched route product with estimated-entry-time lagged
state.  The route itself is treated as the assigned service route proxy.  The
estimated-time output is the main Stage2 product; the oracle-time output exposes
actual link entry time only for upper-bound diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TARGET_PREFIXES = ("target_lcs_", "target_iis_", "target_rts_", "target_pmis_")
FORBIDDEN_ESTIMATED_FEATURES = {
    "actual_link_entry_time",
    "actual_link_exit_time",
    "travel_time_sec",
    "mean_speed_mps",
    "median_speed_mps",
    "min_speed_mps",
    "low_speed_time_sec",
    "low_speed_ratio",
    "stop_time_sec",
    "stop_count",
    "stop_duration_ratio",
    "speed_cv",
    "accel_volatility",
    "point_count",
    "observed_distance_m",
}
AUDIT_ONLY_COLUMNS = {
    "actual_link_id",
    "actual_link_seq",
    "mean_match_dist",
    "p90_match_dist",
    "traversal_quality",
    "matcher_version",
    "link_occurrence",
    "realized_label_available",
    "strict_availability_check",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("stage2/output/actual_route_oracle_causal_dataset"))
    parser.add_argument("--fold-config", type=Path, default=Path("rolling_threefold_config.json"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/route_conditioned_dataset"))
    parser.add_argument("--fold", type=int, default=1, help="Fold used for the top-level train/validation/test aliases.")
    parser.add_argument("--dates", default="", help="Optional comma-separated date override.")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def load_folds(path: Path) -> tuple[list[dict], list[str]]:
    config = json.loads(path.read_text(encoding="utf-8"))
    folds = config.get("folds", [])
    dates = sorted({
        date
        for fold in folds
        for date in list(fold.get("train_dates", [])) + [fold.get("validation_date"), fold.get("test_date")]
        if date
    })
    return folds, dates


def feature_availability_timestamp(frame: pd.DataFrame) -> pd.Series:
    columns = [column for column in frame.columns if "availability_timestamp" in column]
    if not columns:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    missing_value = np.iinfo("int64").min
    values = []
    for column in columns:
        series = pd.to_datetime(frame[column], utc=True, errors="coerce")
        raw = series.view("int64").to_numpy(copy=True)
        raw[series.isna().to_numpy()] = missing_value
        values.append(raw)
    maximum = np.vstack(values).max(axis=0)
    result = pd.to_datetime(pd.Series(maximum, index=frame.index).mask(maximum == missing_value), utc=True)
    return result


def add_common_route_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["route_link_id"] = frame.get("planned_link_id", frame.get("actual_link_id")).astype(str)
    frame["route_link_seq"] = frame.get("planned_link_seq", frame.get("actual_link_seq")).astype("int32")
    frame["route_link_count"] = frame.get("planned_route_link_count", frame.groupby("order_id")["route_link_id"].transform("size")).astype("int32")
    if "planned_link_length_m" in frame:
        frame["route_link_length_m"] = pd.to_numeric(frame["planned_link_length_m"], errors="coerce")
    elif "link_length_m" in frame:
        frame["route_link_length_m"] = pd.to_numeric(frame["link_length_m"], errors="coerce")
    frame["assigned_route_proxy"] = "map_matched_completed_route"
    frame["route_conditioned_setting"] = "given_route"
    frame["feature_availability_timestamp"] = feature_availability_timestamp(frame)
    return frame


def parse_timestamp(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="s", utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


def add_time_columns(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    frame = frame.copy()
    if mode == "estimated":
        prediction_time = parse_timestamp(frame["estimated_link_entry_time"])
        frame["time_mode"] = "estimated_time"
    else:
        prediction_time = parse_timestamp(frame["actual_link_entry_time"])
        frame["actual_link_entry_time"] = prediction_time
        frame["time_mode"] = "oracle_time_upper_bound"
        local_actual = prediction_time.dt.tz_convert("Asia/Shanghai")
        frame["actual_time_bin"] = (local_actual.dt.hour * 2 + (local_actual.dt.minute >= 30).astype(int)).astype("Int16")
        frame["actual_hour"] = local_actual.dt.hour.astype("Int16")
        frame["actual_is_weekend"] = local_actual.dt.dayofweek.ge(5).astype("Int8")
    frame["prediction_link_entry_time"] = prediction_time
    frame["prediction_time_bin"] = frame.get("estimated_time_bin")
    local = prediction_time.dt.tz_convert("Asia/Shanghai")
    frame["prediction_hour"] = local.dt.hour.astype("Int16")
    frame["prediction_weekday"] = local.dt.dayofweek.astype("Int16")
    frame["prediction_is_weekend"] = local.dt.dayofweek.ge(5).astype("Int8")
    frame["route_conditioned_deployable"] = mode == "estimated"
    if "strict_availability_check" in frame:
        frame["route_conditioned_time_check"] = frame["strict_availability_check"].astype(bool) & (
            frame["feature_availability_timestamp"].isna() | frame["feature_availability_timestamp"].lt(prediction_time)
        )
    else:
        frame["route_conditioned_time_check"] = frame["feature_availability_timestamp"].isna() | frame["feature_availability_timestamp"].lt(prediction_time)
    return frame


def target_columns(columns: Iterable[str]) -> list[str]:
    return [column for column in columns if column.startswith(TARGET_PREFIXES)]


def select_columns(frame: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    id_columns = [
        column for column in [
            "order_id", "driver_id", "date", "route_link_id", "route_link_seq",
            "route_link_count", "assigned_route_proxy", "route_conditioned_setting",
            "time_mode", "route_conditioned_deployable",
        ] if column in frame.columns
    ]
    time_columns = [
        column for column in [
            "dispatch_time", "origin_timestamp", "estimated_link_entry_time",
            "prediction_link_entry_time", "prediction_time_bin", "prediction_hour",
            "prediction_weekday", "prediction_is_weekend", "feature_availability_timestamp",
            "route_conditioned_time_check",
        ] if column in frame.columns
    ]
    if mode == "oracle":
        time_columns += [column for column in ["actual_link_entry_time", "actual_time_bin", "actual_hour", "actual_is_weekend"] if column in frame.columns]
    route_columns = [
        column for column in [
            "position_ratio", "distance_to_destination_ratio", "route_link_length_m",
            "estimated_link_travel_time_sec", "road_class", "area_grid",
            "endpoint_degree", "link_fragmentation", "minor_road",
            "activity_intensity_index",
        ] if column in frame.columns
    ]
    poi_columns = [column for column in frame.columns if column.startswith("poi_density_100m_")]
    rolling_columns = [column for column in frame.columns if column.startswith("rolling_")]
    state_columns = [
        column for column in frame.columns
        if any(column.startswith(prefix) for prefix in ["link_recent_", "area_recent_", "network_recent_", "upstream_recent_", "downstream_recent_"])
        or column.endswith("neighbor_count")
    ]
    labels = target_columns(frame.columns)
    audit = [column for column in AUDIT_ONLY_COLUMNS if column in frame.columns]
    if mode == "oracle":
        audit += [column for column in ["actual_link_exit_time", "travel_time_sec"] if column in frame.columns]
    feature_columns = [
        column for column in time_columns + route_columns + poi_columns + rolling_columns + state_columns
        if column not in labels and (mode == "oracle" or column not in FORBIDDEN_ESTIMATED_FEATURES)
    ]
    ordered = []
    for column in id_columns + feature_columns + labels + audit:
        if column in frame.columns and column not in ordered:
            ordered.append(column)
    metadata = {
        "id_columns": id_columns,
        "feature_columns": [column for column in feature_columns if column in ordered],
        "target_columns": [column for column in labels if column in ordered],
        "audit_only_columns": [column for column in audit if column in ordered],
        "forbidden_estimated_feature_columns": sorted(FORBIDDEN_ESTIMATED_FEATURES),
    }
    return frame[ordered].copy(), metadata


def normalize_timestamp_precision(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            frame[column] = frame[column].dt.round("us")
    return frame


def write_day(source: Path, estimated_path: Path, oracle_path: Path, skip_existing: bool) -> dict:
    if skip_existing and estimated_path.exists() and oracle_path.exists():
        try:
            estimated = pd.read_parquet(estimated_path, columns=["order_id", "route_conditioned_time_check"])
            oracle = pd.read_parquet(oracle_path, columns=["order_id", "route_conditioned_time_check"])
            return {
                "estimated_rows": len(estimated),
                "oracle_rows": len(oracle),
                "estimated_orders": int(estimated["order_id"].nunique()),
                "oracle_orders": int(oracle["order_id"].nunique()),
                "estimated_time_check_ratio": float(estimated["route_conditioned_time_check"].mean()),
                "oracle_time_check_ratio": float(oracle["route_conditioned_time_check"].mean()),
                "skipped": True,
            }
        except Exception:
            pass
    frame = pd.read_parquet(source)
    frame = add_common_route_columns(frame)
    estimated, estimated_meta = select_columns(add_time_columns(frame, "estimated"), "estimated")
    oracle, oracle_meta = select_columns(add_time_columns(frame, "oracle"), "oracle")
    estimated = normalize_timestamp_precision(estimated)
    oracle = normalize_timestamp_precision(oracle)
    estimated_path.parent.mkdir(parents=True, exist_ok=True)
    oracle_path.parent.mkdir(parents=True, exist_ok=True)
    estimated.to_parquet(estimated_path, index=False, compression="zstd")
    oracle.to_parquet(oracle_path, index=False, compression="zstd")
    return {
        "estimated_rows": len(estimated),
        "oracle_rows": len(oracle),
        "estimated_orders": int(estimated["order_id"].nunique()),
        "oracle_orders": int(oracle["order_id"].nunique()),
        "estimated_time_check_ratio": float(estimated["route_conditioned_time_check"].mean()),
        "oracle_time_check_ratio": float(oracle["route_conditioned_time_check"].mean()),
        "estimated_feature_count": len(estimated_meta["feature_columns"]),
        "oracle_feature_count": len(oracle_meta["feature_columns"]),
        "estimated_target_count": len(estimated_meta["target_columns"]),
        "oracle_target_count": len(oracle_meta["target_columns"]),
        "skipped": False,
    }


def concat_split(day_root: Path, dates: list[str], output_file: Path) -> dict:
    parts = [pd.read_parquet(day_root / f"day={date}.parquet") for date in dates]
    frame = pd.concat(parts, ignore_index=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_file, index=False, compression="zstd")
    return {"rows": len(frame), "orders": int(frame["order_id"].nunique()), "dates": dates}


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    folds, config_dates = load_folds(args.fold_config)
    dates = [part.strip() for part in args.dates.split(",") if part.strip()] if args.dates else config_dates
    estimated_day_root = args.output_root / "estimated_time_daily"
    oracle_day_root = args.output_root / "oracle_time_daily"
    manifest = {
        "route_conditioned_contract": "matched actual route is the assigned/planned service route proxy",
        "source_root": str(args.source_root),
        "estimated_time_deployable": True,
        "oracle_time_deployable": False,
        "days": {},
        "folds": {},
    }
    first_meta_written = False
    for date in dates:
        source = args.source_root / f"day={date}.parquet"
        if not source.exists():
            raise FileNotFoundError(source)
        stats = write_day(
            source,
            estimated_day_root / f"day={date}.parquet",
            oracle_day_root / f"day={date}.parquet",
            args.skip_existing,
        )
        manifest["days"][date] = stats
        if not first_meta_written:
            sample_est = pd.read_parquet(estimated_day_root / f"day={date}.parquet")
            sample_oracle = pd.read_parquet(oracle_day_root / f"day={date}.parquet")
            _, estimated_meta = select_columns(sample_est, "estimated")
            _, oracle_meta = select_columns(sample_oracle, "oracle")
            (args.output_root / "route_conditioned_estimated_time_schema.json").write_text(json.dumps(estimated_meta, indent=2), encoding="utf-8")
            (args.output_root / "route_conditioned_oracle_time_schema.json").write_text(json.dumps(oracle_meta, indent=2), encoding="utf-8")
            first_meta_written = True
        print(f"route-conditioned day={date} {stats}", flush=True)
    for fold in folds:
        fold_id = int(fold["fold"])
        fold_dir = args.output_root / f"fold={fold_id}"
        split_dates = {
            "train": fold["train_dates"],
            "validation": [fold["validation_date"]],
            "test": [fold["test_date"]],
        }
        manifest["folds"][str(fold_id)] = {}
        for split, split_values in split_dates.items():
            est_stats = concat_split(
                estimated_day_root,
                split_values,
                fold_dir / f"route_conditioned_estimated_time_{split}.parquet",
            )
            oracle_stats = concat_split(
                oracle_day_root,
                split_values,
                fold_dir / f"route_conditioned_oracle_time_{split}.parquet",
            )
            manifest["folds"][str(fold_id)][split] = {"estimated": est_stats, "oracle": oracle_stats}
            if fold_id == args.fold:
                concat_split(estimated_day_root, split_values, args.output_root / f"route_conditioned_estimated_time_{split}.parquet")
                concat_split(oracle_day_root, split_values, args.output_root / f"route_conditioned_oracle_time_{split}.parquet")
    (args.output_root / "route_conditioned_dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
