"""Build split-aware Stage2 link-level prediction datasets.

Each output row is one order_id x link_id x link_seq traversal. The builder
streams existing Stage1 primitive and link-label partitions into one parquet
file per temporal split, so it does not need to load all split rows at once.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DIMENSIONS = ["lcs", "iis", "gns", "rts", "pmis"]
TARGET_DIMS = ["lcs", "iis", "rts", "pmis"]
POI_CATEGORIES = [
    "school", "hospital", "commercial", "restaurant", "transit",
    "bus_stop", "residential", "office", "scenic", "parking",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-output-root", type=Path, default=Path("stage1/output/prediction_split"))
    parser.add_argument("--split-config", type=Path, default=Path("split_config.json"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/link_dataset"))
    parser.add_argument("--timezone", default=None)
    parser.add_argument("--max-parts-per-day", type=int, default=None, help="Debug limit; omit for full build.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def split_dates(config: dict) -> dict[str, list[str]]:
    return {
        "train": list(config["train_dates"]),
        "validation": [config["validation_date"]],
        "test": [config["test_date"]],
    }


def available_columns(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema_arrow.names


def read_parquet_columns(path: Path, desired: list[str]) -> pd.DataFrame:
    columns = [column for column in desired if column in available_columns(path)]
    return pd.read_parquet(path, columns=columns)


def parse_hour(values: pd.Series, timezone: str) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        parsed = pd.to_datetime(values, unit="s", utc=True, errors="coerce")
    else:
        parsed = pd.to_datetime(values, utc=True, errors="coerce")
    try:
        parsed = parsed.dt.tz_convert(timezone)
    except Exception:
        pass
    return parsed.dt.hour.astype("float").astype("Int64")


def ensure_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame


def add_targets(frame: pd.DataFrame) -> pd.DataFrame:
    for dimension in TARGET_DIMS:
        source = f"{dimension}_pct_link"
        frame[f"target_{dimension}_pct"] = frame[source] if source in frame.columns else pd.NA
        frame[f"target_high_{dimension}_90"] = (
            frame[source].ge(0.90).astype("boolean") if source in frame.columns else pd.Series(pd.NA, index=frame.index, dtype="boolean")
        )
    for dimension in TARGET_DIMS:
        source = f"{dimension}_pct_link"
        frame[f"{dimension}_valid"] = frame[source].notna() if source in frame.columns else False
    return frame


def add_temporal_features(frame: pd.DataFrame, timezone: str) -> pd.DataFrame:
    if "enter_time" in frame.columns:
        frame["hour"] = parse_hour(frame["enter_time"], timezone)
    elif "time_bin" in frame.columns:
        frame["hour"] = np.floor(pd.to_numeric(frame["time_bin"], errors="coerce") / 2).astype("Int64")
    else:
        frame["hour"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    if "weekday_type" in frame.columns:
        frame["is_weekend"] = frame["weekday_type"].astype(str).str.lower().eq("weekend")
    else:
        frame["is_weekend"] = pd.NA
    return frame


def add_route_position_features(frame: pd.DataFrame) -> pd.DataFrame:
    if "link_length_m_static" in frame.columns:
        frame["stage2_link_length_m"] = pd.to_numeric(frame["link_length_m_static"], errors="coerce")
    elif "link_length_m" in frame.columns:
        frame["stage2_link_length_m"] = pd.to_numeric(frame["link_length_m"], errors="coerce")
    else:
        frame["stage2_link_length_m"] = 1.0
    frame["stage2_link_length_m"] = frame["stage2_link_length_m"].fillna(0).clip(lower=0)

    if "link_seq" not in frame.columns:
        frame["link_seq"] = 0
    frame = frame.sort_values(["order_id", "link_seq"], kind="mergesort").copy()
    route_link_count = frame.groupby("order_id", sort=False)["link_seq"].transform("count")
    frame["route_link_count"] = route_link_count.astype("int32")
    denominator = (route_link_count - 1).clip(lower=1)
    frame["position_ratio"] = pd.to_numeric(frame["link_seq"], errors="coerce").fillna(0) / denominator

    total_length = frame.groupby("order_id", sort=False)["stage2_link_length_m"].transform("sum")
    cum_length = frame.groupby("order_id", sort=False)["stage2_link_length_m"].cumsum()
    remaining = (total_length - cum_length).clip(lower=0)
    frame["distance_to_destination_ratio"] = np.where(total_length > 0, remaining / total_length, 0.0)
    return frame


def add_quality_features(frame: pd.DataFrame) -> pd.DataFrame:
    if "traversal_quality" in frame.columns:
        frame["observed_or_inferred"] = np.where(frame["traversal_quality"].eq("inferred_path"), "inferred", "observed")
        frame["low_quality_flag"] = frame["traversal_quality"].eq("low")
    else:
        frame["observed_or_inferred"] = pd.NA
        frame["low_quality_flag"] = pd.NA
    return frame


def add_poi_behavior_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    if "activity_intensity_index" in frame.columns:
        poi_exposed = pd.to_numeric(frame["activity_intensity_index"], errors="coerce").fillna(0).gt(0)
    else:
        poi_exposed = pd.Series(False, index=frame.index)
    if "low_speed_ratio_on_poi_link" not in frame.columns:
        frame["low_speed_ratio_on_poi_link"] = frame["low_speed_ratio"].where(poi_exposed, 0.0) if "low_speed_ratio" in frame.columns else pd.NA
    if "stop_time_on_poi_link" not in frame.columns:
        frame["stop_time_on_poi_link"] = frame["stop_time_sec"].where(poi_exposed, 0.0) if "stop_time_sec" in frame.columns else pd.NA
    if "delay_on_poi_link" not in frame.columns:
        frame["delay_on_poi_link"] = pd.NA
    return frame


def build_part(primitive_path: Path, label_path: Path, timezone: str) -> pd.DataFrame:
    primitive_columns = [
        "order_id", "driver_id", "date", "link_id", "link_seq", "enter_time", "exit_time",
        "time_bin", "weekday_type", "peak_offpeak", "matcher_version", "traversal_quality",
        "road_class", "link_length_m", "link_length_m_static", "curvature_deg_per_km_link",
        "minor_road", "endpoint_degree", "link_fragmentation", "area_grid",
        "activity_intensity_index", "low_speed_ratio", "stop_time_sec", "delay_on_poi_link",
        "reference_travel_time_sec", "excess_time_ratio", "tail_delay_ratio",
    ] + [f"poi_density_100m_{category}" for category in POI_CATEGORIES]
    label_columns = [
        "order_id", "link_id", "link_seq",
        "travel_time_sec", "observed_distance_m", "traversal_quality",
        "reference_travel_time_sec", "excess_time_ratio", "tail_delay_ratio",
    ] + [f"{dimension}_{suffix}" for dimension in DIMENSIONS for suffix in [
        "pct_link", "cohort_level_used", "cohort_sample_size"
    ]]
    primitive = read_parquet_columns(primitive_path, primitive_columns)
    labels = read_parquet_columns(label_path, label_columns)
    frame = primitive.merge(labels, on=["order_id", "link_id", "link_seq"], how="inner", suffixes=("", "_label"))

    for column in ["traversal_quality", "reference_travel_time_sec", "excess_time_ratio", "tail_delay_ratio"]:
        label_column = f"{column}_label"
        if label_column in frame.columns:
            frame[column] = frame[column].combine_first(frame[label_column]) if column in frame.columns else frame[label_column]
            frame = frame.drop(columns=[label_column])
    frame = add_targets(frame)
    frame = add_temporal_features(frame, timezone)
    frame = add_route_position_features(frame)
    frame = add_quality_features(frame)
    frame = add_poi_behavior_aliases(frame)

    if "link_length_m_static" in frame.columns:
        frame["static_link_length_m"] = frame["link_length_m_static"]
    elif "link_length_m" in frame.columns:
        frame["static_link_length_m"] = frame["link_length_m"]
    else:
        frame["static_link_length_m"] = pd.NA

    keep_columns = [
        "order_id", "driver_id", "date", "link_id", "link_seq", "enter_time", "time_bin",
        "weekday_type", "peak_offpeak", "matcher_version", "traversal_quality",
        "target_lcs_pct", "target_iis_pct", "target_rts_pct", "target_pmis_pct",
        "target_high_lcs_90", "target_high_iis_90", "target_high_rts_90", "target_high_pmis_90",
        "road_class", "static_link_length_m", "curvature_deg_per_km_link", "minor_road",
        "endpoint_degree", "link_fragmentation", "area_grid", "gns_pct_link",
        "activity_intensity_index",
    ] + [f"poi_density_100m_{category}" for category in POI_CATEGORIES] + [
        "hour", "is_weekend", "route_link_count", "position_ratio", "distance_to_destination_ratio",
        "observed_or_inferred", "low_quality_flag", "lcs_valid", "iis_valid", "rts_valid", "pmis_valid",
        "travel_time_sec", "observed_distance_m", "reference_travel_time_sec", "excess_time_ratio", "tail_delay_ratio",
        "low_speed_ratio_on_poi_link", "stop_time_on_poi_link", "delay_on_poi_link",
        "lcs_cohort_level_used", "iis_cohort_level_used", "rts_cohort_level_used", "pmis_cohort_level_used",
        "lcs_cohort_sample_size", "iis_cohort_sample_size", "rts_cohort_sample_size", "pmis_cohort_sample_size",
    ]
    frame = ensure_columns(frame, keep_columns)
    frame = frame[keep_columns].rename(columns={"static_link_length_m": "link_length_m"})
    return frame


def write_split(split: str, dates: list[str], args: argparse.Namespace, timezone: str) -> dict[str, object]:
    writer: pq.ParquetWriter | None = None
    output_path = args.output_root / f"{split}.parquet"
    if output_path.exists() and not args.dry_run:
        output_path.unlink()
    row_count = 0
    part_count = 0
    date_rows: dict[str, int] = {}
    try:
        for date in dates:
            primitive_dir = args.stage1_output_root / "primitives" / f"day={date}"
            label_dir = args.stage1_output_root / "link_labels" / f"day={date}"
            primitive_files = sorted(primitive_dir.glob("*.parquet"))
            if args.max_parts_per_day is not None:
                primitive_files = primitive_files[: args.max_parts_per_day]
            for primitive_path in primitive_files:
                label_path = label_dir / primitive_path.name
                if not label_path.exists():
                    continue
                frame = build_part(primitive_path, label_path, timezone)
                part_count += 1
                row_count += len(frame)
                date_rows[date] = date_rows.get(date, 0) + len(frame)
                if args.dry_run:
                    continue
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
                writer.write_table(table)
                print(f"{split} {date} {primitive_path.name}: rows={len(frame):,}", flush=True)
    finally:
        if writer is not None:
            writer.close()
    return {
        "split": split,
        "dates": dates,
        "rows": row_count,
        "parts": part_count,
        "output": str(output_path),
        "date_rows": date_rows,
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.split_config.read_text(encoding="utf-8"))
    timezone = args.timezone or config.get("timezone", "Asia/Shanghai")
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for split, dates in split_dates(config).items():
        summaries.append(write_split(split, dates, args, timezone))
    manifest = {
        "split_config": str(args.split_config),
        "stage1_output_root": str(args.stage1_output_root),
        "timezone": timezone,
        "dry_run": args.dry_run,
        "splits": summaries,
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
