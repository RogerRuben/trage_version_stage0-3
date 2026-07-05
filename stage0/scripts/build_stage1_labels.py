"""Build five-dimensional link labels with streaming cohort normalization.

The implementation uses part-level summaries and empirical histograms, so fitting
does not require loading all monthly traversals into memory. Reference speed is
estimated from weighted part medians with hierarchical fallback.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely


DIMENSIONS = ["lcs", "iis", "gns", "rts", "pmis"]
LEVELS = ["level1", "level2", "level3", "level4", "level5", "level6"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traversal-root", type=Path, required=True)
    parser.add_argument("--movement-root", type=Path, required=True)
    parser.add_argument("--poi-exposure", type=Path, required=True)
    parser.add_argument("--roads", type=Path, required=True)
    parser.add_argument("--order-base-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("stage0_output"))
    parser.add_argument("--fit-dates", default="all", help="all or comma-separated YYYYMMDD")
    parser.add_argument("--target-dates", default="all", help="all or comma-separated YYYYMMDD")
    parser.add_argument("--min-cohort-size", type=int, default=100)
    parser.add_argument("--histogram-bins", type=int, default=200)
    parser.add_argument("--high-threshold", type=float, default=0.90)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def discover_dates(root: Path) -> list[str]:
    return sorted(path.name.split("=", 1)[1] for path in root.glob("day=*") if path.is_dir())


def select_dates(value: str, available: list[str]) -> list[str]:
    if value == "all":
        return available
    selected = [part.strip() for part in value.split(",") if part.strip()]
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"dates not available: {missing}")
    return selected


def link_curvature(line) -> float:
    coords = np.asarray(line.coords)
    if len(coords) < 3:
        return 0.0
    delta = np.diff(coords, axis=0)
    lengths = np.hypot(delta[:, 0], delta[:, 1])
    heading = np.arctan2(delta[:, 1], delta[:, 0])
    changes = np.abs((np.diff(heading) + np.pi) % (2 * np.pi) - np.pi)
    valid = (lengths[:-1] >= 2) & (lengths[1:] >= 2)
    return float(np.degrees(changes[valid]).sum()) if valid.any() else 0.0


def road_features(path: Path) -> pd.DataFrame:
    roads = gpd.read_parquet(path).to_crs(32649).reset_index(drop=True)
    degree = pd.concat([roads.from_node, roads.to_node]).value_counts()
    midpoint = shapely.line_interpolate_point(roads.geometry.to_numpy(), shapely.length(roads.geometry.to_numpy()) / 2)
    minor = {"residential", "unclassified", "service", "living_street", "track"}
    frame = pd.DataFrame({
        "link_id": roads.link_id.astype(str), "road_class": roads.road_class.fillna("unknown").astype(str),
        "link_length_m_static": roads.length_m.astype(float),
        "curvature_deg_per_km_link": [link_curvature(line) for line in roads.geometry],
        "minor_road": roads.road_class.fillna("unknown").astype(str).isin(minor).astype(float),
        "endpoint_degree": (
            roads.from_node.map(degree).fillna(0).to_numpy() + roads.to_node.map(degree).fillna(0).to_numpy()
        ) / 2,
        "area_grid": [f"{int(x // 1000)}_{int(y // 1000)}" for x, y in zip(shapely.get_x(midpoint), shapely.get_y(midpoint))],
    })
    frame["curvature_deg_per_km_link"] /= (frame.link_length_m_static / 1000).replace(0, np.nan)
    frame["link_fragmentation"] = (1000 / frame.link_length_m_static.clip(lower=20)).clip(upper=20)
    return frame


def add_context(frame: pd.DataFrame, roads: pd.DataFrame) -> pd.DataFrame:
    frame = frame.merge(roads, on="link_id", how="left", validate="many_to_one")
    local = pd.to_datetime(frame.enter_time, unit="s", utc=True).dt.tz_convert("Asia/Shanghai")
    minute = local.dt.hour * 60 + local.dt.minute
    frame["time_bin"] = (minute // 30).astype("int16")
    frame["weekday_type"] = np.where(local.dt.dayofweek < 5, "weekday", "weekend")
    frame["peak_offpeak"] = np.where(
        minute.between(7 * 60, 9 * 60 + 30) | minute.between(17 * 60, 19 * 60 + 30), "peak", "offpeak"
    )
    frame["level1"] = frame.link_id.astype(str) + "|" + frame.time_bin.astype(str) + "|" + frame.weekday_type
    frame["level2"] = frame.link_id.astype(str) + "|" + frame.peak_offpeak
    frame["level3"] = frame.road_class.astype(str) + "|" + frame.area_grid.astype(str) + "|" + frame.time_bin.astype(str)
    frame["level4"] = frame.road_class.astype(str) + "|" + frame.time_bin.astype(str)
    frame["level5"] = frame.road_class.astype(str)
    frame["level6"] = "global"
    return frame


def reduce_reference(accumulator: pd.DataFrame | None, addition: pd.DataFrame) -> pd.DataFrame:
    combined = addition if accumulator is None else pd.concat([accumulator, addition], ignore_index=True)
    return combined.groupby("key", as_index=False).agg(weighted_value=("weighted_value", "sum"), sample_size=("sample_size", "sum"))


def fit_reference(files: list[Path], roads: pd.DataFrame, output: Path) -> dict[str, pd.DataFrame]:
    models: dict[str, pd.DataFrame | None] = {level: None for level in LEVELS}
    for file_no, path in enumerate(files, start=1):
        frame = pd.read_parquet(path, columns=[
            "link_id", "enter_time", "travel_time_sec", "observed_distance_m", "traversal_quality"
        ])
        frame = add_context(frame, roads)
        frame = frame[
            frame.traversal_quality.ne("low") & frame.travel_time_sec.gt(0) & frame.observed_distance_m.ge(10)
        ].copy()
        frame["sec_per_m"] = (frame.travel_time_sec / frame.observed_distance_m).clip(0.01, 10.0)
        for level in LEVELS:
            summary = frame.groupby(level).sec_per_m.agg(["median", "count"]).reset_index().rename(columns={level: "key"})
            summary["weighted_value"] = summary["median"] * summary["count"]
            summary = summary.rename(columns={"count": "sample_size"})[["key", "weighted_value", "sample_size"]]
            models[level] = reduce_reference(models[level], summary)
        if file_no % 16 == 0:
            print(f"reference summaries: {file_no}/{len(files)}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    finalized: dict[str, pd.DataFrame] = {}
    for level in LEVELS:
        model = models[level]
        assert model is not None
        model["reference_sec_per_m"] = model.weighted_value / model.sample_size
        finalized[level] = model[["key", "sample_size", "reference_sec_per_m"]]
        finalized[level].to_parquet(output / f"{level}.parquet", index=False, compression="zstd")
    return finalized


def choose_reference(frame: pd.DataFrame, models: dict[str, pd.DataFrame], minimum: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference = np.full(len(frame), np.nan); level_used = np.full(len(frame), 6, dtype="int8"); sample_size = np.zeros(len(frame), dtype="int32")
    unresolved = np.ones(len(frame), dtype=bool)
    for level_no, level in enumerate(LEVELS, start=1):
        model = models[level].set_index("key")
        counts = frame[level].map(model.sample_size).fillna(0).to_numpy()
        values = frame[level].map(model.reference_sec_per_m).to_numpy(dtype=float)
        eligible = unresolved & ((counts >= minimum) | (level_no == 6)) & np.isfinite(values)
        reference[eligible] = values[eligible]; level_used[eligible] = level_no; sample_size[eligible] = counts[eligible]
        unresolved[eligible] = False
    return reference, level_used, sample_size


def bounded(value: pd.Series, scale: float = 1.0) -> pd.Series:
    clean = value.fillna(0).clip(lower=0)
    return clean / (clean + scale)


def enrich_part(
    traversal_path: Path, movement_path: Path | None, roads: pd.DataFrame, exposure: pd.DataFrame,
    references: dict[str, pd.DataFrame], minimum: int,
) -> pd.DataFrame:
    frame = add_context(pd.read_parquet(traversal_path), roads)
    frame = frame.merge(exposure, on="link_id", how="left", validate="many_to_one")
    if movement_path and movement_path.exists():
        movement = pd.read_parquet(movement_path).rename(columns={"movement_seq": "link_seq"})
        movement = movement[[
            "order_id", "link_seq", "turn_angle", "node_degree", "junction_complexity",
            "intersection_low_speed_time", "intersection_stop_time", "movement_quality",
        ]]
        frame = frame.merge(movement, on=["order_id", "link_seq"], how="left")
    else:
        for column in ["turn_angle", "node_degree", "junction_complexity", "intersection_low_speed_time", "intersection_stop_time"]:
            frame[column] = np.nan
        frame["movement_quality"] = "missing"
    density_columns = [column for column in frame.columns if column.startswith("poi_density_100m_")]
    frame[density_columns + ["activity_intensity_index"]] = frame[density_columns + ["activity_intensity_index"]].fillna(0)

    reference, ref_level, ref_size = choose_reference(frame, references, minimum)
    frame["reference_sec_per_m"] = reference
    frame["reference_level_used"] = ref_level
    frame["reference_sample_size"] = ref_size
    frame["reference_travel_time_sec"] = reference * frame.observed_distance_m
    frame["excess_time_ratio"] = (
        (frame.travel_time_sec - frame.reference_travel_time_sec) / frame.reference_travel_time_sec.replace(0, np.nan)
    )
    frame["tail_delay_ratio"] = frame.excess_time_ratio.clip(lower=0)

    speed_component = bounded(frame.speed_cv, 1.0)
    accel_component = bounded(frame.accel_volatility, 1.0)
    frame["lcs_raw"] = pd.concat([
        frame.low_speed_ratio.fillna(0), frame.stop_duration_ratio.fillna(0), speed_component, accel_component
    ], axis=1).mean(axis=1).clip(0, 1)
    frame["iis_raw"] = pd.concat([
        (frame.intersection_low_speed_time / frame.travel_time_sec.replace(0, np.nan)).fillna(0).clip(0, 1),
        (frame.intersection_stop_time / frame.travel_time_sec.replace(0, np.nan)).fillna(0).clip(0, 1),
        (frame.turn_angle.abs() / 180).fillna(0).clip(0, 1),
        (frame.node_degree / 8).fillna(0).clip(0, 1),
    ], axis=1).mean(axis=1)
    frame["gns_raw"] = pd.concat([
        bounded(frame.curvature_deg_per_km_link, 90), frame.minor_road.fillna(0),
        (frame.link_fragmentation / 10).clip(0, 1), (frame.endpoint_degree / 8).clip(0, 1),
    ], axis=1).mean(axis=1)
    frame["rts_raw"] = bounded(frame.tail_delay_ratio, 1.0).clip(0, 1)
    behavioral = (frame.lcs_raw + frame.rts_raw) / 2
    frame["pmis_raw"] = (frame.activity_intensity_index.clip(0, 1) * behavioral).clip(0, 1)
    frame["delay_on_poi_link"] = frame.tail_delay_ratio.where(frame.activity_intensity_index.ge(0.75), 0.0)
    no_realized_observation = frame.traversal_quality.isin(["low", "inferred_path"])
    frame.loc[no_realized_observation, ["lcs_raw", "iis_raw", "rts_raw", "pmis_raw"]] = np.nan
    last_link = frame.link_seq.eq(frame.groupby("order_id").link_seq.transform("max"))
    unreliable_movement = frame.movement_quality.eq("low") & ~last_link
    frame.loc[unreliable_movement, "iis_raw"] = np.nan
    return frame


def write_final_poi_behavior(frame: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    output = frame[["order_id", "driver_id", "date", "link_id", "link_seq", "activity_intensity_index"]].copy()
    for category in ["school", "hospital", "commercial", "transit"]:
        output[f"poi_density_{category}"] = frame.get(f"poi_density_100m_{category}", 0.0)
    exposed = frame.activity_intensity_index.ge(0.75)
    output["low_speed_ratio_on_poi_link"] = frame.low_speed_ratio.where(exposed, 0.0)
    output["stop_time_on_poi_link"] = frame.stop_time_sec.where(exposed, 0.0)
    output["delay_on_poi_link"] = frame.delay_on_poi_link
    output["poi_interaction_candidate"] = exposed & (
        frame.low_speed_ratio.ge(0.25) | frame.stop_time_sec.gt(0) | frame.tail_delay_ratio.gt(0.25)
    )
    output["delay_reference_status"] = "historical_hierarchical_link_baseline"
    output.to_parquet(target, index=False, compression="zstd")


def reduce_hist(accumulator: pd.DataFrame | None, addition: pd.DataFrame) -> pd.DataFrame:
    combined = addition if accumulator is None else pd.concat([accumulator, addition], ignore_index=True)
    return combined.groupby(["key", "bin"], as_index=False)["count"].sum()


def fit_histograms(files: list[Path], model_dir: Path, bins: int) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    for dimension in DIMENSIONS:
        accumulators: dict[str, pd.DataFrame | None] = {level: None for level in LEVELS}
        for file_no, path in enumerate(files, start=1):
            columns = LEVELS + [f"{dimension}_raw"]
            frame = pd.read_parquet(path, columns=columns)
            frame = frame[frame[f"{dimension}_raw"].notna()].copy()
            value_bin = np.minimum((frame[f"{dimension}_raw"].clip(0, 1) * bins).astype(int), bins - 1)
            for level in LEVELS:
                summary = pd.DataFrame({"key": frame[level], "bin": value_bin}).value_counts().rename("count").reset_index()
                accumulators[level] = reduce_hist(accumulators[level], summary)
            if file_no % 32 == 0:
                print(f"hist {dimension}: {file_no}/{len(files)}", flush=True)
        for level in LEVELS:
            model = accumulators[level]
            assert model is not None
            model = model.sort_values(["key", "bin"])
            model["sample_size"] = model.groupby("key")["count"].transform("sum")
            model["cdf_midrank"] = (
                model.groupby("key")["count"].cumsum() - model["count"] / 2
            ) / model.sample_size
            model.to_parquet(model_dir / f"{dimension}_{level}.parquet", index=False, compression="zstd")


def normalize_part(frame: pd.DataFrame, model_dir: Path, bins: int, minimum: int) -> pd.DataFrame:
    common_level = np.full(len(frame), 6, dtype="int8"); common_size = np.zeros(len(frame), dtype="int32")
    unresolved = np.ones(len(frame), dtype=bool)
    count_models: dict[str, pd.Series] = {}
    for level_no, level in enumerate(LEVELS, start=1):
        model = pd.read_parquet(model_dir / f"lcs_{level}.parquet", columns=["key", "sample_size"]).drop_duplicates("key")
        counts = model.set_index("key").sample_size
        count_models[level] = counts
        values = frame[level].map(counts).fillna(0).to_numpy(dtype=int)
        eligible = unresolved & ((values >= minimum) | (level_no == 6))
        common_level[eligible] = level_no; common_size[eligible] = values[eligible]; unresolved[eligible] = False
    frame["cohort_level_used"] = common_level
    frame["cohort_sample_size"] = common_size
    for dimension in DIMENSIONS:
        result = np.full(len(frame), np.nan)
        value_bin = np.minimum((frame[f"{dimension}_raw"].fillna(0).clip(0, 1) * bins).astype(int), bins - 1)
        for level_no, level in enumerate(LEVELS, start=1):
            mask = common_level == level_no
            if not mask.any():
                continue
            model = pd.read_parquet(model_dir / f"{dimension}_{level}.parquet", columns=["key", "bin", "cdf_midrank"])
            lookup = model.set_index(["key", "bin"]).cdf_midrank
            index = pd.MultiIndex.from_arrays([frame.loc[mask, level], value_bin.loc[mask]])
            result[mask] = lookup.reindex(index).fillna(0.5).to_numpy()
        result[frame[f"{dimension}_raw"].isna().to_numpy()] = np.nan
        frame[f"{dimension}_pct_link"] = result
    return frame


def aggregate_orders(labels: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: list[dict] = []
    for order_id, group in labels.groupby("order_id", sort=False):
        row: dict[str, object] = {"order_id": order_id}
        maxima = []
        for dimension in DIMENSIONS:
            if dimension == "gns":
                length_column = next(
                    column for column in ["link_length_m", "link_length_m_x", "link_length_m_static"]
                    if column in group.columns
                )
                weight_series = group[length_column]
            else:
                weight_series = group.travel_time_sec
            weights = weight_series.fillna(0).clip(lower=0).to_numpy(dtype=float)
            if weights.sum() <= 0: weights = np.ones(len(group), dtype=float)
            values = group[f"{dimension}_pct_link"].to_numpy(dtype=float)
            valid = np.isfinite(values)
            if not valid.any():
                for suffix in ["mean", "max", "tail", "persistence"]: row[f"{dimension}_{suffix}"] = np.nan
                maxima.append(np.nan); continue
            valid_values = values[valid]; valid_weights = weights[valid]
            if valid_weights.sum() <= 0: valid_weights = np.ones(len(valid_values), dtype=float)
            row[f"{dimension}_mean"] = float(np.average(valid_values, weights=valid_weights))
            row[f"{dimension}_max"] = float(np.max(valid_values))
            tail_values = valid_values[valid_values >= threshold]
            row[f"{dimension}_tail"] = float(tail_values.mean()) if len(tail_values) else float(np.max(valid_values))
            row[f"{dimension}_persistence"] = float(valid_weights[valid_values >= threshold].sum() / valid_weights.sum())
            maxima.append(float(np.max(valid_values)))
        for cutoff in [0.85, 0.90, 0.95]:
            row[f"high_odd_exceedance_{int(cutoff*100)}"] = any(
                row[f"{dimension}_tail"] >= cutoff and row[f"{dimension}_persistence"] >= 0.05
                for dimension in DIMENSIONS
            )
        row["composite_mean"] = float(np.nanmean([row[f"{d}_mean"] for d in DIMENSIONS]))
        row["composite_tail"] = float(np.nanmean([row[f"{d}_tail"] for d in DIMENSIONS]))
        high_quality = group.traversal_quality.eq("high").mean()
        usable_quality = group.traversal_quality.isin(["high", "usable"]).mean()
        row["quality_tier"] = "A" if high_quality >= 0.8 else "B" if usable_quality >= 0.8 else "C"
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    available = discover_dates(args.traversal_root)
    fit_dates = select_dates(args.fit_dates, available); target_dates = select_dates(args.target_dates, available)
    roads = road_features(args.roads)
    exposure = pd.read_parquet(args.poi_exposure).drop(columns=["link_length_m"], errors="ignore")
    fit_traversals = [path for date in fit_dates for path in sorted((args.traversal_root / f"day={date}").glob("*.parquet"))]
    reference_dir = args.output_root / "stage1_models" / "travel_time_reference"
    references = fit_reference(fit_traversals, roads, reference_dir)

    primitive_files: list[Path] = []
    for date in sorted(set(fit_dates) | set(target_dates)):
        traversal_dir = args.traversal_root / f"day={date}"
        movement_dir = args.movement_root / f"day={date}"
        primitive_dir = args.output_root / "stage1_primitives" / f"day={date}"
        primitive_dir.mkdir(parents=True, exist_ok=True)
        for traversal_path in sorted(traversal_dir.glob("*.parquet")):
            part = traversal_path.stem.split("=")[-1].split("_")[-1]
            movement_path = movement_dir / f"part={part}.parquet"
            target = primitive_dir / f"part={part}.parquet"
            behavior_target = args.output_root / "stage0_order_link_poi_behavior" / f"day={date}" / f"part={part}.parquet"
            if args.force or not target.exists():
                frame = enrich_part(traversal_path, movement_path, roads, exposure, references, args.min_cohort_size)
                frame.to_parquet(target, index=False, compression="zstd")
            else:
                frame = pd.read_parquet(target)
            write_final_poi_behavior(frame, behavior_target)
            if date in fit_dates:
                primitive_files.append(target)
    model_dir = args.output_root / "stage1_models" / "cohort_histograms"
    fit_histograms(primitive_files, model_dir, args.histogram_bins)

    for date in target_dates:
        primitive_dir = args.output_root / "stage1_primitives" / f"day={date}"
        label_dir = args.output_root / "stage1_link_labels" / f"day={date}"
        label_dir.mkdir(parents=True, exist_ok=True)
        order_parts: list[pd.DataFrame] = []
        for source in sorted(primitive_dir.glob("*.parquet")):
            part = source.stem.split("=")[-1].split("_")[-1]
            frame = normalize_part(pd.read_parquet(source), model_dir, args.histogram_bins, args.min_cohort_size)
            columns = [
                "order_id", "driver_id", "date", "link_id", "link_seq", "enter_time", "exit_time",
                "travel_time_sec", "observed_distance_m", "traversal_quality", "reference_travel_time_sec",
                "excess_time_ratio", "tail_delay_ratio", "cohort_level_used", "cohort_sample_size",
            ] + [f"{dimension}_{suffix}" for dimension in DIMENSIONS for suffix in ["raw", "pct_link"]]
            frame[columns].to_parquet(label_dir / f"part={part}.parquet", index=False, compression="zstd")
            order_parts.append(aggregate_orders(frame, args.high_threshold))
            print(f"labels day={date} part={part} rows={len(frame):,}", flush=True)
        orders = pd.concat(order_parts, ignore_index=True)
        if args.order_base_root:
            base_path = args.order_base_root / f"day={date}.parquet"
            if base_path.exists():
                base = pd.read_parquet(base_path, columns=["order_id", "quality_tier"]).rename(columns={"quality_tier": "stage0_quality_tier"})
                orders = orders.merge(base, on="order_id", how="left")
        order_path = args.output_root / "stage1_order_labels" / f"day={date}.parquet"
        order_path.parent.mkdir(parents=True, exist_ok=True)
        orders.to_parquet(order_path, index=False, compression="zstd")
    manifest = {
        "fit_dates": fit_dates, "target_dates": target_dates, "dimensions": DIMENSIONS,
        "min_cohort_size": args.min_cohort_size, "histogram_bins": args.histogram_bins,
        "high_threshold": args.high_threshold, "complete": True,
    }
    manifest_dir = args.output_root / "manifests"; manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "stage1_labels.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
