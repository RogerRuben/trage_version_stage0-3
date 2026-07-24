"""Build canonical Stage 1 schema-v2 labels from an explicit Stage 0 manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canonical_pipeline.manifest import load_manifest, require_canonical_input
from stage1.canonical.labels import DIMENSIONS, aggregate_order_labels_v2
from stage1.canonical.quantiles import empirical_cdf_interpolated
from stage1.scripts.build_stage1_labels import LEVELS, add_context, enrich_part, road_features


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--fit-dates", required=True)
    parser.add_argument("--target-dates", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-cohort-size", type=int, default=100)
    parser.add_argument("--reference-bins", type=int, default=4096)
    parser.add_argument("--cdf-bins", type=int, default=1000)
    parser.add_argument("--high-threshold", type=float, default=0.90)
    parser.add_argument("--legacy-v1-root", type=Path)
    parser.add_argument("--schema", type=Path, default=Path("config/artifact_manifest.schema.json"))
    return parser.parse_args()


def dates(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def role_paths(manifest, workspace: Path) -> dict[str, Path]:
    return {item["role"]: workspace / item["path"] for item in manifest.data["files"]}


def fit_references_v2(files: list[Path], roads: pd.DataFrame, bins: int, output: Path):
    edges = np.geomspace(0.01, 10.0, bins + 1)
    accumulators: dict[str, dict[str, np.ndarray]] = {level: {} for level in LEVELS}
    for path in files:
        frame = pd.read_parquet(path, columns=[
            "link_id", "enter_time", "travel_time_sec", "observed_distance_m", "traversal_quality"
        ])
        frame = add_context(frame, roads)
        frame = frame[
            frame.traversal_quality.ne("low")
            & frame.travel_time_sec.gt(0)
            & frame.observed_distance_m.ge(10)
        ].copy()
        frame["sec_per_m"] = (frame.travel_time_sec / frame.observed_distance_m).clip(edges[0], edges[-1])
        for level in LEVELS:
            for key, values in frame.groupby(level, sort=False).sec_per_m:
                counts = np.histogram(values.to_numpy(dtype=float), bins=edges)[0].astype(np.int64)
                current = accumulators[level].get(str(key))
                accumulators[level][str(key)] = counts if current is None else current + counts
    output.mkdir(parents=True, exist_ok=True)
    models = {}
    for level in LEVELS:
        rows = []
        for key, counts in accumulators[level].items():
            total = int(counts.sum())
            cumulative = np.cumsum(counts)
            index = int(np.searchsorted(cumulative, max(1, (total + 1) // 2), side="left"))
            before = int(cumulative[index - 1]) if index else 0
            fraction = np.clip(((total - 1) / 2 - before + 0.5) / max(int(counts[index]), 1), 0, 1)
            median = edges[index] + fraction * (edges[index + 1] - edges[index])
            rows.append({"key": key, "sample_size": total, "reference_sec_per_m": float(median)})
        model = pd.DataFrame(rows)
        model.to_parquet(output / f"{level}.parquet", index=False, compression="zstd")
        models[level] = model
    (output / "quantile_method.json").write_text(json.dumps({
        "method": "fixed_global_log_edges_mergeable_histogram",
        "partition_invariant": True,
        "bins": bins,
        "minimum": float(edges[0]),
        "maximum": float(edges[-1]),
        "maximum_log_bin_width": float(np.max(np.diff(np.log(edges)))),
    }, indent=2) + "\n", encoding="utf-8")
    return models


def choose_reference(frame: pd.DataFrame, models: dict[str, pd.DataFrame], minimum: int):
    reference = np.full(len(frame), np.nan)
    level_used = np.full(len(frame), len(LEVELS), dtype=np.int8)
    sample_size = np.zeros(len(frame), dtype=np.int32)
    unresolved = np.ones(len(frame), dtype=bool)
    for level_no, level in enumerate(LEVELS, start=1):
        model = models[level].set_index("key")
        counts = frame[level].map(model.sample_size).fillna(0).to_numpy(dtype=int)
        values = frame[level].map(model.reference_sec_per_m).to_numpy(dtype=float)
        eligible = unresolved & np.isfinite(values) & ((counts >= minimum) | (level_no == len(LEVELS)))
        reference[eligible] = values[eligible]
        level_used[eligible] = level_no
        sample_size[eligible] = counts[eligible]
        unresolved[eligible] = False
    return reference, level_used, sample_size


def prepare_primitive(traversal: Path, movement: Path, roads, exposure, references, minimum):
    # ``enrich_part`` is reused only for physical formulas. Its v1 reference is
    # replaced immediately by the partition-invariant v2 model below.
    frame = enrich_part(traversal, movement, roads, exposure, references, minimum)
    reference, level_used, sample_size = choose_reference(frame, references, minimum)
    frame["reference_sec_per_m"] = reference
    frame["reference_level_used"] = level_used
    frame["reference_sample_size"] = sample_size
    frame["reference_travel_time_sec"] = reference * frame.observed_distance_m
    frame["excess_time_ratio"] = (
        (frame.travel_time_sec - frame.reference_travel_time_sec)
        / frame.reference_travel_time_sec.replace(0, np.nan)
    )
    frame["tail_delay_ratio"] = frame.excess_time_ratio.clip(lower=0)
    clean_delay = frame.tail_delay_ratio.fillna(0).clip(lower=0)
    frame["rts_raw"] = clean_delay / (clean_delay + 1.0)
    behavioral = (frame.lcs_raw + frame.rts_raw) / 2
    frame["pmis_raw"] = (frame.activity_intensity_index.clip(0, 1) * behavioral).clip(0, 1)
    movement_observable = frame.movement_quality.notna() & frame.movement_quality.ne("missing")
    frame["iis_applicable"] = movement_observable & frame.node_degree.fillna(0).ge(3)
    frame["iis_severity_available"] = movement_observable & frame.iis_raw.notna()
    frame.loc[~frame.iis_severity_available, "iis_raw"] = np.nan
    return frame


def fit_cdf_models(frames: list[pd.DataFrame], output: Path, bins: int):
    output.mkdir(parents=True, exist_ok=True)
    edges = np.linspace(0.0, 1.0, bins + 1)
    for dimension in DIMENSIONS:
        for level in LEVELS:
            rows = []
            combined = pd.concat([
                frame[[level, f"{dimension}_raw"]]
                for frame in frames
            ], ignore_index=True).dropna()
            combined["bin"] = np.minimum(
                (combined[f"{dimension}_raw"].clip(0, 1) * bins).astype(int), bins - 1
            )
            for key, group in combined.groupby(level, sort=False):
                counts = group.bin.value_counts().sort_index()
                for bin_no, count in counts.items():
                    rows.append({
                        "key": str(key), "bin": int(bin_no), "count": int(count),
                        "support_value": float((edges[bin_no] + edges[bin_no + 1]) / 2),
                        "sample_size": int(len(group)),
                    })
            pd.DataFrame(rows).to_parquet(output / f"{dimension}_{level}.parquet", index=False, compression="zstd")


def normalize_v2(frame: pd.DataFrame, model_dir: Path, minimum: int) -> pd.DataFrame:
    for dimension in DIMENSIONS:
        result = np.full(len(frame), np.nan)
        selected_level = np.full(len(frame), len(LEVELS), dtype=np.int8)
        selected_size = np.zeros(len(frame), dtype=np.int32)
        unresolved = np.ones(len(frame), dtype=bool)
        models = {}
        for level_no, level in enumerate(LEVELS, start=1):
            model = pd.read_parquet(model_dir / f"{dimension}_{level}.parquet")
            models[level] = model
            supports = model.drop_duplicates("key").set_index("key").sample_size if len(model) else pd.Series(dtype=int)
            counts = frame[level].astype(str).map(supports).fillna(0).to_numpy(dtype=int)
            eligible = unresolved & ((counts >= minimum) | (level_no == len(LEVELS)))
            selected_level[eligible] = level_no
            selected_size[eligible] = counts[eligible]
            unresolved[eligible] = False
        for level_no, level in enumerate(LEVELS, start=1):
            level_mask = selected_level == level_no
            if not level_mask.any():
                continue
            model = models[level]
            for key, indexes in frame.loc[level_mask].groupby(level, sort=False).groups.items():
                support = model[model.key.eq(str(key))]
                if support.empty:
                    continue
                result[indexes] = empirical_cdf_interpolated(
                    frame.loc[indexes, f"{dimension}_raw"].to_numpy(dtype=float),
                    support.support_value.to_numpy(dtype=float),
                    support["count"].to_numpy(dtype=float),
                )
        result[frame[f"{dimension}_raw"].isna().to_numpy()] = np.nan
        frame[f"{dimension}_pct_link"] = result
        frame[f"{dimension}_cohort_level_used"] = selected_level
        frame[f"{dimension}_cohort_sample_size"] = selected_size
    return frame


def compare_v1_v2(v1_root: Path | None, output: Path, v2_orders: dict[str, pd.DataFrame]):
    rows = []
    for date, v2 in v2_orders.items():
        v1_path = v1_root / f"day={date}.parquet" if v1_root else None
        if v1_path is None or not v1_path.exists():
            rows.append({"date": date, "overlap_orders": 0, "status": "NO_V1_COMPARISON"})
            continue
        v1 = pd.read_parquet(v1_path)
        merged = v1.merge(v2, on="order_id", suffixes=("_v1", "_v2"))
        row = {"date": date, "overlap_orders": int(len(merged)), "status": "COMPARED"}
        for dimension in DIMENSIONS:
            a = f"{dimension}_mean_v1"; b = f"{dimension}_mean_v2"
            if a in merged and b in merged:
                row[f"{dimension}_mean_delta"] = float((merged[b] - merged[a]).mean())
                row[f"{dimension}_spearman"] = float(merged[[a, b]].corr(method="spearman").iloc[0, 1])
        rows.append(row)
    pd.DataFrame(rows).to_csv(output, index=False)


def main() -> None:
    args = arguments()
    workspace = Path.cwd()
    manifest = load_manifest(args.input_manifest, args.schema, workspace)
    require_canonical_input(manifest)
    if manifest.data["stage"] != "stage0":
        raise ValueError("Stage1 v2 requires a Stage0 manifest")
    paths = role_paths(manifest, workspace)
    fit_dates = dates(args.fit_dates); target_dates = dates(args.target_dates)
    roads_path = paths["roads"]
    exposure_path = paths["poi_exposure"]
    roads = road_features(roads_path)
    exposure = pd.read_parquet(exposure_path).drop(columns=["link_length_m"], errors="ignore")
    fit_files = [paths[f"link_traversals_{date}"] for date in fit_dates]
    reference_dir = args.output_root / "models" / "travel_time_reference_v2"
    references = fit_references_v2(fit_files, roads, args.reference_bins, reference_dir)
    primitives = {}
    for date in sorted(set(fit_dates) | set(target_dates)):
        frame = prepare_primitive(
            paths[f"link_traversals_{date}"], paths[f"turn_movements_{date}"],
            roads, exposure, references, args.min_cohort_size,
        )
        primitives[date] = frame
        target = args.output_root / "primitives" / f"day={date}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target, index=False, compression="zstd")
    cdf_dir = args.output_root / "models" / "cohort_cdf_v2"
    fit_cdf_models([primitives[date] for date in fit_dates], cdf_dir, args.cdf_bins)
    order_outputs = {}
    for date in target_dates:
        labels = normalize_v2(primitives[date].copy(), cdf_dir, args.min_cohort_size)
        link_path = args.output_root / "link_labels" / f"day={date}.parquet"
        link_path.parent.mkdir(parents=True, exist_ok=True)
        labels.to_parquet(link_path, index=False, compression="zstd")
        orders = aggregate_order_labels_v2(labels, args.high_threshold)
        order_path = args.output_root / "order_labels" / f"day={date}.parquet"
        order_path.parent.mkdir(parents=True, exist_ok=True)
        orders.to_parquet(order_path, index=False, compression="zstd")
        order_outputs[date] = orders
    comparison = args.output_root / "v1_v2_comparison.csv"
    compare_v1_v2(args.legacy_v1_root, comparison, order_outputs)
    summary = {
        "status": "PASS",
        "schema_version": "stage1_label_schema_v2",
        "input_artifact_id": manifest.artifact_id,
        "fit_dates": fit_dates,
        "target_dates": target_dates,
        "partition_invariant_quantiles": True,
        "cdf_empty_bin_policy": "ordered_support_interpolation",
        "pmis_in_core_composite": False,
        "missing_modality_policy": "preserve_na_with_mask",
        "orders": {date: int(len(frame)) for date, frame in order_outputs.items()},
    }
    files = [
        {"role": f"reference_{level}", "path": (reference_dir / f"{level}.parquet").as_posix()}
        for level in LEVELS
    ]
    files += [
        {"role": f"cdf_{dimension}_{level}", "path": (cdf_dir / f"{dimension}_{level}.parquet").as_posix()}
        for dimension in DIMENSIONS for level in LEVELS
    ]
    files += [
        {"role": f"link_labels_{date}", "path": (args.output_root / "link_labels" / f"day={date}.parquet").as_posix()}
        for date in target_dates
    ]
    files += [
        {"role": f"order_labels_{date}", "path": (args.output_root / "order_labels" / f"day={date}.parquet").as_posix()}
        for date in target_dates
    ]
    files.append({"role": "v1_v2_comparison", "path": comparison.as_posix()})
    summary["files"] = files
    (args.output_root / "stage1_v2_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
