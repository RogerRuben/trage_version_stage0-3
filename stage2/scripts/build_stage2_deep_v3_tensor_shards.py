"""Build fold-aware daily mmap tensor shards for RC-MSTNet.

Metadata is fitted from training dates only.  Each date is read and encoded
independently, so the builder never materializes a multi-day pandas frame.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage2_deep_v3_utils import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    ID_COLUMNS,
    LINK_TARGETS,
    DeepV3Metadata,
    RouteConditionedDataset,
    _balanced_order_budget,
    _truncate_routes,
    add_route_position_buckets,
    default_static_numeric_columns,
    dynamic_channel_names,
    dynamic_feature_columns,
    existing_columns,
    load_fold_config,
    read_filtered_day,
    safe_float,
    unique_existing,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("stage2/output/route_conditioned_dataset_5k/estimated_time_daily"))
    parser.add_argument("--fold-config", type=Path, default=Path("rolling_threefold_config.json"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/deep_v3_tensor_shards_5k"))
    parser.add_argument("--folds", default="1,2,3")
    parser.add_argument("--max-train-orders", type=int, default=0, help="0 means all available orders across train dates.")
    parser.add_argument("--max-eval-orders", type=int, default=0, help="0 means all available orders on validation/test dates.")
    parser.add_argument("--max-seq-len", type=int, default=160)
    parser.add_argument("--link-id-min-count", type=int, default=5)
    parser.add_argument("--feature-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def required_columns(path: Path) -> list[str]:
    columns = existing_columns(path)
    needed = list(dict.fromkeys(
        ID_COLUMNS
        + CATEGORICAL_COLUMNS
        + ["route_position_bucket", "route_length_bucket"]
        + default_static_numeric_columns(columns)
        + [column for values in dynamic_feature_columns(columns).values() for column in values]
        + [f"target_{target}_raw" for target in LINK_TARGETS]
        + [f"target_{target}_tail90_raw" for target in LINK_TARGETS]
        + [f"target_{target}_valid" for target in LINK_TARGETS]
    ))
    return unique_existing(path, needed)


def read_day(dataset_root: Path, date: str, order_ids: np.ndarray | None, max_seq_len: int) -> pd.DataFrame:
    path = dataset_root / f"day={date}.parquet"
    frame = read_filtered_day(path, required_columns(path), order_ids)
    frame = add_route_position_buckets(frame)
    frame = frame.sort_values(["order_id", "route_link_seq"], kind="mergesort")
    return _truncate_routes(frame, max_seq_len)


def select_orders(order_ids_by_date: dict[str, np.ndarray], max_orders: int | None, seed: int) -> dict[str, np.ndarray | None]:
    allocation = _balanced_order_budget({date: len(values) for date, values in order_ids_by_date.items()}, max_orders)
    rng = np.random.default_rng(seed)
    selected: dict[str, np.ndarray | None] = {}
    for date, values in order_ids_by_date.items():
        take = allocation[date]
        selected[date] = None if take == len(values) else rng.choice(values, size=take, replace=False)
    return selected


class StreamingMoments:
    def __init__(self, columns: list[str]):
        self.columns = columns
        self.count = {column: 0 for column in columns}
        self.total = {column: 0.0 for column in columns}
        self.total_sq = {column: 0.0 for column in columns}

    def update(self, frame: pd.DataFrame) -> None:
        for column in self.columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            self.count[column] += int(len(values))
            self.total[column] += float(values.sum(dtype="float64"))
            self.total_sq[column] += float(np.square(values).sum(dtype="float64"))

    def means(self) -> dict[str, float]:
        return {column: self.total[column] / self.count[column] if self.count[column] else 0.0 for column in self.columns}

    def stds(self) -> dict[str, float]:
        result = {}
        for column in self.columns:
            count = self.count[column]
            if count <= 1:
                result[column] = 1.0
                continue
            variance = (self.total_sq[column] - self.total[column] ** 2 / count) / (count - 1)
            value = float(np.sqrt(max(variance, 0.0)))
            result[column] = value if np.isfinite(value) and value > 0 else 1.0
        return result


def fit_metadata(
    dataset_root: Path,
    train_dates: list[str],
    selections: dict[str, np.ndarray | None],
    max_seq_len: int,
    link_id_min_count: int,
) -> tuple[DeepV3Metadata, dict]:
    first_path = dataset_root / f"day={train_dates[0]}.parquet"
    source_columns = existing_columns(first_path)
    static_columns = default_static_numeric_columns(source_columns)
    dynamic_by_window = dynamic_feature_columns(source_columns)
    dynamic_columns = sorted({column for values in dynamic_by_window.values() for column in values})
    categorical = [column for column in CATEGORICAL_COLUMNS + ["route_position_bucket", "route_length_bucket"] if column in required_columns(first_path) or column in {"route_position_bucket", "route_length_bucket"}]
    static_moments = StreamingMoments(static_columns)
    dynamic_moments = StreamingMoments(dynamic_columns)
    category_counts = {column: Counter() for column in categorical}
    rows = 0
    orders_by_date = {}
    for date in train_dates:
        frame = read_day(dataset_root, date, selections[date], max_seq_len)
        static_moments.update(frame)
        dynamic_moments.update(frame)
        for column in categorical:
            counts = frame[column].astype("string").fillna("__MISSING__").value_counts()
            category_counts[column].update({str(value): int(count) for value, count in counts.items()})
        rows += len(frame)
        orders_by_date[date] = int(frame["order_id"].nunique())
        del frame
    category_maps = {}
    for column in categorical:
        counts = category_counts[column]
        if column == "route_link_id":
            common = sorted(value for value, count in counts.items() if count >= link_id_min_count)
            has_rare = any(count < link_id_min_count for count in counts.values())
            values = sorted(common + (["__RARE_LINK__"] if has_rare and "__RARE_LINK__" not in common else []))
        else:
            values = sorted(counts)
        category_maps[column] = {str(value): index + 1 for index, value in enumerate(values)}
    metadata = DeepV3Metadata(
        static_numeric_columns=static_columns,
        dynamic_columns_by_window=dynamic_by_window,
        dynamic_channel_names=dynamic_channel_names(),
        categorical_columns=categorical,
        category_maps=category_maps,
        numeric_mean=static_moments.means(),
        numeric_std=static_moments.stds(),
        dynamic_mean=dynamic_moments.means(),
        dynamic_std=dynamic_moments.stds(),
        max_seq_len=max_seq_len,
        link_id_min_count=link_id_min_count,
    )
    return metadata, {"rows": rows, "orders_by_date": orders_by_date}


def stable_hash(payload: object) -> str:
    encoded = json.dumps(safe_float(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def selected_values(all_values: np.ndarray, selected: np.ndarray | None) -> np.ndarray:
    return all_values if selected is None else selected


def selection_hash(values: np.ndarray) -> str:
    return stable_hash(sorted(str(value) for value in values))


def reusable_shard(day_root: Path, metadata_hash: str, selected_hash: str, feature_dtype: str) -> dict | None:
    manifest_path = day_root / "manifest.json"
    if not manifest_path.exists():
        return None
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        existing.get("format_version") == 1
        and existing.get("metadata_sha256") == metadata_hash
        and existing.get("selection_sha256") == selected_hash
        and existing.get("feature_dtype") == feature_dtype
    ):
        return existing
    return None


def write_shard(
    day_root: Path,
    frame: pd.DataFrame,
    metadata: DeepV3Metadata,
    metadata_hash: str,
    selected_hash: str,
    date: str,
    feature_dtype: str,
    skip_existing: bool,
) -> dict:
    manifest_path = day_root / "manifest.json"
    if skip_existing and manifest_path.exists():
        existing = reusable_shard(day_root, metadata_hash, selected_hash, feature_dtype)
        if existing is not None:
            return existing
    day_root.mkdir(parents=True, exist_ok=True)
    dataset = RouteConditionedDataset(frame, metadata)
    arrays = {
        "static_numeric": dataset.static_numeric.astype(feature_dtype),
        "dynamic": dataset.dynamic.astype(feature_dtype),
        "categorical": dataset.categorical.astype("int32"),
        "target": dataset.target.astype("float32"),
        "tail": dataset.tail.astype("uint8"),
        "mask": dataset.mask.astype("uint8"),
        "offsets": dataset.offsets.astype("int64"),
        "lengths": dataset.lengths.astype("int32"),
    }
    for name, array in arrays.items():
        np.save(day_root / f"{name}.npy", array, allow_pickle=False)
    pd.DataFrame(dataset.ids).to_parquet(day_root / "ids.parquet", index=False, compression="zstd")
    manifest = {
        "format_version": 1,
        "date": date,
        "rows": int(dataset.offsets[-1]),
        "orders": len(dataset),
        "feature_dtype": feature_dtype,
        "metadata_sha256": metadata_hash,
        "selection_sha256": selected_hash,
        "array_shapes": {name: list(array.shape) for name, array in arrays.items()},
        "array_bytes": {name: int((day_root / f"{name}.npy").stat().st_size) for name in arrays},
        "ids_bytes": int((day_root / "ids.parquet").stat().st_size),
    }
    manifest_path.write_text(json.dumps(safe_float(manifest), indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    folds = load_fold_config(args.fold_config)
    selected_folds = {int(value.strip()) for value in args.folds.split(",") if value.strip()}
    all_dates = sorted({date for fold in folds for date in fold["train_dates"] + [fold["validation_date"], fold["test_date"]]})
    order_ids_by_date = {
        date: pd.read_parquet(args.dataset_root / f"day={date}.parquet", columns=["order_id"])["order_id"].astype(str).drop_duplicates().to_numpy()
        for date in all_dates
    }
    root_manifest = {"dataset_root": str(args.dataset_root), "folds": {}}
    for fold in folds:
        fold_id = int(fold["fold"])
        if fold_id not in selected_folds:
            continue
        fold_root = args.output_root / f"fold={fold_id}"
        train_ids = {date: order_ids_by_date[date] for date in fold["train_dates"]}
        max_train = None if args.max_train_orders <= 0 else args.max_train_orders
        train_selection = select_orders(train_ids, max_train, args.seed + fold_id)
        fold_root.mkdir(parents=True, exist_ok=True)
        fit_config = {
            "format_version": 1,
            "dataset_root": str(args.dataset_root.resolve()),
            "train_dates": fold["train_dates"],
            "selection_sha256_by_date": {
                date: selection_hash(selected_values(order_ids_by_date[date], train_selection[date]))
                for date in fold["train_dates"]
            },
            "source_files": {
                date: {
                    "size": (args.dataset_root / f"day={date}.parquet").stat().st_size,
                    "mtime_ns": (args.dataset_root / f"day={date}.parquet").stat().st_mtime_ns,
                }
                for date in fold["train_dates"]
            },
            "max_seq_len": args.max_seq_len,
            "link_id_min_count": args.link_id_min_count,
        }
        fit_config_hash = stable_hash(fit_config)
        old_fold_manifest_path = fold_root / "manifest.json"
        old_fold_manifest = json.loads(old_fold_manifest_path.read_text(encoding="utf-8")) if old_fold_manifest_path.exists() else {}
        can_reuse_metadata = (
            args.skip_existing
            and old_fold_manifest.get("fit_config_sha256") == fit_config_hash
            and (fold_root / "metadata.json").exists()
        )
        if can_reuse_metadata:
            metadata = DeepV3Metadata.from_json(json.loads((fold_root / "metadata.json").read_text(encoding="utf-8")))
            fit_summary = old_fold_manifest["fit_summary"]
        else:
            metadata, fit_summary = fit_metadata(
                args.dataset_root, fold["train_dates"], train_selection, args.max_seq_len, args.link_id_min_count
            )
        metadata_payload = safe_float(metadata.to_json())
        metadata_hash = stable_hash(metadata_payload)
        (fold_root / "metadata.json").write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
        split_specs = {
            "train": (fold["train_dates"], train_selection),
            "validation": ([fold["validation_date"]], select_orders({fold["validation_date"]: order_ids_by_date[fold["validation_date"]]}, None if args.max_eval_orders <= 0 else args.max_eval_orders, args.seed + fold_id)),
            "test": ([fold["test_date"]], select_orders({fold["test_date"]: order_ids_by_date[fold["test_date"]]}, None if args.max_eval_orders <= 0 else args.max_eval_orders, args.seed + fold_id)),
        }
        fold_manifest = {"fit_summary": fit_summary, "splits": {}}
        for split, (dates, selections) in split_specs.items():
            split_rows = []
            for date in dates:
                day_root = fold_root / split / f"day={date}"
                selected_hash = selection_hash(selected_values(order_ids_by_date[date], selections[date]))
                shard = reusable_shard(day_root, metadata_hash, selected_hash, args.feature_dtype) if args.skip_existing else None
                if shard is None:
                    frame = read_day(args.dataset_root, date, selections[date], args.max_seq_len)
                    shard = write_shard(
                        day_root, frame, metadata, metadata_hash, selected_hash,
                        date, args.feature_dtype, args.skip_existing,
                    )
                    del frame
                split_rows.append(shard)
                print(f"fold={fold_id} split={split} date={date} orders={shard['orders']} rows={shard['rows']}", flush=True)
            fold_manifest["splits"][split] = split_rows
        fold_manifest["metadata_sha256"] = metadata_hash
        fold_manifest["fit_config_sha256"] = fit_config_hash
        (fold_root / "manifest.json").write_text(json.dumps(safe_float(fold_manifest), indent=2), encoding="utf-8")
        root_manifest["folds"][str(fold_id)] = fold_manifest
    (args.output_root / "manifest.json").write_text(json.dumps(safe_float(root_manifest), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
