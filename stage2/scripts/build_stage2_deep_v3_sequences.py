"""Build manifests for Stage2 Deep v3 route-conditioned sequence inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage2_deep_v3_utils import build_metadata, load_fold_config, read_dates, safe_float  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("stage2/output/route_conditioned_dataset/estimated_time_daily"))
    parser.add_argument("--fold-config", type=Path, default=Path("rolling_threefold_config.json"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/deep_v3/data_manifests"))
    parser.add_argument("--max-seq-len", type=int, default=160)
    parser.add_argument("--link-id-min-count", type=int, default=5)
    parser.add_argument("--sample-train-orders", type=int, default=0, help="0 means all available train orders.")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def describe(frame: pd.DataFrame, max_seq_len: int) -> dict:
    lengths = frame.groupby("order_id").size()
    return {
        "orders": int(lengths.size),
        "rows": int(len(frame)),
        "mean_links_per_order": float(lengths.mean()),
        "p50_links_per_order": float(lengths.quantile(0.50)),
        "p90_links_per_order": float(lengths.quantile(0.90)),
        "max_links_per_order_after_truncation": int(lengths.max()) if len(lengths) else 0,
        "truncated_order_upper_bound_ratio": float(lengths.ge(max_seq_len).mean()) if len(lengths) else 0.0,
    }


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    folds = load_fold_config(args.fold_config)
    manifest = {
        "dataset_root": str(args.dataset_root),
        "fold_config": str(args.fold_config),
        "max_seq_len": args.max_seq_len,
        "link_id_min_count": args.link_id_min_count,
        "sample_train_orders": args.sample_train_orders,
        "folds": {},
    }
    for fold in folds:
        fold_id = int(fold["fold"])
        train = read_dates(
            args.dataset_root,
            fold["train_dates"],
            None if args.sample_train_orders <= 0 else args.sample_train_orders,
            args.seed + fold_id,
            args.max_seq_len,
        )
        validation = read_dates(args.dataset_root, [fold["validation_date"]], None, args.seed + fold_id, args.max_seq_len)
        test = read_dates(args.dataset_root, [fold["test_date"]], None, args.seed + fold_id, args.max_seq_len)
        metadata = build_metadata(train, args.max_seq_len, args.link_id_min_count)
        fold_dir = args.output_root / f"fold={fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / "metadata.json").write_text(json.dumps(safe_float(metadata.to_json()), indent=2), encoding="utf-8")
        manifest["folds"][str(fold_id)] = {
            "train_dates": fold["train_dates"],
            "validation_date": fold["validation_date"],
            "test_date": fold["test_date"],
            "train": describe(train, args.max_seq_len),
            "validation": describe(validation, args.max_seq_len),
            "test": describe(test, args.max_seq_len),
            "static_numeric_columns": metadata.static_numeric_columns,
            "dynamic_windows": list(metadata.dynamic_columns_by_window),
            "categorical_columns": metadata.categorical_columns,
        }
        print(f"fold={fold_id} train_orders={manifest['folds'][str(fold_id)]['train']['orders']} test_orders={manifest['folds'][str(fold_id)]['test']['orders']}", flush=True)
    (args.output_root / "deep_v3_sequence_manifest.json").write_text(json.dumps(safe_float(manifest), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
