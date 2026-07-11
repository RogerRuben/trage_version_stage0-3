"""Validate Stage2 Deep v3 mmap tensor shard contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-shard-root", type=Path, default=Path("stage2/output/deep_v3_tensor_shards_5k"))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def audit_day(day_root: Path, fold_metadata_hash: str) -> dict:
    manifest = json.loads((day_root / "manifest.json").read_text(encoding="utf-8"))
    arrays = {
        name: np.load(day_root / f"{name}.npy", mmap_mode="r")
        for name in ["static_numeric", "dynamic", "categorical", "target", "tail", "mask", "offsets", "lengths"]
    }
    rows = int(manifest["rows"])
    orders = int(manifest["orders"])
    errors = []
    for name in ["static_numeric", "dynamic", "categorical", "target", "tail", "mask"]:
        if len(arrays[name]) != rows:
            errors.append(f"{name}: first dimension {len(arrays[name])} != rows {rows}")
    if len(arrays["offsets"]) != orders + 1:
        errors.append("offset count does not equal orders + 1")
    if len(arrays["lengths"]) != orders:
        errors.append("length count does not equal orders")
    if arrays["offsets"][0] != 0 or arrays["offsets"][-1] != rows:
        errors.append("offset boundaries do not match [0, rows]")
    if not np.array_equal(np.diff(arrays["offsets"]), arrays["lengths"]):
        errors.append("lengths do not equal diff(offsets)")
    for name in ["static_numeric", "dynamic", "target"]:
        if not np.isfinite(arrays[name]).all():
            errors.append(f"{name} contains non-finite values")
    id_rows = pq.ParquetFile(day_root / "ids.parquet").metadata.num_rows
    if id_rows != rows:
        errors.append(f"ids rows {id_rows} != rows {rows}")
    if manifest.get("metadata_sha256") != fold_metadata_hash:
        errors.append("day metadata fingerprint differs from fold metadata fingerprint")
    return {
        "day_root": str(day_root),
        "date": manifest.get("date"),
        "rows": rows,
        "orders": orders,
        "feature_dtype": manifest.get("feature_dtype"),
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
    }


def main() -> None:
    args = parse_args()
    folds = []
    for fold_root in sorted(args.tensor_shard_root.glob("fold=*")):
        fold_manifest = json.loads((fold_root / "manifest.json").read_text(encoding="utf-8"))
        days = []
        for split in ["train", "validation", "test"]:
            for day_root in sorted((fold_root / split).glob("day=*")):
                result = audit_day(day_root, fold_manifest["metadata_sha256"])
                result["split"] = split
                days.append(result)
        folds.append({
            "fold": fold_root.name.split("=", 1)[-1],
            "status": "PASS" if all(day["status"] == "PASS" for day in days) else "FAIL",
            "days": days,
        })
    report = {
        "tensor_shard_root": str(args.tensor_shard_root),
        "status": "PASS" if folds and all(fold["status"] == "PASS" for fold in folds) else "FAIL",
        "folds": folds,
    }
    output = args.output or args.tensor_shard_root / "audit_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "folds": {fold["fold"]: fold["status"] for fold in folds}}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
