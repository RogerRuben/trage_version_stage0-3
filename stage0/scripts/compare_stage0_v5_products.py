#!/usr/bin/env python
"""Compare two Stage 0 v5 product snapshots by sorted field values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PRODUCT_KEYS = {
    "order_base": ["order_id"],
    "route_parts": ["order_id", "route_sequence"],
    "link_traversals": ["order_id", "traversal_id"],
    "turn_movements": ["order_id", "movement_sequence"],
    "route_quality": ["order_id"],
}


def _read_product(root: Path, product: str) -> pd.DataFrame:
    files = sorted((root / product).rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files for {product} under {root}")
    frame = pd.concat(
        [pd.read_parquet(path) for path in files],
        ignore_index=True,
        sort=False,
    )
    keys = [column for column in PRODUCT_KEYS[product] if column in frame]
    columns = sorted(frame.columns)
    return frame.loc[:, columns].sort_values(keys, kind="stable").reset_index(drop=True)


def _column_equal(left: pd.Series, right: pd.Series, tolerance: float) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
        a = pd.to_numeric(left, errors="coerce").to_numpy(float)
        b = pd.to_numeric(right, errors="coerce").to_numpy(float)
        return np.isclose(a, b, rtol=0.0, atol=tolerance, equal_nan=True)
    a = left.astype("string").fillna("<NA>").to_numpy()
    b = right.astype("string").fillna("<NA>").to_numpy()
    return a == b


def compare(left_root: Path, right_root: Path, tolerance: float) -> dict[str, Any]:
    report: dict[str, Any] = {
        "left_root": str(left_root.resolve()),
        "right_root": str(right_root.resolve()),
        "numeric_absolute_tolerance": tolerance,
        "products": {},
    }
    for product, keys in PRODUCT_KEYS.items():
        left = _read_product(left_root, product)
        right = _read_product(right_root, product)
        product_report: dict[str, Any] = {
            "left_rows": len(left),
            "right_rows": len(right),
            "schema_equal": list(left.columns) == list(right.columns),
            "mismatched_columns": {},
        }
        if len(left) != len(right) or not product_report["schema_equal"]:
            product_report["equal"] = False
            report["products"][product] = product_report
            continue
        mismatch_mask = np.zeros(len(left), dtype=bool)
        for column in left.columns:
            equal = _column_equal(left[column], right[column], tolerance)
            count = int((~equal).sum())
            if count:
                product_report["mismatched_columns"][column] = count
                mismatch_mask |= ~equal
        product_report["mismatched_rows"] = int(mismatch_mask.sum())
        if mismatch_mask.any() and "order_id" in left:
            product_report["example_order_ids"] = (
                left.loc[mismatch_mask, "order_id"].astype(str).drop_duplicates().head(20).tolist()
            )
        product_report["equal"] = not bool(mismatch_mask.any())
        report["products"][product] = product_report
    report["all_products_equal"] = all(
        value["equal"] for value in report["products"].values()
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-root", type=Path, required=True)
    parser.add_argument("--right-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--numeric-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    result = compare(args.left_root, args.right_root, args.numeric_tolerance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["all_products_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
