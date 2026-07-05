"""Stream a large headerless DiDi GPS file and extract complete sample orders."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd


COLUMNS = ["driver_id", "order_id", "timestamp", "lon", "lat"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-local", default="2016-10-01 17:00:00")
    parser.add_argument("--hours", type=float, default=2.0)
    parser.add_argument("--orders", type=int, default=500)
    parser.add_argument("--min-window-points", type=int, default=10)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    return parser.parse_args()


def stable_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(args.start_local, tz="Asia/Shanghai")
    end = start + pd.Timedelta(hours=args.hours)
    start_epoch = int(start.timestamp())
    end_epoch = int(end.timestamp())

    counts: Counter[str] = Counter()
    full_rows = 0
    window_rows = 0
    invalid_city_bounds = 0
    min_ts = None
    max_ts = None

    reader = pd.read_csv(
        args.input,
        header=None,
        names=COLUMNS,
        chunksize=args.chunksize,
        dtype={"driver_id": "string", "order_id": "string"},
    )
    for i, chunk in enumerate(reader, start=1):
        full_rows += len(chunk)
        chunk_min = int(chunk["timestamp"].min())
        chunk_max = int(chunk["timestamp"].max())
        min_ts = chunk_min if min_ts is None else min(min_ts, chunk_min)
        max_ts = chunk_max if max_ts is None else max(max_ts, chunk_max)
        in_city = chunk["lon"].between(108.5, 109.5) & chunk["lat"].between(33.5, 35.0)
        invalid_city_bounds += int((~in_city).sum())
        mask = chunk["timestamp"].between(start_epoch, end_epoch - 1)
        selected = chunk.loc[mask, "order_id"]
        window_rows += len(selected)
        counts.update(selected.astype(str).value_counts().to_dict())
        print(f"pass1 chunk={i} rows={full_rows:,} window_rows={window_rows:,}", flush=True)

    eligible = [order for order, n in counts.items() if n >= args.min_window_points]
    eligible.sort(key=stable_key)
    chosen = eligible[: args.orders]
    chosen_set = set(chosen)
    if len(chosen) < args.orders:
        print(f"warning: requested {args.orders} orders, only {len(chosen)} eligible")

    frames: list[pd.DataFrame] = []
    reader = pd.read_csv(
        args.input,
        header=None,
        names=COLUMNS,
        chunksize=args.chunksize,
        dtype={"driver_id": "string", "order_id": "string"},
    )
    extracted_rows = 0
    for i, chunk in enumerate(reader, start=1):
        selected = chunk[chunk["order_id"].isin(chosen_set)]
        if not selected.empty:
            selected = selected.copy()
            selected["source_row"] = selected.index.astype("int64")
            frames.append(selected)
            extracted_rows += len(selected)
        print(f"pass2 chunk={i} extracted_rows={extracted_rows:,}", flush=True)

    sample = pd.concat(frames, ignore_index=True)
    sample["local_datetime"] = (
        pd.to_datetime(sample["timestamp"], unit="s", utc=True)
        .dt.tz_convert("Asia/Shanghai")
        .dt.tz_localize(None)
    )
    sample = sample.sort_values(["order_id", "timestamp"], kind="stable").reset_index(drop=True)
    sample.to_parquet(out / "sample_raw.parquet", index=False)
    sample.to_csv(out / "sample_raw.csv.gz", index=False, compression="gzip")

    metadata = {
        "source": str(args.input.resolve()),
        "inferred_schema": COLUMNS,
        "schema_reason": "id1 has driver-like cardinality; id2 has order-like cardinality",
        "timezone": "Asia/Shanghai",
        "window_start_local": start.isoformat(),
        "window_end_local": end.isoformat(),
        "source_rows": full_rows,
        "source_min_local": pd.to_datetime(min_ts, unit="s", utc=True).tz_convert("Asia/Shanghai").isoformat(),
        "source_max_local": pd.to_datetime(max_ts, unit="s", utc=True).tz_convert("Asia/Shanghai").isoformat(),
        "source_outside_broad_xian_bounds": invalid_city_bounds,
        "window_rows": window_rows,
        "window_orders": len(counts),
        "eligible_orders": len(eligible),
        "sample_orders": int(sample["order_id"].nunique()),
        "sample_drivers": int(sample["driver_id"].nunique()),
        "sample_rows": len(sample),
        "sample_bounds_wgs84": {
            "west": float(sample["lon"].min()),
            "south": float(sample["lat"].min()),
            "east": float(sample["lon"].max()),
            "north": float(sample["lat"].max()),
        },
    }
    (out / "sample_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
