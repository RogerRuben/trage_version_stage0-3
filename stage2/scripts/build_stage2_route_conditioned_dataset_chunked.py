"""Chunked route-conditioned dataset builder for full-day inference.

The standard builder materializes a full day before selecting columns.  For a
full 2016-10-23 day this can require several GB of contiguous RAM.  This
script applies the same estimated-time column contract to parquet parts and
streams them into one daily parquet file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_stage2_route_conditioned_dataset import (  # noqa: E402
    add_common_route_columns,
    add_time_columns,
    normalize_timestamp_precision,
    select_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dates", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=["estimated"], default="estimated")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _source_parts(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    parts = sorted(path.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(path)
    return parts


def main() -> None:
    args = parse_args()
    dates = [part.strip() for part in args.dates.split(",") if part.strip()]
    out_day_root = args.output_root / "estimated_time_daily"
    out_day_root.mkdir(parents=True, exist_ok=True)
    manifest = {"dates": dates, "days": {}, "mode": args.mode, "chunked": True}
    schema_meta = None
    for date in dates:
        source_path = args.source_root / f"day={date}.parquet"
        output_path = out_day_root / f"day={date}.parquet"
        if output_path.exists():
            if not args.overwrite:
                raise FileExistsError(f"{output_path} exists; pass --overwrite")
            output_path.unlink()
        started = time.time()
        writer: pq.ParquetWriter | None = None
        total_rows = 0
        order_ids: set[str] = set()
        time_checks = []
        try:
            for index, part_path in enumerate(_source_parts(source_path)):
                frame = pd.read_parquet(part_path)
                frame = add_common_route_columns(frame)
                frame = add_time_columns(frame, args.mode)
                selected, metadata = select_columns(frame, args.mode)
                selected = normalize_timestamp_precision(selected)
                total_rows += int(len(selected))
                order_ids.update(selected["order_id"].astype(str).unique().tolist())
                if "route_conditioned_time_check" in selected:
                    time_checks.append(float(selected["route_conditioned_time_check"].mean()))
                table = pa.Table.from_pandas(selected, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
                    schema_meta = metadata
                else:
                    table = table.cast(writer.schema)
                writer.write_table(table)
                print(f"route-conditioned chunk day={date} part={index} rows={len(selected)}", flush=True)
        finally:
            if writer is not None:
                writer.close()
        day_meta = {
            "rows": total_rows,
            "orders": len(order_ids),
            "estimated_time_check_ratio": float(sum(time_checks) / len(time_checks)) if time_checks else 0.0,
            "output": str(output_path),
            "runtime_sec": time.time() - started,
        }
        manifest["days"][date] = day_meta
        print(f"route-conditioned chunked day={date} {day_meta}", flush=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "route_conditioned_dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if schema_meta is not None:
        (args.output_root / "route_conditioned_estimated_time_schema.json").write_text(json.dumps(schema_meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
