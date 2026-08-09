"""Stream the narrow Train-only input used by v5.2 support/static fitting."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


COLUMNS = (
    "split", "date", "order_id", "traversal_id", "observed_directed_edge_uid",
    "route_sequence", "canonical_highway", "road_class", "observed_direction",
    "bridge", "tunnel", "synthetic_reverse_edge", "osm_direction_disagreement",
)


def build_train_artifact_input(
    *, route_feature_root: Path, dates: tuple[str, ...], output: Path,
    batch_size: int = 250_000,
) -> tuple[int, int]:
    """Write one atomic, column-projected parquet without accumulating days in memory."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    writer: pq.ParquetWriter | None = None
    row_count = 0
    batch_count = 0
    try:
        for date in dates:
            source = route_feature_root / f"day={date}.parquet"
            if not source.is_file():
                raise FileNotFoundError(source)
            parquet = pq.ParquetFile(source)
            missing = sorted(set(COLUMNS) - set(parquet.schema_arrow.names))
            if missing:
                raise ValueError(f"{source} is missing columns: {missing}")
            for batch in parquet.iter_batches(batch_size=batch_size, columns=list(COLUMNS)):
                if not pc.all(pc.equal(batch.column("split"), "train")).as_py():
                    raise ValueError(f"{source} contains non-Train rows")
                if not pc.all(pc.equal(batch.column("date"), date)).as_py():
                    raise ValueError(f"{source} contains rows outside date {date}")
                table = pa.Table.from_batches([batch]).append_column(
                    "row_id",
                    pa.array(np.arange(row_count, row_count + batch.num_rows, dtype=np.int64)),
                )
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
                writer.write_table(table)
                row_count += table.num_rows
                batch_count += 1
        if writer is None or row_count == 0:
            raise ValueError("no Train rows were written")
    finally:
        if writer is not None:
            writer.close()
    os.replace(temporary, output)
    return row_count, batch_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-feature-root", required=True, type=Path)
    parser.add_argument("--dates", required=True, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=250_000)
    args = parser.parse_args()
    rows, batches = build_train_artifact_input(
        route_feature_root=args.route_feature_root,
        dates=tuple(args.dates),
        output=args.output,
        batch_size=args.batch_size,
    )
    print({"status": "PASS", "rows": rows, "batches": batches, "output": str(args.output)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
