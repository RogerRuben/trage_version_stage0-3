"""Deterministically sample complete orders from nested raw daily archives.

The source daily files are much larger than the canonical smoke.  This script
uses two streaming passes: the first retains only the smallest stable order
hashes, and the second writes every row for those orders in bounded batches.
No full-day frame is materialized in RAM.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


COLUMNS = ("driver_id", "order_id", "timestamp", "lon", "lat")
SCHEMA = pa.schema([
    ("driver_id", pa.string()),
    ("order_id", pa.string()),
    ("timestamp", pa.int64()),
    ("lon", pa.float64()),
    ("lat", pa.float64()),
])


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--member", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--orders", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--batch-rows", type=int, default=100_000)
    return parser.parse_args()


def lines(source: Path, member: str):
    with tarfile.open(source, mode="r:gz") as archive:
        handle = archive.extractfile(member)
        if handle is None:
            raise FileNotFoundError(f"member {member!r} absent from {source}")
        for raw in handle:
            yield raw.decode("utf-8", errors="strict").rstrip("\r\n")


def stable_rank(seed: int, date: str, order_id: str) -> int:
    payload = f"{seed}|{date}|{order_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def select_orders(args: argparse.Namespace) -> tuple[set[str], int, int]:
    heap: list[tuple[int, str]] = []
    seen: set[str] = set()
    raw_rows = 0
    malformed = 0
    for line in lines(args.source, args.member):
        raw_rows += 1
        fields = line.split(",")
        if len(fields) != 5:
            malformed += 1
            continue
        order_id = fields[1]
        if order_id in seen:
            continue
        seen.add(order_id)
        rank = stable_rank(args.seed, args.date, order_id)
        item = (-rank, order_id)
        if len(heap) < args.orders:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    if len(heap) < args.orders:
        raise ValueError(f"requested {args.orders} orders but source has only {len(seen)}")
    return {order_id for _, order_id in heap}, raw_rows, malformed


def parse_row(line: str) -> dict[str, object] | None:
    fields = line.split(",")
    if len(fields) != 5:
        return None
    try:
        return {
            "driver_id": fields[0],
            "order_id": fields[1],
            "timestamp": int(fields[2]),
            "lon": float(fields[3]),
            "lat": float(fields[4]),
        }
    except ValueError:
        return None


def write_selected(args: argparse.Namespace, selected: set[str]) -> tuple[int, int]:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(args.output, SCHEMA, compression="zstd")
    buffer: dict[str, list] = {column: [] for column in COLUMNS}
    written = 0
    malformed = 0
    try:
        for line in lines(args.source, args.member):
            fields = line.split(",", 2)
            if len(fields) < 3 or fields[1] not in selected:
                continue
            row = parse_row(line)
            if row is None:
                malformed += 1
                continue
            for column in COLUMNS:
                buffer[column].append(row[column])
            if len(buffer["order_id"]) >= args.batch_rows:
                writer.write_table(pa.Table.from_pydict(buffer, schema=SCHEMA))
                written += len(buffer["order_id"])
                buffer = {column: [] for column in COLUMNS}
        if buffer["order_id"]:
            writer.write_table(pa.Table.from_pydict(buffer, schema=SCHEMA))
            written += len(buffer["order_id"])
    finally:
        writer.close()
    return written, malformed


def main() -> None:
    args = arguments()
    if args.orders <= 0:
        raise ValueError("--orders must be positive")
    selected, source_rows, malformed_first_pass = select_orders(args)
    written, malformed_second_pass = write_selected(args, selected)
    table = pq.read_table(args.output, columns=["order_id", "timestamp"])
    orders = table.column("order_id").to_pylist()
    timestamps = table.column("timestamp").to_numpy()
    audit = {
        "status": "PASS" if len(set(orders)) == args.orders and written > 0 else "FAIL",
        "date": args.date,
        "source": args.source.as_posix(),
        "member": args.member,
        "seed": args.seed,
        "sampling": "smallest_sha256_order_rank_complete_order_two_pass",
        "source_rows": source_rows,
        "selected_orders": len(set(orders)),
        "selected_rows": written,
        "malformed_first_pass": malformed_first_pass,
        "malformed_selected_rows": malformed_second_pass,
        "minimum_timestamp": int(timestamps.min()),
        "maximum_timestamp": int(timestamps.max()),
        "peak_buffer_rows": args.batch_rows,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
