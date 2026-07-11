"""Stream first/last GPS observations per order from nested daily archives.

This extraction does not retain point-level data and does not rerun map matching.
Raw coordinates are preserved and a GCJ-02 -> WGS84 conversion is added using
the same empirically selected interpretation as the Stage0 production pipeline.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from run_full_day_2017 import gcj02_to_wgs84
from run_monthly_stage0 import archive_days


COLUMNS = ["driver_id", "order_id", "timestamp", "source_lon", "source_lat"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("stage0/output/order_od"))
    parser.add_argument("--start-date", default="20161009")
    parser.add_argument("--end-date", default="20161019")
    parser.add_argument("--unrar", type=Path, default=Path(r"C:\Program Files\WinRAR\UnRAR.exe"))
    parser.add_argument("--work-root", type=Path, default=Path("stage0/output/_od_work"))
    parser.add_argument("--chunk-rows", type=int, default=1_000_000)
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def extract_nested(archive: Path, member: str, work: Path, unrar: Path) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    nested = work / Path(member).name
    if not nested.exists():
        subprocess.run([
            str(unrar), "e", "-o+", "-idq", str(archive.resolve()), member.replace("/", "\\"), str(work.resolve()) + "\\",
        ], check=True)
    return nested


def chunk_endpoints(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk.dropna(subset=["order_id", "timestamp", "source_lon", "source_lat"])
    chunk["timestamp"] = pd.to_numeric(chunk["timestamp"], errors="coerce")
    chunk = chunk.dropna(subset=["timestamp"])
    minimum = chunk.loc[chunk.groupby("order_id", sort=False)["timestamp"].idxmin()].copy()
    maximum = chunk.loc[chunk.groupby("order_id", sort=False)["timestamp"].idxmax()].copy()
    minimum = minimum.rename(columns={
        "driver_id": "origin_driver_id", "timestamp": "origin_timestamp",
        "source_lon": "origin_source_lon", "source_lat": "origin_source_lat",
    })[["order_id", "origin_driver_id", "origin_timestamp", "origin_source_lon", "origin_source_lat"]]
    maximum = maximum.rename(columns={
        "driver_id": "destination_driver_id", "timestamp": "destination_timestamp",
        "source_lon": "destination_source_lon", "source_lat": "destination_source_lat",
    })[["order_id", "destination_driver_id", "destination_timestamp", "destination_source_lon", "destination_source_lat"]]
    counts = chunk.groupby("order_id", sort=False).size().rename("raw_point_count").reset_index()
    return minimum.merge(maximum, on="order_id", how="outer").merge(counts, on="order_id", how="outer")


def reduce_summaries(parts: list[pd.DataFrame], date: str) -> pd.DataFrame:
    combined = pd.concat(parts, ignore_index=True)
    origin = combined.loc[combined.groupby("order_id", sort=False)["origin_timestamp"].idxmin(), [
        "order_id", "origin_driver_id", "origin_timestamp", "origin_source_lon", "origin_source_lat",
    ]]
    destination = combined.loc[combined.groupby("order_id", sort=False)["destination_timestamp"].idxmax(), [
        "order_id", "destination_driver_id", "destination_timestamp", "destination_source_lon", "destination_source_lat",
    ]]
    counts = combined.groupby("order_id", sort=False)["raw_point_count"].sum().reset_index()
    output = origin.merge(destination, on="order_id", how="outer").merge(counts, on="order_id", how="outer")
    output["driver_id"] = output["origin_driver_id"].fillna(output["destination_driver_id"])
    origin_lon, origin_lat = gcj02_to_wgs84(output["origin_source_lon"].to_numpy(), output["origin_source_lat"].to_numpy())
    destination_lon, destination_lat = gcj02_to_wgs84(output["destination_source_lon"].to_numpy(), output["destination_source_lat"].to_numpy())
    output["origin_lon"] = origin_lon
    output["origin_lat"] = origin_lat
    output["destination_lon"] = destination_lon
    output["destination_lat"] = destination_lat
    output["duration_sec"] = output["destination_timestamp"] - output["origin_timestamp"]
    output["date"] = date
    output["od_source"] = "first_last_gps_observation"
    output["coordinate_interpretation"] = "raw_gcj02_converted_to_wgs84"
    columns = [
        "order_id", "driver_id", "date", "raw_point_count", "origin_timestamp", "destination_timestamp", "duration_sec",
        "origin_source_lon", "origin_source_lat", "destination_source_lon", "destination_source_lat",
        "origin_lon", "origin_lat", "destination_lon", "destination_lat", "od_source", "coordinate_interpretation",
    ]
    return output[columns].sort_values("order_id").reset_index(drop=True)


def process_day(nested: Path, date: str, chunk_rows: int) -> tuple[pd.DataFrame, int]:
    parts = []
    input_rows = 0
    reader = pd.read_csv(
        nested, compression="tar", header=None, names=COLUMNS, chunksize=chunk_rows,
        dtype={"driver_id": "string", "order_id": "string", "source_lon": "float64", "source_lat": "float64"},
    )
    for number, chunk in enumerate(reader, start=1):
        input_rows += len(chunk)
        parts.append(chunk_endpoints(chunk))
        if number % 10 == 0:
            print(f"OD day={date} chunks={number} rows={input_rows:,}", flush=True)
    if not parts:
        raise RuntimeError(f"no GPS rows found in {nested}")
    return reduce_summaries(parts, date), input_rows


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    entries = [row for row in archive_days(args.archive) if args.start_date <= row[0] <= args.end_date]
    if not entries:
        raise ValueError("no requested daily archives found")
    manifests = []
    for date, member in entries:
        target = args.output_root / f"day={date}.parquet"
        manifest_path = args.output_root / f"day={date}.json"
        if target.exists() and manifest_path.exists() and not args.force:
            manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
            print(f"OD day={date} already complete", flush=True)
            continue
        work = args.work_root / f"day={date}"
        nested = extract_nested(args.archive, member, work, args.unrar)
        output, input_rows = process_day(nested, date, args.chunk_rows)
        output.to_parquet(target, index=False, compression="zstd")
        manifest = {
            "date": date, "archive_member": member, "input_rows": input_rows,
            "orders": int(len(output)), "complete": True,
            "od_source": "first_last_gps_observation",
            "coordinate_interpretation": "raw_gcj02_converted_to_wgs84",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifests.append(manifest)
        if not args.keep_work:
            shutil.rmtree(work, ignore_errors=True)
        print(json.dumps(manifest, indent=2), flush=True)
    all_manifests = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.output_root.glob("day=*.json"))
    ]
    (args.output_root / "order_od_manifest.json").write_text(json.dumps({"days": all_manifests}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
