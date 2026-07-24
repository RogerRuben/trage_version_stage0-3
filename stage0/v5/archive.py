"""One-day-at-a-time RAR inventory, order prescan, and stable sampling."""

from __future__ import annotations

import json
import hashlib
import re
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from .config import Stage0Config, stable_hash
from .manifest import base_manifest, write_manifest


COLUMNS = ["driver_id", "order_id", "timestamp", "lon", "lat"]
DATE_RE = re.compile(r"(?:^|[\\/])10-(\d{1,2})[\\/].*\.tar\.gz$", re.IGNORECASE)


def sampling_run_id(dates: list[str], orders_per_day: int, seed: int) -> str:
    """Return a stable namespace so gate-specific samples cannot overwrite one another."""
    payload = json.dumps(
        {"dates": sorted(map(str, dates)), "orders_per_day": int(orders_per_day), "seed": int(seed)},
        sort_keys=True,
    )
    return f"orders={int(orders_per_day)}__dates={hashlib.sha256(payload.encode()).hexdigest()[:12]}"


def sampled_orders_path(manifest_root: Path, run_id: str, date: str) -> Path:
    return manifest_root / "sampling_runs" / run_id / "daily_sampled_orders" / f"day={date}.parquet"


def list_archive_members(archive: Path, seven_zip: Path) -> list[dict[str, Any]]:
    result = subprocess.run(
        [str(seven_zip), "l", "-ba", str(archive)], check=True, capture_output=True
    )
    text = result.stdout.decode("utf-8", errors="replace")
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = re.match(r"(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})\s+\S+\s+(\d+)\s+(\d+)\s+(.+)$", line.strip())
        if not match:
            continue
        name = match.group(5).strip()
        date_match = DATE_RE.search(name)
        if not date_match:
            continue
        date = f"201610{int(date_match.group(1)):02d}"
        rows.append({
            "date": date, "source_member": name,
            "uncompressed_size": int(match.group(3)), "compressed_size": int(match.group(4)),
            "archive_timestamp": f"{match.group(1)}T{match.group(2)}",
        })
    return sorted(rows, key=lambda row: row["date"])


def extract_daily_archive(
    archive: Path, member: str, seven_zip: Path, work_root: Path, force: bool = False
) -> Path:
    date_match = DATE_RE.search(member)
    if not date_match:
        raise ValueError(f"unable to infer date from member: {member}")
    date = f"201610{int(date_match.group(1)):02d}"
    target_dir = work_root / "archive_cache" / f"day={date}"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "source.tar.gz"
    if target.exists() and target.stat().st_size > 0 and not force:
        return target
    temporary = target.with_suffix(".tar.gz.tmp")
    with temporary.open("wb") as stream:
        process = subprocess.run(
            [str(seven_zip), "e", "-so", str(archive), member], stdout=stream, stderr=subprocess.PIPE
        )
    if process.returncode != 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(process.stderr.decode("utf-8", errors="replace"))
    temporary.replace(target)
    return target


def daily_csv_member(tar_path: Path) -> str:
    with tarfile.open(tar_path, "r:gz") as archive:
        candidates = [member for member in archive.getmembers() if member.isfile() and member.size > 0]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one GPS file in {tar_path}, found {len(candidates)}")
    return candidates[0].name


def iter_daily_chunks(tar_path: Path, chunksize: int = 500_000) -> Iterator[pd.DataFrame]:
    member_name = daily_csv_member(tar_path)
    with tarfile.open(tar_path, "r:gz") as archive:
        stream = archive.extractfile(member_name)
        if stream is None:
            raise RuntimeError(f"cannot stream {member_name}")
        reader = pd.read_csv(
            stream, header=None, names=COLUMNS, chunksize=chunksize,
            dtype={"driver_id": "string", "order_id": "string"},
        )
        for chunk in reader:
            yield chunk


def scan_daily_orders(tar_path: Path, date: str, source_member: str) -> pd.DataFrame:
    """Vectorized per-chunk summaries followed by a vectorized cross-chunk merge."""
    summaries: list[pd.DataFrame] = []
    for chunk in iter_daily_chunks(tar_path):
        chunk["timestamp"] = pd.to_numeric(chunk.timestamp, errors="coerce")
        chunk["lon"] = pd.to_numeric(chunk.lon, errors="coerce")
        chunk["lat"] = pd.to_numeric(chunk.lat, errors="coerce")
        chunk["valid_coordinate"] = chunk.lon.between(-180, 180) & chunk.lat.between(-90, 90)
        chunk = chunk.loc[chunk.order_id.notna() & chunk.order_id.astype(str).str.strip().ne("")]
        base = chunk.groupby("order_id", sort=False).agg(
            driver_id=("driver_id", "first"), point_count=("order_id", "size")
        )
        valid = chunk.loc[chunk.valid_coordinate & chunk.timestamp.notna()].sort_values("timestamp", kind="stable")
        if len(valid):
            valid_summary = valid.groupby("order_id", sort=False).agg(
                start_time=("timestamp", "first"), end_time=("timestamp", "last"),
                valid_point_count=("order_id", "size"), start_lon=("lon", "first"),
                start_lat=("lat", "first"), end_lon=("lon", "last"), end_lat=("lat", "last"),
            )
            base = base.join(valid_summary, how="left")
        else:
            for column in ("start_time", "end_time", "valid_point_count", "start_lon", "start_lat", "end_lon", "end_lat"):
                base[column] = np.nan
        summaries.append(base.reset_index())
    if not summaries:
        return pd.DataFrame()
    combined = pd.concat(summaries, ignore_index=True)
    totals = combined.groupby("order_id", sort=False).agg(
        driver_id=("driver_id", "first"), point_count=("point_count", "sum"),
        valid_point_count=("valid_point_count", "sum"), start_time=("start_time", "min"),
        end_time=("end_time", "max"),
    )
    start_coordinates = combined.sort_values("start_time", kind="stable").groupby("order_id", sort=False)[["start_lon", "start_lat"]].first()
    end_coordinates = combined.sort_values("end_time", ascending=False, kind="stable").groupby("order_id", sort=False)[["end_lon", "end_lat"]].first()
    rows = totals.join(start_coordinates).join(end_coordinates).reset_index()
    if rows.empty:
        return rows
    rows["date"] = date
    rows["source_member"] = source_member
    rows["valid_point_count"] = rows.valid_point_count.fillna(0).astype("int64")
    rows["duration_s"] = rows.end_time - rows.start_time
    rows["time_order_valid"] = rows.end_time.gt(rows.start_time)
    rows["eligible"] = rows.valid_point_count.ge(3) & rows.time_order_valid
    reason = np.select(
        [rows.valid_point_count.eq(0), rows.valid_point_count.lt(3), ~rows.time_order_valid],
        ["all_coordinates_invalid", "too_few_valid_points", "nonpositive_or_unordered_time"],
        default="",
    )
    rows["ineligibility_reason"] = reason
    return rows.sort_values("order_id").reset_index(drop=True)


def stable_sample(orders: pd.DataFrame, date: str, count: int, seed: int) -> pd.DataFrame:
    eligible = orders.loc[orders.eligible].copy()
    eligible["sample_hash"] = eligible.order_id.map(lambda order: stable_hash(date, order, seed=seed))
    eligible = eligible.sort_values(["sample_hash", "order_id"]).head(min(count, len(eligible))).copy()
    probability = min(1.0, count / max(len(orders.loc[orders.eligible]), 1))
    eligible["sampling_probability"] = probability
    eligible["sampling_weight"] = 1.0 / probability
    eligible["sampling_seed"] = seed
    return eligible


def materialize_sampled_points(
    tar_path: Path,
    sampled_orders: pd.DataFrame,
    target: Path,
    date: str,
    chunksize: int = 500_000,
    buckets: int = 128,
    force: bool = False,
) -> dict[str, Any]:
    """Stream selected complete orders to partitioned fragments without day-sized RAM use."""
    selected = set(sampled_orders.order_id.astype(str))
    target.mkdir(parents=True, exist_ok=True)
    success = target / "_SUCCESS.json"
    if success.exists() and not force:
        return json.loads(success.read_text(encoding="utf-8"))
    rows = 0
    written_orders: set[str] = set()
    partition_ids: set[int] = set()
    fragment_number = 0
    for chunk in iter_daily_chunks(tar_path, chunksize):
        chunk = chunk.loc[chunk.order_id.astype(str).isin(selected)].copy()
        if chunk.empty:
            continue
        chunk["date"] = date
        chunk["bucket"] = chunk.order_id.astype(str).map(lambda value: stable_hash(date, value, seed=0) % int(buckets))
        rows += len(chunk)
        for bucket, group in chunk.groupby("bucket"):
            bucket = int(bucket)
            partition = target / f"part={bucket:03d}"
            partition.mkdir(parents=True, exist_ok=True)
            frame = group.drop(columns="bucket").sort_values(["order_id", "timestamp"], kind="stable")
            frame.to_parquet(partition / f"fragment={fragment_number:06d}.parquet", index=False, compression="zstd")
            written_orders.update(frame.order_id.astype(str))
            partition_ids.add(bucket)
            fragment_number += 1
    result = {
        "date": date,
        "point_rows": rows,
        "sampled_orders": len(selected),
        "materialized_orders": len(written_orders),
        "partitions": len(partition_ids),
        "bucket_count": int(buckets),
    }
    temporary = success.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(success)
    return result


def build_inventory_and_samples(
    config: Stage0Config,
    repo: Path,
    dates: list[str] | None = None,
    orders_per_day: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    archive = config.path("archive", repo)
    seven_zip = config.path("seven_zip", repo)
    output = config.path("output", repo) / "manifests"
    work = config.path("work", repo)
    output.mkdir(parents=True, exist_ok=True)
    members = list_archive_members(archive, seven_zip)
    configured_dates = [str(value) for split in config.section("split").values() for value in split]
    chosen = dates or configured_dates
    member_lookup = {row["date"]: row for row in members}
    missing = sorted(set(chosen) - set(member_lookup))
    if missing:
        raise RuntimeError(f"archive missing configured dates: {missing}")
    inventory = {
        **base_manifest(repo, config.digest, [archive]), "status": "PASS",
        "members": members, "configured_dates": configured_dates,
    }
    write_manifest(output / "archive_inventory.json", inventory)
    sampling_rows: list[pd.DataFrame] = []
    count_rows: list[dict[str, Any]] = []
    sample_count = orders_per_day or int(config.section("sampling")["orders_per_day"])
    seed = int(config.section("sampling")["seed"])
    run_id = sampling_run_id(chosen, sample_count, seed)
    run_root = output / "sampling_runs" / run_id
    for date in chosen:
        eligible_path = output / "daily_eligible_orders" / f"day={date}.parquet"
        sampled_path = sampled_orders_path(output, run_id, date)
        if eligible_path.exists() and not force:
            orders = pd.read_parquet(eligible_path)
            if sampled_path.exists():
                sample = pd.read_parquet(sampled_path)
            else:
                sample = stable_sample(orders, date, sample_count, seed)
                sampled_path.parent.mkdir(parents=True, exist_ok=True)
                sample.to_parquet(sampled_path, index=False, compression="zstd")
            if len(sample) != min(sample_count, int(orders.eligible.sum())):
                sample = stable_sample(orders, date, sample_count, seed)
                sample.to_parquet(sampled_path, index=False, compression="zstd")
        else:
            tar_path = extract_daily_archive(archive, member_lookup[date]["source_member"], seven_zip, work, force=False)
            orders = scan_daily_orders(tar_path, date, member_lookup[date]["source_member"])
            sample = stable_sample(orders, date, sample_count, seed)
            eligible_path.parent.mkdir(parents=True, exist_ok=True)
            sampled_path.parent.mkdir(parents=True, exist_ok=True)
            orders.to_parquet(eligible_path, index=False, compression="zstd")
            sample.to_parquet(sampled_path, index=False, compression="zstd")
        count_rows.append({
            "date": date, "raw_orders": len(orders), "eligible_orders": int(orders.eligible.sum()),
            "ineligible_orders": int((~orders.eligible).sum()), "sampled_orders": len(sample),
        })
        sampling_rows.append(sample[[
            "date", "order_id", "driver_id", "source_member", "sampling_probability",
            "sampling_weight", "sampling_seed", "sample_hash",
        ]])
    run_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(count_rows).to_csv(run_root / "daily_order_counts.csv", index=False)
    pd.concat(sampling_rows, ignore_index=True).to_parquet(run_root / "sampling_manifest.parquet", index=False, compression="zstd")
    result = {
        "status": "PASS", "dates_scanned": len(chosen), "orders_per_day": sample_count,
        "sampling_seed": seed, "runtime_sec": time.perf_counter() - started,
        "sampling_run_id": run_id,
        "sampling_manifest_path": str(run_root / "sampling_manifest.parquet"),
        "counts": count_rows,
    }
    write_manifest(run_root / "sampling_run_manifest.json", result)
    if list(map(str, chosen)) == configured_dates and sample_count == int(config.section("sampling")["orders_per_day"]):
        pd.DataFrame(count_rows).to_csv(output / "daily_order_counts.csv", index=False)
        write_manifest(output / "sampling_run_manifest.json", result)
    return result
