"""Chunked full-day variant of attach_stage2_planned_lagged_state.py.

The original utility is fine for sampled route-conditioned experiments but it
concatenates all link-group merge_asof results in memory.  A full 2016-10-23
day has more than three million route-link rows, so this script processes
orders in chunks and writes a parquet dataset directory per day.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from attach_stage2_planned_lagged_state import (  # noqa: E402
    TARGETS,
    add_profile,
    asof_by_group,
    combine_history,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planned-route-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, default=Path("stage2/output/lagged_state_store"))
    parser.add_argument("--strict-target-root", type=Path, default=Path("stage2/output/strict_targets"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dates", required=True)
    parser.add_argument("--history-days", type=int, default=7)
    parser.add_argument("--max-state-age-minutes", type=int, default=60)
    parser.add_argument("--chunk-orders", type=int, default=5000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_order_ids(path: Path) -> np.ndarray:
    return pd.read_parquet(path, columns=["order_id"])["order_id"].astype(str).drop_duplicates().to_numpy()


def _read_order_chunk(path: Path, order_ids: np.ndarray) -> pd.DataFrame:
    dataset = ds.dataset(path, format="parquet")
    table = dataset.to_table(filter=ds.field("order_id").isin(order_ids.tolist()))
    return table.to_pandas()


def _attach_chunk(
    frame: pd.DataFrame,
    *,
    date: str,
    dates: list[str],
    link: pd.DataFrame,
    area: pd.DataFrame,
    network: pd.DataFrame,
    tolerance: pd.Timedelta,
    model_root: Path,
    history_days: int,
) -> pd.DataFrame:
    frame = frame.copy()
    frame["estimated_link_entry_time"] = pd.to_datetime(frame["estimated_link_entry_time"], utc=True)
    local = frame["estimated_link_entry_time"].dt.tz_convert("Asia/Shanghai")
    frame["estimated_time_bin"] = (local.dt.hour * 2 + (local.dt.minute >= 30).astype(int)).astype("int16")
    frame = asof_by_group(frame, link, "planned_link_id", "link_id", "estimated_link_entry_time", "link_state_bin_time", tolerance)
    if "area_grid" in frame:
        frame = asof_by_group(frame, area, "area_grid", "area_grid", "estimated_link_entry_time", "area_state_bin_time", tolerance)
    frame = pd.merge_asof(
        frame.sort_values("estimated_link_entry_time"),
        network.sort_values("network_state_bin_time"),
        left_on="estimated_link_entry_time",
        right_on="network_state_bin_time",
        direction="backward",
        tolerance=tolerance,
    )
    previous = [value for value in dates if value < date][-history_days:]
    for target in TARGETS:
        frame = add_profile(frame, combine_history(model_root, previous, target), target)
    availability_columns = [column for column in frame.columns if "availability_timestamp" in column]
    checks = []
    for column in availability_columns:
        values = pd.to_datetime(frame[column], utc=True, errors="coerce")
        checks.append(values.isna() | values.lt(frame["estimated_link_entry_time"]))
    frame["strict_availability_check"] = np.logical_and.reduce(checks) if checks else True
    return frame.sort_values(["order_id", "planned_link_seq"], kind="mergesort")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    dates = [part.strip() for part in args.dates.split(",") if part.strip()]
    all_context_dates = sorted(set(dates + ["20161017", "20161018", "20161019", "20161020", "20161021", "20161022"]))

    link = pd.read_parquet(args.state_root / "link_state.parquet").rename(columns={
        "bin_time": "link_state_bin_time",
        "feature_timestamp": "link_state_feature_timestamp",
        "availability_timestamp": "link_state_availability_timestamp",
    })
    area = pd.read_parquet(args.state_root / "area_state.parquet").rename(columns={
        "bin_time": "area_state_bin_time",
        "feature_timestamp": "area_state_feature_timestamp",
        "availability_timestamp": "area_state_availability_timestamp",
    })
    network = pd.read_parquet(args.state_root / "network_state.parquet").rename(columns={
        "bin_time": "network_state_bin_time",
        "feature_timestamp": "network_state_feature_timestamp",
        "availability_timestamp": "network_state_availability_timestamp",
    })
    tolerance = pd.Timedelta(minutes=args.max_state_age_minutes)
    model_root = args.strict_target_root / "models" / "daily_stats"
    manifest = {"dates": dates, "days": {}, "strict_availability_rule": True, "chunk_orders": args.chunk_orders}

    for date in dates:
        source_path = args.planned_route_root / f"day={date}.parquet"
        target_dir = args.output_root / f"day={date}.parquet"
        if target_dir.exists():
            if not args.overwrite:
                raise FileExistsError(f"{target_dir} exists; pass --overwrite")
            for child in target_dir.glob("*"):
                child.unlink()
        target_dir.mkdir(parents=True, exist_ok=True)
        order_ids = _read_order_ids(source_path)
        total_rows = 0
        strict_values = []
        coverage_values = []
        for start in range(0, len(order_ids), args.chunk_orders):
            selected = order_ids[start:start + args.chunk_orders]
            chunk = _read_order_chunk(source_path, selected)
            chunk = _attach_chunk(
                chunk,
                date=date,
                dates=all_context_dates,
                link=link,
                area=area,
                network=network,
                tolerance=tolerance,
                model_root=model_root,
                history_days=args.history_days,
            )
            total_rows += int(len(chunk))
            strict_values.append(float(chunk["strict_availability_check"].mean()))
            if "link_recent_traversal_count_15m" in chunk:
                coverage_values.append(float(chunk["link_recent_traversal_count_15m"].notna().mean()))
            chunk.to_parquet(target_dir / f"part-{start // args.chunk_orders:05d}.parquet", index=False, compression="zstd")
            print(f"lagged chunk day={date} chunk={start // args.chunk_orders} orders={len(selected)} rows={len(chunk)}", flush=True)
        manifest["days"][date] = {
            "orders": int(len(order_ids)),
            "rows": total_rows,
            "strict_availability_pass_ratio": float(np.mean(strict_values)) if strict_values else 0.0,
            "link_state_coverage": float(np.mean(coverage_values)) if coverage_values else 0.0,
            "output": str(target_dir),
        }
        print(f"chunked causal planned day={date} {manifest['days'][date]}", flush=True)
    (args.output_root / "planned_causal_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
