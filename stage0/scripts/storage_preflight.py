"""Estimate retained monthly Stage0/Stage1 storage before launching all days."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("stage0/output"))
    parser.add_argument("--pilot-date", default="20161001")
    parser.add_argument("--days", type=int, default=9)
    parser.add_argument("--retention-mode", choices=["compact", "full"], default="compact")
    parser.add_argument("--allow-insufficient", action="store_true")
    return parser.parse_args()


def collection_stats(directory: Path) -> dict:
    files = list(directory.glob("*.parquet")) if directory.exists() else []
    rows = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    size = sum(path.stat().st_size for path in files)
    return {"files": len(files), "rows": rows, "bytes": size, "bytes_per_row": size / rows if rows else None}


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    geometric = collection_stats(root / "matched_points" / f"day={args.pilot_date}")
    hmm = collection_stats(root / "hmm_matched_points" / f"day={args.pilot_date}")
    traversals = collection_stats(root / "hmm_link_traversals" / f"day={args.pilot_date}")
    if not traversals["files"]:
        traversals = collection_stats(root / "stage0_link_traversals" / f"day={args.pilot_date}")
    drive = Path(root.anchor)
    import shutil
    usage = shutil.disk_usage(drive)
    # A fresh compact output root has no pilot yet. Never interpret that as
    # zero storage: use conservative Xi'an day-size defaults until a real day
    # can replace them.
    has_pilot = bool(geometric["bytes"] or hmm["bytes"] or traversals["bytes"])
    default_geo_day = 2 * 1024**3
    default_hmm_day = 4 * 1024**3
    default_retained_day = 1 * 1024**3
    geo_day = geometric["bytes"] if geometric["bytes"] else default_geo_day
    geo_month = geo_day * args.days
    hmm_day = hmm["bytes"] * 128 / hmm["files"] if hmm["files"] else 0
    if not hmm_day:
        hmm_day = int(geometric["rows"] * 120) if geometric["rows"] else default_hmm_day
    hmm_month = int(hmm_day * args.days)
    traversal_day = (
        traversals["bytes"] * 128 / traversals["files"] if traversals["files"] else (
            geometric["bytes"] * 0.15 if geometric["bytes"] else default_retained_day / 2.5
        )
    )
    ancillary_month = int(traversal_day * 2.5 * args.days)
    if args.retention_mode == "compact":
        retained_month = ancillary_month
        peak_working_set = int(geo_day + hmm_day + traversal_day * 2.5)
        required = retained_month + peak_working_set
    else:
        retained_month = geo_month + hmm_month + ancillary_month
        peak_working_set = 0
        required = retained_month
    report = {
        "pilot_date": args.pilot_date, "days": args.days, "free_bytes": usage.free,
        "estimate_source": "pilot" if has_pilot else "conservative_defaults",
        "geometric_pilot": geometric, "hmm_pilot": hmm, "traversal_pilot": traversals,
        "estimated_geometric_month_bytes": geo_month, "estimated_hmm_month_bytes": hmm_month,
        "retention_mode": args.retention_mode,
        "estimated_ancillary_month_bytes": ancillary_month,
        "estimated_retained_bytes": retained_month, "estimated_peak_working_set_bytes": peak_working_set,
        "estimated_total_bytes": required,
        "estimated_headroom_bytes": usage.free - required,
        "sufficient": usage.free >= required * 1.1,
    }
    manifest = root / "manifests"; manifest.mkdir(parents=True, exist_ok=True)
    (manifest / "storage_preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["sufficient"] and not args.allow_insufficient:
        raise SystemExit("storage preflight failed: estimated headroom is below the 10% safety margin")


if __name__ == "__main__":
    main()
