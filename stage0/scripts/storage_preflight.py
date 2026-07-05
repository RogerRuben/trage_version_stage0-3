"""Estimate retained monthly Stage0/Stage1 storage before launching all days."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("stage0_output"))
    parser.add_argument("--pilot-date", default="20161001")
    parser.add_argument("--days", type=int, default=31)
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
    traversals = collection_stats(root / "stage0_link_traversals" / f"day={args.pilot_date}")
    drive = Path(root.anchor)
    import shutil
    usage = shutil.disk_usage(drive)
    geo_month = geometric["bytes"] * args.days if geometric["bytes"] else 0
    hmm_day = hmm["bytes"] * 128 / hmm["files"] if hmm["files"] else 0
    hmm_month = int(hmm_day * args.days) if hmm_day else (
        int(geometric["rows"] * 120 * args.days) if geometric["rows"] else 0
    )
    traversal_day = (
        traversals["bytes"] * 128 / traversals["files"] if traversals["files"] else geometric["bytes"] * 0.15
    )
    ancillary_month = int(traversal_day * 2.5 * args.days)
    required = geo_month + hmm_month + ancillary_month
    report = {
        "pilot_date": args.pilot_date, "days": args.days, "free_bytes": usage.free,
        "geometric_pilot": geometric, "hmm_pilot": hmm, "traversal_pilot": traversals,
        "estimated_geometric_month_bytes": geo_month, "estimated_hmm_month_bytes": hmm_month,
        "estimated_ancillary_month_bytes": ancillary_month, "estimated_total_bytes": required,
        "estimated_headroom_bytes": usage.free - required,
        "sufficient": usage.free >= required * 1.1,
    }
    manifest = root / "manifests"; manifest.mkdir(parents=True, exist_ok=True)
    (manifest / "storage_preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
