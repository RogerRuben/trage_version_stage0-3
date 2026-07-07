"""Delete heavy point-level daily outputs after compact products and case traces exist."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


HEAVY_COLLECTIONS = [
    "matched_points", "hmm_matched_points", "hmm_state_sequences", "hmm_route_parts",
    "hmm_quality_parts", "route_parts", "hmm_observed_link_traversals", "hmm_observed_turn_movements",
    "stage0_link_traversals", "stage0_turn_movements",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--execute", action="store_true", help="Actually delete; otherwise dry-run")
    return parser.parse_args()


def main() -> None:
    args = parse_args(); root = args.output_root.resolve()
    required = [
        root / "order_base" / f"day={args.date}.parquet",
        root / "hmm_link_traversals" / f"day={args.date}",
        root / "hmm_turn_movements" / f"day={args.date}",
        root / "stage0_order_link_poi_behavior" / f"day={args.date}",
        root / "matcher_comparison" / f"day={args.date}" / "order_comparison.parquet",
        root / "case_traces" / f"day={args.date}" / "case_index.csv",
        root / "reports" / "threshold_sensitivity" / f"day={args.date}.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("refusing to prune; compact prerequisites missing:\n" + "\n".join(missing))
    targets = []
    for collection in HEAVY_COLLECTIONS:
        target = (root / collection / f"day={args.date}").resolve()
        if target.exists():
            if root not in target.parents:
                raise ValueError(f"unsafe prune target outside output root: {target}")
            targets.append(target)
    files = [path for target in targets for path in target.rglob("*") if path.is_file()]
    logical_bytes = sum(path.stat().st_size for path in files)
    physical_bytes = sum(path.stat().st_size for path in files if path.stat().st_nlink <= 1)
    retained_hardlink_bytes = logical_bytes - physical_bytes
    print(json.dumps({
        "date": args.date, "execute": args.execute, "targets": [str(x) for x in targets],
        "logical_bytes_removed": logical_bytes,
        "estimated_physical_bytes_reclaimed": physical_bytes,
        "bytes_retained_by_other_hardlinks": retained_hardlink_bytes,
    }, indent=2))
    if not args.execute:
        return
    for target in targets:
        shutil.rmtree(target)
    manifest = root / "manifests" / f"day={args.date}.pruned.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "date": args.date, "complete": True,
        "logical_bytes_removed": logical_bytes,
        "estimated_physical_bytes_reclaimed": physical_bytes,
        "bytes_retained_by_other_hardlinks": retained_hardlink_bytes,
        "deleted_collections": [path.parent.name for path in targets],
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
