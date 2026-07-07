"""Reuse legacy Stage0 day products in the split-aware tree via hard links.

Only products whose schema is compatible with the current pipeline are linked.
Legacy ``hmm_route_parts`` are intentionally excluded because older runs stored
candidate-state rows there; the current route reconstruction must regenerate
them from ``hmm_state_sequences``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


DAY_COLLECTIONS = [
    "matched_points",
    "route_parts",
    "stage0_link_traversals",
    "stage0_turn_movements",
    "hmm_matched_points",
    "hmm_quality_parts",
    "hmm_state_sequences",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--date", required=True)
    return parser.parse_args()


def link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def migrate_tree(source: Path, target: Path, counts: dict[str, int]) -> None:
    if not source.exists():
        return
    for path in sorted(source.rglob("*")):
        if path.is_file():
            mode = link_or_copy(path, target / path.relative_to(source))
            counts[mode] = counts.get(mode, 0) + 1


def main() -> None:
    args = parse_args()
    legacy = args.legacy_root.resolve()
    output = args.output_root.resolve()
    if legacy == output:
        raise ValueError("legacy-root and output-root must differ")
    if not legacy.exists():
        raise FileNotFoundError(legacy)
    counts: dict[str, int] = {}

    for name in DAY_COLLECTIONS:
        migrate_tree(
            legacy / name / f"day={args.date}",
            output / name / f"day={args.date}",
            counts,
        )

    order_base = legacy / "order_base" / f"day={args.date}.parquet"
    if order_base.exists():
        mode = link_or_copy(order_base, output / "order_base" / order_base.name)
        counts[mode] = counts.get(mode, 0) + 1

    for path in sorted((legacy / "logs").glob(f"*{args.date}*")):
        if path.is_file():
            mode = link_or_copy(path, output / "logs" / path.name)
            counts[mode] = counts.get(mode, 0) + 1
    for path in sorted((legacy / "manifests").glob(f"*{args.date}*")):
        if path.is_file():
            mode = link_or_copy(path, output / "manifests" / f"legacy_{path.name}")
            counts[mode] = counts.get(mode, 0) + 1
            # The geometric controller recognizes the canonical standardize
            # manifest, so expose that compatible manifest under its current
            # name as well. Other legacy manifests remain namespaced.
            if path.name == f"day={args.date}.standardize.json":
                mode = link_or_copy(path, output / "manifests" / path.name)
                counts[mode] = counts.get(mode, 0) + 1

    manifest = {
        "date": args.date,
        "legacy_root": str(legacy),
        "output_root": str(output),
        "counts": counts,
        "excluded": ["hmm_route_parts (must be reconstructed with the current schema)"],
    }
    manifest_dir = output / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"day={args.date}.legacy_migration.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
