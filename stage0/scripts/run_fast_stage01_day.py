"""Run local-topology/FMM matcher and link products for one standardized day."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--roads", type=Path, required=True)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--poi-exposure", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("stage0/output"))
    parser.add_argument("--workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--limit-parts", type=int)
    parser.add_argument("--raw-gap-repair-detour-ratio", type=float, default=6.0)
    parser.add_argument("--raw-gap-repair-extra-m", type=float, default=800.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write("\n$ " + subprocess.list2cmdline(command) + "\n")
        stream.flush()
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)


def products_complete(output: Path, date: str) -> bool:
    required = [
        output / "fast_matched_points" / f"day={date}",
        output / "fast_route_parts" / f"day={date}",
        output / "fast_quality_parts" / f"day={date}",
        output / "fast_link_traversals" / f"day={date}",
        output / "fast_turn_movements" / f"day={date}",
        output / "stage0_order_link_poi_behavior_fast" / f"day={date}",
        output / "matcher_comparison_fast" / f"day={date}" / "order_comparison.parquet",
    ]
    return all(path.exists() for path in required)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    scripts = root / "stage0" / "scripts"
    output = args.output_root.resolve()
    geometric = output / "matched_points" / f"day={args.date}"
    if not geometric.exists():
        raise FileNotFoundError(f"standardized geometric day missing: {geometric}")
    if products_complete(output, args.date) and not args.force:
        print(json.dumps({"date": args.date, "complete": True, "skipped": True}, indent=2))
        return

    logs = output / "logs"
    processes: list[tuple[subprocess.Popen, object]] = []
    for worker in range(args.workers):
        log_handle = (logs / f"day={args.date}.fast.worker={worker}.log").open("a", encoding="utf-8")
        command = [
            sys.executable, str(scripts / "fast_topology_fmm_matcher.py"),
            "--matched-dir", str(geometric), "--roads", str(args.roads), "--nodes", str(args.nodes),
            "--output-root", str(output), "--date", args.date,
            "--worker-count", str(args.workers), "--worker-index", str(worker),
            "--raw-gap-repair-detour-ratio", str(args.raw_gap_repair_detour_ratio),
            "--raw-gap-repair-extra-m", str(args.raw_gap_repair_extra_m),
        ]
        if args.limit_parts is not None:
            command += ["--limit-parts", str(args.limit_parts)]
        if args.force:
            command.append("--force")
        log_handle.write("\n$ " + subprocess.list2cmdline(command) + "\n")
        log_handle.flush()
        processes.append((subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT), log_handle))
    failures = []
    for process, handle in processes:
        code = process.wait()
        handle.close()
        if code:
            failures.append(code)
    if failures:
        raise subprocess.CalledProcessError(failures[0], "fast matcher worker")

    run([
        sys.executable, str(scripts / "compare_matchers.py"),
        "--order-base", str(output / "order_base" / f"day={args.date}.parquet"),
        "--hmm-quality-dir", str(output / "fast_quality_parts" / f"day={args.date}"),
        "--output-root", str(output), "--date", args.date,
        "--geometric-matched-dir", str(output / "matched_points" / f"day={args.date}"),
        "--hmm-matched-dir", str(output / "fast_matched_points" / f"day={args.date}"),
        "--roads", str(args.roads),
        "--comparison-collection", "matcher_comparison_fast",
        "--matcher-label", "Local topology + FMM",
        "--quality-report-collection", "fast_quality_reports",
        "--manifest-glob", "day={date}.fast.worker=*.json",
    ], logs / f"day={args.date}.fast_comparison.log")
    link_processes: list[tuple[subprocess.Popen, object]] = []
    for worker in range(args.workers):
        log_handle = (logs / f"day={args.date}.fast_link_products.worker={worker}.log").open("a", encoding="utf-8")
        command = [
            sys.executable, str(scripts / "build_link_products.py"),
            "--matched-dir", str(output / "fast_matched_points" / f"day={args.date}"),
            "--roads", str(args.roads), "--output-root", str(output), "--date", args.date,
            "--matcher-version", "local_topology_fmm",
            "--traversal-collection", "fast_observed_link_traversals",
            "--movement-collection", "fast_observed_turn_movements",
            "--worker-count", str(args.workers), "--worker-index", str(worker),
            *(["--force"] if args.force else []),
        ]
        log_handle.write("\n$ " + subprocess.list2cmdline(command) + "\n"); log_handle.flush()
        link_processes.append((subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT), log_handle))
    link_failures = []
    for process, handle in link_processes:
        code = process.wait(); handle.close()
        if code: link_failures.append(code)
    if link_failures:
        raise subprocess.CalledProcessError(link_failures[0], "link-product worker")
    run([
        sys.executable, str(scripts / "augment_hmm_traversals.py"),
        "--observed-traversal-dir", str(output / "fast_observed_link_traversals" / f"day={args.date}"),
        "--route-dir", str(output / "fast_route_parts" / f"day={args.date}"),
        "--roads", str(args.roads),
        "--traversal-output-dir", str(output / "fast_link_traversals" / f"day={args.date}"),
        "--movement-output-dir", str(output / "fast_turn_movements" / f"day={args.date}"),
        *(["--force"] if args.force else []),
    ], logs / f"day={args.date}.fast_augment.log")
    run([
        sys.executable, str(scripts / "build_order_link_poi_behavior.py"),
        "--traversal-dir", str(output / "fast_link_traversals" / f"day={args.date}"),
        "--poi-exposure", str(args.poi_exposure), "--output-root", str(output), "--date", args.date,
        "--output-collection", "stage0_order_link_poi_behavior_fast",
        *(["--force"] if args.force else []),
    ], logs / f"day={args.date}.fast_poi_behavior.log")
    manifest = {
        "date": args.date,
        "complete": True,
        "matcher_version": "local_topology_fmm",
        "workers": args.workers,
        "stage1_preconditions": {
            "fast_matching": True,
            "link_traversals": True,
            "turn_movements": True,
            "link_poi_exposure": True,
            "order_link_poi_behavior": True,
        },
    }
    manifest_path = output / "manifests" / f"day={args.date}.fast_stage01_preconditions.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
