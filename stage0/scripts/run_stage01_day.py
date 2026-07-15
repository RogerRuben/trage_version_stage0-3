"""Run resume-safe HMM matching and link products for one standardized day."""

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
    parser.add_argument("--candidates", type=int, default=3)
    return parser.parse_args()


def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write("\n$ " + subprocess.list2cmdline(command) + "\n"); stream.flush()
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    scripts = root / "stage0" / "scripts"
    output = args.output_root.resolve()
    geometric = output / "matched_points" / f"day={args.date}"
    if not geometric.exists():
        raise FileNotFoundError(f"standardized geometric day missing: {geometric}")
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[subprocess.Popen, object]] = []
    for worker in range(args.workers):
        log_handle = (logs / f"day={args.date}.hmm.worker={worker}.log").open("a", encoding="utf-8")
        command = [
            sys.executable, str(scripts / "hmm_viterbi_matcher.py"),
            "--matched-dir", str(geometric), "--roads", str(args.roads), "--nodes", str(args.nodes),
            "--output-root", str(output), "--date", args.date, "--candidates", str(args.candidates),
            "--worker-count", str(args.workers), "--worker-index", str(worker),
        ]
        log_handle.write("\n$ " + subprocess.list2cmdline(command) + "\n"); log_handle.flush()
        processes.append((subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT), log_handle))
    failures = []
    for process, handle in processes:
        code = process.wait(); handle.close()
        if code:
            failures.append(code)
    if failures:
        raise subprocess.CalledProcessError(failures[0], "HMM worker")

    run([
        sys.executable, str(scripts / "compare_matchers.py"),
        "--order-base", str(output / "order_base" / f"day={args.date}.parquet"),
        "--hmm-quality-dir", str(output / "hmm_quality_parts" / f"day={args.date}"),
        "--output-root", str(output), "--date", args.date,
        "--geometric-matched-dir", str(output / "matched_points" / f"day={args.date}"),
        "--hmm-matched-dir", str(output / "hmm_matched_points" / f"day={args.date}"),
        "--roads", str(args.roads),
    ], logs / f"day={args.date}.comparison.log")
    run([
        sys.executable, str(scripts / "reconstruct_hmm_routes.py"),
        "--matched-dir", str(output / "hmm_matched_points" / f"day={args.date}"),
        "--roads", str(args.roads), "--nodes", str(args.nodes),
        "--output-dir", str(output / "hmm_route_parts" / f"day={args.date}"),
        "--force",
    ], logs / f"day={args.date}.hmm_routes.log")
    run([
        sys.executable, str(scripts / "build_link_products.py"),
        "--matched-dir", str(output / "hmm_matched_points" / f"day={args.date}"),
        "--roads", str(args.roads), "--output-root", str(output), "--date", args.date,
        "--matcher-version", "hmm_viterbi", "--traversal-collection", "hmm_observed_link_traversals",
        "--movement-collection", "hmm_observed_turn_movements",
    ], logs / f"day={args.date}.hmm_link_products.log")
    run([
        sys.executable, str(scripts / "augment_hmm_traversals.py"),
        "--observed-traversal-dir", str(output / "hmm_observed_link_traversals" / f"day={args.date}"),
        "--route-dir", str(output / "hmm_route_parts" / f"day={args.date}"),
        "--roads", str(args.roads),
        "--traversal-output-dir", str(output / "hmm_link_traversals" / f"day={args.date}"),
        "--movement-output-dir", str(output / "hmm_turn_movements" / f"day={args.date}"),
    ], logs / f"day={args.date}.hmm_augment.log")
    run([
        sys.executable, str(scripts / "build_order_link_poi_behavior.py"),
        "--traversal-dir", str(output / "hmm_link_traversals" / f"day={args.date}"),
        "--poi-exposure", str(args.poi_exposure), "--output-root", str(output), "--date", args.date,
        "--output-collection", "stage0_order_link_poi_behavior",
    ], logs / f"day={args.date}.poi_behavior.log")
    manifest = {
        "date": args.date, "complete": True, "hmm_workers": args.workers,
        "candidate_count": args.candidates, "stage1_preconditions": {
            "hmm_matching": True, "link_traversals": True, "turn_movements": True,
            "link_poi_exposure": True, "order_link_poi_behavior": True,
        },
    }
    manifest_path = output / "manifests" / f"day={args.date}.stage01_preconditions.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
