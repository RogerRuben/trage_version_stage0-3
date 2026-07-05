"""Top-level resume-safe monthly Stage0/HMM/Stage1 orchestrator."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--roads", type=Path, required=True)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--poi", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start-date", default="20161001")
    parser.add_argument("--end-date", default="20161031")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--minimum-free-gb", type=float, default=200.0)
    parser.add_argument("--allow-low-space", action="store_true")
    parser.add_argument("--skip-labels", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("$ " + subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    scripts = root / "stage0" / "scripts"
    output = args.output_root.resolve(); output.mkdir(parents=True, exist_ok=True)
    requested_days = (pd.Timestamp(args.end_date) - pd.Timestamp(args.start_date)).days + 1
    free_gb = shutil.disk_usage(output.anchor).free / 1024**3
    if requested_days > 1 and free_gb < args.minimum_free_gb and not args.allow_low_space:
        raise OSError(
            f"monthly retained dual-matcher run needs at least {args.minimum_free_gb:.0f} GB free; "
            f"only {free_gb:.1f} GB available. Use a larger --output-root."
        )
    if not (output / "stage0_link_poi_exposure.parquet").exists():
        run([
            sys.executable, str(scripts / "process_poi.py"), "--poi", str(args.poi),
            "--roads", str(args.roads), "--nodes", str(args.nodes), "--output-root", str(output), "--input-crs", "auto",
        ])
    dates = pd.date_range(args.start_date, args.end_date, freq="D").strftime("%Y%m%d").tolist()
    for date in dates:
        run([
            sys.executable, str(scripts / "run_monthly_stage0.py"), "--archive", str(args.archive),
            "--roads", str(args.roads), "--nodes", str(args.nodes), "--output-root", str(output),
            "--start-date", date, "--end-date", date,
        ])
        run([
            sys.executable, str(scripts / "run_stage01_day.py"), "--date", date,
            "--roads", str(args.roads), "--nodes", str(args.nodes),
            "--poi-exposure", str(output / "stage0_link_poi_exposure.parquet"),
            "--output-root", str(output), "--workers", str(args.workers),
        ])
    if args.skip_labels:
        return
    run([
        sys.executable, str(scripts / "build_stage1_labels.py"),
        "--traversal-root", str(output / "hmm_link_traversals"),
        "--movement-root", str(output / "hmm_turn_movements"),
        "--poi-exposure", str(output / "stage0_link_poi_exposure.parquet"),
        "--roads", str(args.roads), "--order-base-root", str(output / "order_base"),
        "--output-root", str(output), "--fit-dates", "all", "--target-dates", "all",
    ])
    for date in dates:
        run([
            sys.executable, str(scripts / "audit_stage1_labels.py"),
            "--output-root", str(output), "--date", date, "--roads", str(args.roads),
            "--matched-dir", str(output / "hmm_matched_points" / f"day={date}"),
            "--poi-exposure", str(output / "stage0_link_poi_exposure.parquet"),
        ])
    run([sys.executable, str(scripts / "summarize_stage1_validity.py"), "--output-root", str(output)])


if __name__ == "__main__":
    main()
