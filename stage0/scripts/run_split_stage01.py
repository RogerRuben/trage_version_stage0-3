"""Run the compact train/validation/test Stage0/Stage1 experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-config", type=Path, default=Path("split_config.json"))
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--roads", type=Path, required=True)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--poi", type=Path, required=True)
    parser.add_argument("--stage0-output", type=Path, default=Path("stage0/output"))
    parser.add_argument("--stage1-output", type=Path, default=Path("stage1/output/prediction_split"))
    parser.add_argument("--workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--matcher", choices=["hmm_viterbi", "local_topology_fmm"])
    parser.add_argument("--raw-gap-repair-detour-ratio", type=float, default=100.0)
    parser.add_argument("--raw-gap-repair-extra-m", type=float, default=10000.0)
    parser.add_argument("--phase", choices=["stage0", "stage1", "audit", "all"], default="all")
    parser.add_argument(
        "--limit-days", type=int,
        help="Run Stage0 for the first N split dates (defaults to the full split)",
    )
    parser.add_argument("--no-prune", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the execution plan only")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("$ " + subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


def validate_config(config: dict) -> tuple[list[str], dict[str, str]]:
    train = config["train_dates"]
    if len(train) != 7 or len(set(train)) != 7:
        raise ValueError("split_config must contain 7 distinct train dates")
    validation = config["validation_date"]; test = config["test_date"]
    dates = train + [validation, test]
    if len(set(dates)) != 9 or dates != sorted(dates):
        raise ValueError("train/validation/test dates must be distinct and chronological")
    split = {date: "train" for date in train}; split[validation] = "validation"; split[test] = "test"
    return dates, split


def matcher_collections(matcher: str) -> dict[str, str]:
    if matcher == "local_topology_fmm":
        return {
            "matched": "fast_matched_points",
            "route": "fast_route_parts",
            "quality": "fast_quality_parts",
            "comparison": "matcher_comparison_fast",
            "quality_report": "fast_quality_reports",
            "observed_traversal": "fast_observed_link_traversals",
            "observed_movement": "fast_observed_turn_movements",
            "link_traversal": "fast_link_traversals",
            "turn_movement": "fast_turn_movements",
            "poi_behavior": "stage0_order_link_poi_behavior_fast",
            "link_quality_report": "fast_link_level_quality_summary.csv",
            "runner": "run_fast_stage01_day.py",
        }
    return {
        "matched": "hmm_matched_points",
        "route": "hmm_route_parts",
        "quality": "hmm_quality_parts",
        "comparison": "matcher_comparison",
        "quality_report": "hmm_quality_reports",
        "observed_traversal": "hmm_observed_link_traversals",
        "observed_movement": "hmm_observed_turn_movements",
        "link_traversal": "hmm_link_traversals",
        "turn_movement": "hmm_turn_movements",
        "poi_behavior": "stage0_order_link_poi_behavior",
        "link_quality_report": "link_level_quality_summary.csv",
        "runner": "run_stage01_day.py",
    }


def stage01_products_complete(root: Path, date: str, collections: dict[str, str]) -> bool:
    required = [
        root / "order_base" / f"day={date}.parquet",
        root / collections["link_traversal"] / f"day={date}",
        root / collections["turn_movement"] / f"day={date}",
        root / collections["poi_behavior"] / f"day={date}",
        root / collections["comparison"] / f"day={date}" / "order_comparison.parquet",
    ]
    return all(path.exists() for path in required)


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    stage0_scripts = repo / "stage0" / "scripts"; stage1_scripts = repo / "stage1" / "scripts"
    config = json.loads(args.split_config.read_text(encoding="utf-8"))
    dates, split_lookup = validate_config(config)
    matcher = args.matcher or config.get("stage0_matcher", "hmm_viterbi")
    collections = matcher_collections(matcher)
    limit_days = args.limit_days or len(dates)
    if not 1 <= limit_days <= len(dates):
        raise ValueError(f"limit-days must be between 1 and {len(dates)}")
    if args.phase == "all" and limit_days < len(dates):
        raise ValueError("partial gates must use --phase stage0; Stage1 requires every Stage0 split day")
    stage0_dates = dates[:limit_days]
    for label, path in {
        "archive": args.archive, "roads": args.roads, "nodes": args.nodes, "poi": args.poi,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} input not found: {path}")
    stage0 = args.stage0_output.resolve(); stage1 = args.stage1_output.resolve()
    if args.dry_run:
        plan = {
            "experiment_name": config.get("experiment_name"),
            "phase": args.phase,
            "train_dates": config["train_dates"],
            "validation_date": config["validation_date"],
            "test_date": config["test_date"],
            "fit_dates": config.get("stage1_measurement_fit_dates", config["train_dates"]),
            "target_dates": dates,
            "stage0_dates": stage0_dates,
            "stage0_matcher": matcher,
            "stage0_output": str(stage0),
            "stage1_output": str(stage1),
            "compact_pruning": not args.no_prune,
        }
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    stage0.mkdir(parents=True, exist_ok=True); stage1.mkdir(parents=True, exist_ok=True)
    for root in [stage0, stage1]:
        (root / "manifests").mkdir(parents=True, exist_ok=True)
        serialized_config = json.dumps(config, ensure_ascii=False, indent=2)
        (root / "manifests" / "split_config.json").write_text(serialized_config, encoding="utf-8")
        experiment = config.get("experiment_name", "unnamed").replace("/", "_")
        (root / "manifests" / f"split_config.{experiment}.json").write_text(serialized_config, encoding="utf-8")

    if args.phase in ["stage0", "all"]:
        run([
            sys.executable, str(stage0_scripts / "storage_preflight.py"),
            "--output-root", str(stage0), "--days", str(len(dates)), "--retention-mode", "compact",
        ])
        poi_exposure = stage0 / "poi" / "stage0_link_poi_exposure.parquet"
        if not poi_exposure.exists():
            run([
                sys.executable, str(stage0_scripts / "process_poi.py"), "--poi", str(args.poi),
                "--roads", str(args.roads), "--nodes", str(args.nodes),
                "--output-root", str(stage0), "--input-crs", "auto",
            ])
        for date in stage0_dates:
            if not stage01_products_complete(stage0, date, collections):
                run([
                    sys.executable, str(stage0_scripts / "run_monthly_stage0.py"),
                    "--archive", str(args.archive), "--roads", str(args.roads), "--nodes", str(args.nodes),
                    "--output-root", str(stage0), "--start-date", date, "--end-date", date,
                ])
                command = [
                    sys.executable, str(stage0_scripts / collections["runner"]), "--date", date,
                    "--roads", str(args.roads), "--nodes", str(args.nodes), "--poi-exposure", str(poi_exposure),
                    "--output-root", str(stage0), "--workers", str(args.workers),
                ]
                if matcher == "local_topology_fmm":
                    command += [
                        "--raw-gap-repair-detour-ratio", str(args.raw_gap_repair_detour_ratio),
                        "--raw-gap-repair-extra-m", str(args.raw_gap_repair_extra_m),
                    ]
                run(command)
            case_index = stage0 / "case_traces" / f"day={date}" / "case_index.csv"
            if not case_index.exists():
                run([
                    sys.executable, str(stage0_scripts / "export_case_traces.py"),
                    "--output-root", str(stage0), "--date", date,
                    "--comparison-collection", collections["comparison"],
                    "--matched-collection", collections["matched"],
                    "--route-collection", collections["route"],
                    "--traversal-collection", collections["link_traversal"],
                    "--movement-collection", collections["turn_movement"],
                    "--poi-behavior-collection", collections["poi_behavior"],
                    "--matched-suffix", "fast_points" if matcher == "local_topology_fmm" else "hmm_points",
                ])
            sensitivity = stage0 / "reports" / "threshold_sensitivity" / f"day={date}.json"
            if not sensitivity.exists():
                matched_dir = stage0 / collections["matched"] / f"day={date}"
                if not matched_dir.exists():
                    raise FileNotFoundError(
                        f"threshold sensitivity missing after point data were pruned: {sensitivity}"
                    )
                run([
                    sys.executable, str(stage0_scripts / "export_threshold_sensitivity.py"),
                    "--matched-dir", str(matched_dir),
                    "--output-root", str(stage0), "--date", date,
                ])
            run([
                sys.executable, str(stage0_scripts / "link_level_quality_summary.py"),
                "--output-root", str(stage0),
                "--traversal-collection", collections["link_traversal"],
                "--report-name", collections["link_quality_report"],
            ])
            pruned = stage0 / "manifests" / f"day={date}.pruned.json"
            if not args.no_prune and not pruned.exists():
                run([
                    sys.executable, str(stage0_scripts / "prune_day_outputs.py"),
                    "--output-root", str(stage0), "--date", date, "--execute",
                    "--comparison-collection", collections["comparison"],
                    "--link-traversal-collection", collections["link_traversal"],
                    "--turn-movement-collection", collections["turn_movement"],
                    "--poi-behavior-collection", collections["poi_behavior"],
                ])
            run([
                sys.executable, str(stage0_scripts / "check_stage0_readiness.py"),
                "--output-root", str(stage0),
                "--comparison-collection", collections["comparison"],
                "--link-quality-report", collections["link_quality_report"],
                "--matched-point-collection", collections["matched"],
                "--route-collection", collections["route"],
                "--link-traversal-collection", collections["link_traversal"],
                "--turn-movement-collection", collections["turn_movement"],
                "--poi-behavior-collection", collections["poi_behavior"],
            ])

    train_dates = ",".join(config.get("stage1_measurement_fit_dates", config["train_dates"])); all_dates = ",".join(dates)
    if args.phase in ["stage1", "all"]:
        run([
            sys.executable, str(stage1_scripts / "build_stage1_labels.py"),
            "--traversal-root", str(stage0 / collections["link_traversal"]),
            "--movement-root", str(stage0 / collections["turn_movement"]),
            "--poi-exposure", str(stage0 / "poi" / "stage0_link_poi_exposure.parquet"),
            "--roads", str(args.roads), "--order-base-root", str(stage0 / "order_base"),
            "--stage0-output-root", str(stage0), "--output-root", str(stage1),
            "--fit-dates", train_dates, "--target-dates", all_dates,
        ])

    if args.phase in ["audit", "all"]:
        for date in dates:
            run([
                sys.executable, str(stage1_scripts / "audit_stage1_labels.py"),
                "--output-root", str(stage1), "--date", date, "--split", split_lookup[date],
                "--roads", str(args.roads), "--poi-exposure", str(stage0 / "poi" / "stage0_link_poi_exposure.parquet"),
                "--precomputed-sensitivity", str(stage0 / "reports" / "threshold_sensitivity" / f"day={date}.json"),
            ])
        run([
            sys.executable, str(stage1_scripts / "check_stage1_label_coverage.py"),
            "--output-root", str(stage1), "--split-config", str(args.split_config),
        ])
        run([sys.executable, str(stage1_scripts / "summarize_stage1_validity.py"), "--output-root", str(stage1)])
        run([
            sys.executable, str(stage1_scripts / "split_shift_audit.py"),
            "--output-root", str(stage1), "--split-config", str(args.split_config),
        ])
        run([sys.executable, str(stage1_scripts / "summarize_split.py"), "--output-root", str(stage1)])
        run([
            sys.executable, str(stage1_scripts / "test_day_readiness_summary.py"),
            "--stage0-output-root", str(stage0), "--stage1-output-root", str(stage1),
            "--split-config", str(args.split_config),
        ])


if __name__ == "__main__":
    main()
