#!/usr/bin/env python
"""Unified gated controller for Hybrid Selective-HMM Stage 0 v5."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from stage0.v5.archive import build_inventory_and_samples  # noqa: E402
from stage0.v5.config import Stage0Config, config_hash  # noqa: E402
from stage0.v5.gates import freeze, gate1_readiness, preflight, require_test_freeze, run_gate0  # noqa: E402
from stage0.v5.manual_review import export_review_pack  # noqa: E402
from stage0.v5.network import build_network  # noqa: E402
from stage0.v5.pipeline import prepare_day_points, run_dates  # noqa: E402
from stage0.v5.poi import build_poi  # noqa: E402
from stage0.v5.retention import prune_point_work  # noqa: E402


GATE1_DATES = ["20161010", "20161014", "20161016"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "stage0/config/stage0_v5.yaml")
    parser.add_argument("--output-root", type=Path, default=None, help="Explicit isolated output root for benchmarks.")
    parser.add_argument("--work-root", type=Path, default=None, help="Explicit isolated cache root for benchmarks.")
    parser.add_argument("--phase", required=True, choices=["precheck", "network", "poi", "inventory", "materialize", "gate0", "match", "gate1", "manual-review", "freeze", "prune"])
    parser.add_argument("--dates", nargs="*", default=None)
    parser.add_argument("--split", choices=["train", "validation", "test"])
    parser.add_argument("--orders-per-day", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None, help="Reserved for bounded partition concurrency; current safe executor is sequential.")
    parser.add_argument("--buckets", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retain-points", action="store_true")
    parser.add_argument("--limit-orders", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--bucket-shard-index", type=int, default=0)
    parser.add_argument("--bucket-shard-count", type=int, default=1)
    parser.add_argument("--bucket-ids", nargs="*", type=int, default=None)
    parser.add_argument("--review-pack", choices=["development", "test", "both"], default="development")
    parser.add_argument("--execute", action="store_true", help="Required for point-work pruning.")
    return parser.parse_args()


def resolve_dates(args: argparse.Namespace, config: Stage0Config) -> list[str]:
    if args.dates:
        return [str(value) for value in args.dates]
    if args.split:
        return [str(value) for value in config.section("split")[args.split]]
    return [str(value) for values in config.section("split").values() for value in values]


def main() -> int:
    args = parse_args()
    config = Stage0Config.load(args.config)
    if args.output_root is not None or args.work_root is not None:
        values = {**config.values, "paths": {**config.values["paths"]}}
        if args.output_root is not None:
            values["paths"]["output"] = str(args.output_root.resolve())
        if args.work_root is not None:
            values["paths"]["work"] = str(args.work_root.resolve())
        config = Stage0Config(config.source, values, config_hash(values))
    logging.basicConfig(level=getattr(logging, config.section("runtime")["log_level"]), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    dates = resolve_dates(args, config)
    buckets = args.buckets or int(config.section("runtime")["buckets"])
    workers = args.workers or int(config.section("runtime")["workers"])
    orders = args.limit_orders or args.orders_per_day or int(config.section("sampling")["orders_per_day"])
    if args.dry_run:
        print(json.dumps({"phase": args.phase, "dates": dates, "orders_per_day": orders, "buckets": buckets, "workers_requested": args.workers, "effective_workers": 1, "config_hash": config.digest}, indent=2))
        return 0
    if args.phase == "precheck":
        result = preflight(config, REPO)
    elif args.phase == "network":
        result = build_network(config, REPO, force=args.force)
    elif args.phase == "poi":
        result = build_poi(config, REPO, force=args.force)
    elif args.phase == "inventory":
        result = build_inventory_and_samples(config, REPO, dates=dates, orders_per_day=orders, force=args.force)
    elif args.phase == "materialize":
        require_test_freeze(config, REPO, dates)
        build_inventory_and_samples(config, REPO, dates=dates, orders_per_day=orders, force=False)
        result = {
            "status": "PASS",
            "dates": [prepare_day_points(config, REPO, date, buckets, orders, dates, force=args.force) for date in dates],
        }
    elif args.phase == "gate0":
        result = run_gate0(config, REPO, dates=dates, force=args.force)
    elif args.phase == "match":
        require_test_freeze(config, REPO, dates)
        if args.bucket_shard_count == 1:
            build_inventory_and_samples(config, REPO, dates=dates, orders_per_day=orders, force=False)
        result = run_dates(
            config, REPO, dates, buckets=buckets, resume=args.resume,
            retain_points=args.retain_points, force=args.force, workers=workers,
            bucket_shard_index=args.bucket_shard_index,
            bucket_shard_count=args.bucket_shard_count,
            bucket_ids=set(args.bucket_ids) if args.bucket_ids else None,
            orders_per_day=orders,
        )
    elif args.phase == "gate1":
        dates = GATE1_DATES
        if orders != 2000:
            raise ValueError("Gate 1 is frozen at exactly 2,000 complete orders per day")
        build_inventory_and_samples(config, REPO, dates=dates, orders_per_day=2000, force=False)
        summary = run_dates(
            config, REPO, dates, buckets=buckets, resume=args.resume,
            retain_points=args.retain_points, force=args.force, workers=workers,
            orders_per_day=2000,
        )
        result = gate1_readiness(config, REPO, summary)
    elif args.phase == "manual-review":
        result = {"status": "PASS"}
        if args.review_pack in {"development", "both"}:
            result["development"] = export_review_pack(config, REPO, "development", 300)
        if args.review_pack in {"test", "both"}:
            require_test_freeze(config, REPO, config.section("split")["test"])
            result["blind_test"] = export_review_pack(config, REPO, "test", 200)
    elif args.phase == "freeze":
        result = freeze(config, REPO)
    elif args.phase == "prune":
        result = prune_point_work(config, REPO, execute=args.execute)
    else:
        raise AssertionError(args.phase)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("status", "PASS") in {"PASS", "FROZEN", "SHARD_COMPLETE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
