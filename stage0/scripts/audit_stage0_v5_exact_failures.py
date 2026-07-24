#!/usr/bin/env python
"""Rerun only approximate-mode failures with exact Pareto path search."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from stage0.v5.archive import sampling_run_id  # noqa: E402
from stage0.v5.config import Stage0Config, config_hash  # noqa: E402
from stage0.v5.gates import sample_order_sha256  # noqa: E402
from stage0.v5.matching import CandidateIndex, TransitionEngine, match_order  # noqa: E402
from stage0.v5.routing import CompactMovementRouter  # noqa: E402


LOGGER = logging.getLogger("stage0.v5.exact_audit")


def _read_product(root: Path, product: str) -> pd.DataFrame:
    files = sorted((root / product).rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {product} parquet files under {root}")
    return pd.concat(
        [pd.read_parquet(path) for path in files],
        ignore_index=True,
        sort=False,
    )


def run_exact_audit(
    config: Stage0Config,
    dates: list[str],
    orders_per_day: int,
    buckets: int,
    maximum_orders: int | None,
    expected_sample_sha256: str | None,
) -> dict[str, Any]:
    output = config.path("output", REPO)
    work = config.path("work", REPO)
    order_base = _read_product(output, "order_base")
    failed = order_base.loc[
        order_base["date"].astype(str).isin(dates)
        & order_base["matching_mode"].astype(str).eq(
            "failed_no_continuous_route"
        )
    ].copy()
    failed = failed.sort_values(["date", "order_id"], kind="stable")
    total_approximate_failures = int(len(failed))
    if maximum_orders is not None:
        failed = failed.head(int(maximum_orders))
    failed_ids = set(failed.order_id.astype(str))
    if not failed_ids:
        raise RuntimeError("no failed_no_continuous_route orders to audit")

    edges = gpd.read_parquet(output / "network" / "canonical_edges.parquet")
    movements = pd.read_parquet(output / "network" / "movement_graph.parquet")
    candidate_config = config.section("candidate")
    hmm_config = config.section("hmm")
    exact_network_config = {
        **config.section("network"),
        **hmm_config,
        "pareto_search_mode": "exact",
        "pareto_epsilon_m": 0.0,
        "pareto_max_labels_per_state": 0,
    }
    candidate_index = CandidateIndex(
        edges,
        candidate_config,
        str(work / "candidate_index" / config_hash(candidate_config)),
        config.section("network")["metric_crs"],
        hmm_config,
    )
    router = CompactMovementRouter(edges, movements, exact_network_config)
    engine = TransitionEngine(
        edges, movements, pd.DataFrame(), hmm_config, router
    )

    sample_run = sampling_run_id(
        dates,
        orders_per_day,
        int(config.section("sampling")["seed"]),
    )
    sampling_manifest_path = (
        output
        / "manifests"
        / "sampling_runs"
        / sample_run
        / "sampling_manifest.parquet"
    )
    if not sampling_manifest_path.exists():
        raise FileNotFoundError(
            f"sampling manifest missing: {sampling_manifest_path}"
        )
    sampling_manifest = pd.read_parquet(sampling_manifest_path)
    sampling_manifest = sampling_manifest.loc[
        sampling_manifest["date"].astype(str).isin(dates)
    ].copy()
    observed_sample_sha256 = sample_order_sha256(sampling_manifest)
    if (
        expected_sample_sha256 is not None
        and observed_sample_sha256 != expected_sample_sha256
    ):
        raise RuntimeError(
            "sample SHA-256 mismatch: "
            f"expected={expected_sample_sha256} "
            f"observed={observed_sample_sha256}"
        )
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for date in dates:
        date_ids = {
            order_id
            for order_id in failed_ids
            if str(
                failed.loc[
                    failed.order_id.astype(str).eq(order_id), "date"
                ].iloc[0]
            )
            == date
        }
        if not date_ids:
            continue
        day_root = work / "sampled_points" / sample_run / f"day={date}"
        success_path = day_root / "_SUCCESS.json"
        if not success_path.exists():
            raise FileNotFoundError(
                f"sample materialization manifest missing: {success_path}"
            )
        materialization = json.loads(
            success_path.read_text(encoding="utf-8")
        )
        if int(materialization.get("bucket_count", -1)) != int(buckets):
            raise RuntimeError(
                f"bucket count mismatch for {date}: "
                f"requested={buckets} "
                f"materialized={materialization.get('bucket_count')}"
            )
        fragments = sorted(day_root.rglob("*.parquet"))
        points = pd.concat(
            [pd.read_parquet(path) for path in fragments],
            ignore_index=True,
            sort=False,
        )
        points = points.loc[points.order_id.astype(str).isin(date_ids)]
        for order_id, group in points.groupby("order_id", sort=False):
            order_started = time.perf_counter()
            try:
                _, summary = match_order(
                    group,
                    edges,
                    candidate_index,
                    engine,
                    candidate_config,
                    hmm_config,
                )
                error = ""
            except Exception as exc:  # audit must account for every failure
                LOGGER.exception("exact audit failed order=%s", order_id)
                summary = {"matching_mode": "processing_exception"}
                error = f"{type(exc).__name__}: {exc}"
            previous = failed.loc[
                failed.order_id.astype(str).eq(str(order_id))
            ].iloc[0]
            rows.append({
                "date": date,
                "order_id": str(order_id),
                "approximate_mode": str(previous.matching_mode),
                "exact_mode": str(summary["matching_mode"]),
                "exact_recovered": str(summary["matching_mode"])
                not in {"rejected", "failed_no_continuous_route"},
                "exact_runtime_sec": time.perf_counter() - order_started,
                "exact_path_search_calls": int(
                    summary.get("exact_path_search_calls", 0)
                ),
                "exact_expanded_nodes": int(
                    summary.get("dijkstra_expanded_nodes", 0)
                ),
                "full_hmm_attempted": bool(
                    summary.get("full_hmm_attempted", False)
                ),
                "fallback_reason": str(summary.get("fallback_reason", "")),
                "error": error,
            })
            LOGGER.info(
                "exact audit %d/%d order=%s mode=%s recovered=%s elapsed_s=%.1f",
                len(rows),
                len(failed_ids),
                order_id,
                summary["matching_mode"],
                rows[-1]["exact_recovered"],
                time.perf_counter() - started,
            )
    audit = pd.DataFrame(rows).sort_values(
        ["date", "order_id"], kind="stable"
    )
    report_root = output / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    is_complete = len(failed_ids) == total_approximate_failures
    suffix = "" if is_complete else f"_partial_n={len(failed_ids)}"
    csv_path = (
        report_root / f"fixed600_exact_failed_order_audit{suffix}.csv"
    )
    json_path = (
        report_root / f"fixed600_exact_failed_order_audit{suffix}.json"
    )
    audit.to_csv(csv_path, index=False)
    result = {
        "status": "COMPLETE" if is_complete else "PARTIAL",
        "pareto_search_mode": "exact",
        "sample_order_sha256": observed_sample_sha256,
        "approximate_failure_orders_total": total_approximate_failures,
        "approximate_failure_orders_selected": len(failed_ids),
        "orders_audited": len(audit),
        "orders_recovered": int(audit.exact_recovered.sum()),
        "recovery_share": float(audit.exact_recovered.mean()),
        "processing_exceptions": int(audit.error.astype(str).ne("").sum()),
        "runtime_sec": time.perf_counter() - started,
        "csv": str(csv_path),
    }
    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "stage0/config/stage0_v5.yaml",
    )
    parser.add_argument("--dates", nargs="+", required=True)
    parser.add_argument("--orders-per-day", type=int, required=True)
    parser.add_argument("--buckets", type=int, default=16)
    parser.add_argument("--max-orders", type=int, default=None)
    parser.add_argument(
        "--expected-sample-sha256",
        default=None,
        help="Abort unless the sampled date/order universe has this SHA-256.",
    )
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args()
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
        force=True,
    )
    result = run_exact_audit(
        Stage0Config.load(args.config),
        [str(value) for value in args.dates],
        args.orders_per_day,
        args.buckets,
        args.max_orders,
        args.expected_sample_sha256,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["processing_exceptions"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
