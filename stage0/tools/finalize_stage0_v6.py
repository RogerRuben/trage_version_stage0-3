"""Aggregate Stage 1 production coverage and write the Stage 0 freeze records."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from stage0.v6.config import Stage0V6Config, load_config
from stage0.v6.stage1_production import _sha256_directory, _sha256_file


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _quantiles(counter: Counter[Any]) -> dict[str, float]:
    values: list[int] = sorted(counter.values())
    if not values:
        return {"min": 0, "p25": 0, "p50": 0, "p75": 0, "p90": 0, "max": 0}
    series = pd.Series(values, dtype="float64")
    return {
        "min": float(series.min()),
        "p25": float(series.quantile(0.25)),
        "p50": float(series.quantile(0.50)),
        "p75": float(series.quantile(0.75)),
        "p90": float(series.quantile(0.90)),
        "max": float(series.max()),
    }


def _daily_target(config: Stage0V6Config, split: str, date: str) -> int:
    production = config.section("production")
    if split == "train":
        return int(production["train_daily_target"])
    if split == "validation":
        return int(production["validation_daily_target"])
    if split == "test" and date == str(production["test_date"]):
        return int(production["test_target"])
    return 0


def aggregate(config: Stage0V6Config) -> dict[str, Any]:
    root = config.path("stage1_input")
    bucket_manifests = sorted(root.glob("split=*/date=*/bucket=*/manifest.json"))
    expected_dates = {
        ("train", str(date))
        for date in config.section("production")["train_dates"]
    }
    expected_dates |= {
        ("validation", str(date))
        for date in config.section("production")["validation_dates"]
    }
    expected_dates.add(("test", str(config.section("production")["test_date"])))

    daily: dict[tuple[str, str], dict[str, Any]] = {
        key: {
            "split": key[0],
            "date": key[1],
            "target_count": _daily_target(config, *key),
            "candidate_count": 0,
            "candidate_bucket_capacity": 0,
            "pre_match_eligible_count": 0,
            "matched_count": 0,
            "accepted_count": 0,
            "rejected_count": 0,
            "processing_exception_count": 0,
            "runtime_s": 0.0,
            "peak_rss_mb": 0.0,
            "bucket_count": 0,
        }
        for key in expected_dates
    }
    rejection_reasons: Counter[str] = Counter()
    code_identities: set[str] = set()
    config_hashes: set[str] = set()
    tiles_hashes: set[str] = set()
    product_rows: Counter[str] = Counter()
    accepted_manifest_total = 0
    for path in bucket_manifests:
        split = path.parents[2].name.removeprefix("split=")
        date = path.parents[1].name.removeprefix("date=")
        key = (split, date)
        if key not in daily:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = daily[key]
        row["candidate_bucket_capacity"] += int(payload["candidate_count"])
        # The quota-closing bucket can stop before consuming all rows assigned
        # to it.  Count only candidates actually evaluated in conversion rates.
        row["candidate_count"] += (
            int(payload["accepted_core_count"])
            + int(payload["rejected_count"])
            + int(payload["processing_exception_count"])
        )
        row["pre_match_eligible_count"] += int(
            payload["pre_match_eligible_count"]
        )
        row["matched_count"] += int(payload["matched_count"])
        row["accepted_count"] += int(payload["accepted_core_count"])
        row["rejected_count"] += int(payload["rejected_count"])
        row["processing_exception_count"] += int(
            payload["processing_exception_count"]
        )
        row["runtime_s"] += float(payload["runtime_s"])
        row["peak_rss_mb"] = max(
            row["peak_rss_mb"], float(payload["peak_rss_mb"])
        )
        row["bucket_count"] += 1
        accepted_manifest_total += int(payload["accepted_core_count"])
        product_rows.update(
            {
                name: int(count)
                for name, count in payload.get("product_row_counts", {}).items()
            }
        )
        code_identities.add(str(payload["code_sha"]))
        config_hashes.add(str(payload["config_sha"]))
        tiles_hashes.add(str(payload["tiles_sha"]))

    for path in sorted(root.glob("rejections/split=*/date=*/bucket=*.parquet")):
        split = path.parents[1].name.removeprefix("split=")
        date = path.parent.name.removeprefix("date=")
        if (split, date) not in expected_dates:
            continue
        frame = pd.read_parquet(path, columns=["rejection_reason"])
        rejection_reasons.update(
            frame.rejection_reason.fillna("UNKNOWN").astype(str).tolist()
        )

    unique_edges: set[str] = set()
    edge_observations: Counter[str] = Counter()
    edge_time_slot_observations: Counter[tuple[str, int]] = Counter()
    hour_observations: Counter[int] = Counter()
    road_class_rows: Counter[str] = Counter()
    direct_observed_count = 0
    accepted_rows = 0
    for manifest_path in bucket_manifests:
        split = manifest_path.parents[2].name.removeprefix("split=")
        date = manifest_path.parents[1].name.removeprefix("date=")
        if (split, date) not in expected_dates:
            continue
        bucket = manifest_path.parent
        base = pd.read_parquet(bucket / "order_base.parquet")
        accepted_rows += len(base)
        observations = pd.read_parquet(
            bucket / "link_interval_observations.parquet",
            columns=[
                "canonical_edge_uid",
                "interval_start_time",
                "measurement_source",
                "label_valid",
            ],
        )
        direct = observations[
            observations.measurement_source.eq("direct_observed")
            & observations.label_valid.fillna(False).astype(bool)
        ].copy()
        direct_observed_count += len(direct)
        direct["canonical_edge_uid"] = direct.canonical_edge_uid.astype(str)
        unique_edges.update(direct.canonical_edge_uid)
        edge_observations.update(direct.canonical_edge_uid.tolist())
        hours = (
            pd.to_datetime(
                pd.to_numeric(direct.interval_start_time, errors="coerce"),
                unit="s",
                utc=True,
            )
            .dt.tz_convert("Asia/Shanghai")
            .dt.hour
        )
        hour_observations.update(hours.dropna().astype(int).tolist())
        edge_time_slot_observations.update(
            zip(
                direct.canonical_edge_uid.tolist(),
                hours.fillna(-1).astype(int).tolist(),
            )
        )
        route_parts = pd.read_parquet(
            bucket / "route_parts.parquet",
            columns=["road_class"],
        )
        road_class_rows.update(
            route_parts.road_class.fillna("unknown").astype(str).tolist()
        )

    daily_rows = [daily[key] for key in sorted(daily)]
    for row in daily_rows:
        row["daily_quota_shortfall"] = max(
            int(row["target_count"]) - int(row["accepted_count"]), 0
        )
        row["candidate_to_accept_rate"] = (
            float(row["accepted_count"]) / int(row["candidate_count"])
            if row["candidate_count"]
            else 0.0
        )
    total_exceptions = sum(
        int(row["processing_exception_count"]) for row in daily_rows
    )
    total_shortfall = sum(int(row["daily_quota_shortfall"]) for row in daily_rows)
    expected_total = sum(int(row["target_count"]) for row in daily_rows)
    accepted_total = sum(int(row["accepted_count"]) for row in daily_rows)
    missing_dates = [
        f"{row['split']}/{row['date']}"
        for row in daily_rows
        if not row["bucket_count"]
    ]
    status = (
        "PASS"
        if not missing_dates
        and total_exceptions == 0
        and accepted_rows == accepted_manifest_total == accepted_total
        else "FAIL"
    )
    return {
        "schema_version": "stage1_input_coverage.1",
        "status": status,
        "daily": daily_rows,
        "target_total": expected_total,
        "accepted_total": accepted_total,
        "daily_quota_shortfall": total_shortfall,
        "candidate_total": sum(int(row["candidate_count"]) for row in daily_rows),
        "matched_total": sum(int(row["matched_count"]) for row in daily_rows),
        "processing_exception_count": total_exceptions,
        "accepted_order_row_count": accepted_rows,
        "accepted_manifest_total": accepted_manifest_total,
        "rejection_reasons": dict(rejection_reasons.most_common()),
        "unique_canonical_edge_count": len(unique_edges),
        "direct_observed_interval_count": direct_observed_count,
        "edge_observation_count_distribution": _quantiles(edge_observations),
        "edge_time_slot_sample_count_distribution": _quantiles(
            edge_time_slot_observations
        ),
        "road_class_route_part_rows": dict(road_class_rows.most_common()),
        "hour_direct_observed_intervals": {
            str(hour): int(hour_observations.get(hour, 0)) for hour in range(24)
        },
        "runtime_s_sum_of_buckets": sum(
            float(row["runtime_s"]) for row in daily_rows
        ),
        "peak_rss_mb": max(
            (float(row["peak_rss_mb"]) for row in daily_rows), default=0.0
        ),
        "product_row_counts": dict(product_rows),
        "code_identities": sorted(code_identities),
        "config_hashes": sorted(config_hashes),
        "tiles_hashes": sorted(tiles_hashes),
        "missing_dates": missing_dates,
    }


def _daily_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| split | date | target | accepted | candidates | conversion | shortfall | exceptions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {split} | {date} | {target_count} | {accepted_count} | "
            "{candidate_count} | {candidate_to_accept_rate:.2%} | "
            "{daily_quota_shortfall} | {processing_exception_count} |".format(
                **row
            )
        )
    return "\n".join(lines)


def finalize(config: Stage0V6Config) -> dict[str, Any]:
    coverage = aggregate(config)
    repo = config.repo_root
    root = config.path("stage1_input")
    final600_summary = config.path("output") / "summary.json"
    final600 = json.loads(final600_summary.read_text(encoding="utf-8"))
    verification_path = root / "manifests" / "verification.json"
    verification = (
        json.loads(verification_path.read_text(encoding="utf-8"))
        if verification_path.exists()
        else {"status": "NOT_RUN"}
    )
    production = config.section("production")
    freeze = {
        "schema_version": "stage0_v6_freeze_manifest.1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_status": (
            "FROZEN"
            if coverage["status"] == "PASS"
            and verification.get("status") == "PASS"
            and final600.get("status") == "PASS"
            else "NOT_FROZEN"
        ),
        "git_commit_sha": _git(repo, "rev-parse", "HEAD"),
        "working_tree_clean": not bool(_git(repo, "status", "--porcelain")),
        "code_identities": coverage["code_identities"],
        "config_sha": config.digest,
        "pbf_sha": _sha256_file(config.path("pbf")),
        "valhalla_tiles_sha": _sha256_directory(config.path("valhalla_tiles")),
        "fixed600_sample_sha": config.section("sample")["expected_sha256"],
        "fixed600_summary_sha": _sha256_file(final600_summary),
        "train_dates": production["train_dates"],
        "validation_dates": production["validation_dates"],
        "test_date": production["test_date"],
        "selection_seed": production["selection_seed"],
        "quality_thresholds": {
            "preprocess": config.section("preprocess"),
            "modeling_eligibility": config.section("modeling_eligibility"),
            "quality": config.section("quality"),
            "final_quality": config.section("final_quality"),
            "stage1_core": config.section("stage1_core"),
            "production_prefilter": production["prefilter"],
        },
        "product_schema_version": "stage1_input_v1",
        "daily_order_targets": {
            "train": production["train_daily_target"],
            "validation": production["validation_daily_target"],
            "test": production["test_target"],
        },
        "coverage": coverage,
        "verification": verification,
    }
    _atomic_json(
        repo / "stage0" / "docs" / "stage0_v6_freeze_manifest.json",
        freeze,
    )
    _atomic_json(root / "manifests" / "production_summary.json", coverage)

    coverage_md = f"""# Stage 1 Input Coverage Report

Status: **{coverage["status"]}**

## Daily coverage

{_daily_markdown(coverage["daily"])}

## Aggregate coverage

- Target orders: {coverage["target_total"]}
- Accepted orders: {coverage["accepted_total"]}
- Daily quota shortfall: {coverage["daily_quota_shortfall"]}
- Candidates evaluated: {coverage["candidate_total"]}
- Candidate-to-accept conversion: {coverage["accepted_total"] / max(coverage["candidate_total"], 1):.2%}
- Processing exceptions: {coverage["processing_exception_count"]}
- Unique canonical edges with direct observations: {coverage["unique_canonical_edge_count"]}
- Direct-observed intervals: {coverage["direct_observed_interval_count"]}
- Sum of bucket runtime: {coverage["runtime_s_sum_of_buckets"]:.1f} s
- Peak RSS: {coverage["peak_rss_mb"]:.1f} MB

## Observation distributions

- Per-edge observation counts: `{json.dumps(coverage["edge_observation_count_distribution"], ensure_ascii=False)}`
- Per-edge × hour sample counts: `{json.dumps(coverage["edge_time_slot_sample_count_distribution"], ensure_ascii=False)}`
- By road class (route-part rows): `{json.dumps(coverage["road_class_route_part_rows"], ensure_ascii=False)}`
- By hour (direct intervals): `{json.dumps(coverage["hour_direct_observed_intervals"], ensure_ascii=False)}`

## Rejections

```json
{json.dumps(coverage["rejection_reasons"], ensure_ascii=False, indent=2)}
```
"""
    _atomic_text(
        repo / "stage1" / "docs" / "stage1_input_coverage_report.md",
        coverage_md,
    )

    contract_md = """# Stage 0 to Stage 1 Contract

Stage 1 input is partitioned by `split/date/bucket`. Only rows in `order_base`
with `stage1_core_eligible=true` are core orders. Core orders require
`route_pass`, eligible GPS, resolved canonical identity, usable dynamic status,
valid conservation audits, at least eight direct intervals, and at least five
unique timed edges.

Direct link supervision comes only from `link_interval_observations` rows where
`measurement_source == "direct_observed"` and `label_valid == true`.
`interval_measurements` retains all four provenance classes:
`direct_observed`, `interval_supported`, `engine_interpolated`, and
`unresolved`. Non-direct rows never carry observed time.

`link_traversals` is the physical route-access table: one continuous access per
row. `route_parts` is the directed canonical route sequence, and
`route_segments` carries segment-level GPS, route, dynamic, and canonical
statuses. Failed candidates are absent from core products and retained only in
the lightweight rejection manifests.

Stage 0 is frozen after verification. It may be reopened only for data
loss/duplication, conservation defects, large-scale systematic route mismatch,
or a Stage 1 schema-read failure.
"""
    _atomic_text(
        repo / "stage0" / "docs" / "stage0_to_stage1_contract.md",
        contract_md,
    )

    final_report_md = f"""# Stage 0 v6 Final Report

Stage 0 v6 final status: **{freeze["freeze_status"]}**

- Fixed-600 status: {final600.get("status")}
- Stage 1 verification status: {verification.get("status")}
- Production coverage status: {coverage["status"]}
- Accepted / target: {coverage["accepted_total"]} / {coverage["target_total"]}
- Processing exceptions: {coverage["processing_exception_count"]}
- Daily quota shortfall: {coverage["daily_quota_shortfall"]}
- Test date: {production["test_date"]}
- Selection seed: {production["selection_seed"]}

The fixed 600 run is the final engineering regression, not Gate 1. No Gate 1,
6,000-order experiment, or 2,000-order trial was run. Production used the
frozen quality logic and never relaxed thresholds to fill a quota.

The authoritative reproducibility record is
`stage0/docs/stage0_v6_freeze_manifest.json`; the Stage 1 schema and label
rules are in `stage0/docs/stage0_to_stage1_contract.md`.
"""
    _atomic_text(
        repo / "stage0" / "docs" / "stage0_v6_final_report.md",
        final_report_md,
    )
    return {
        "freeze_status": freeze["freeze_status"],
        "coverage_status": coverage["status"],
        "verification_status": verification.get("status"),
        "accepted_total": coverage["accepted_total"],
        "target_total": coverage["target_total"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(load_config(args.config))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["freeze_status"] == "FROZEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
