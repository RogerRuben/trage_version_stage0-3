"""Gate 0 preflight and Gate 1 evidence-derived readiness."""

from __future__ import annotations

import importlib.metadata
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .archive import build_inventory_and_samples, list_archive_members
from .archive import sampling_run_id
from .config import Stage0Config, sha256_file
from .manifest import base_manifest, write_manifest
from .network import build_network
from .poi import build_poi


REQUIRED_PACKAGES = ("numpy", "pandas", "pyarrow", "shapely", "geopandas", "scipy", "networkx", "pyproj", "osmium")

GATE1_DATES = ("20161010", "20161014", "20161016")
GATE1_ORDERS_PER_DAY = 2_000
GATE1_MINIMUM_MANUAL_REVIEWS = 300
GATE1_MINIMUM_STRICT_CORE_REVIEWS = 50
GATE1_MINIMUM_STRICT_CORE_PRECISION = 0.90
GATE1_MANUAL_STRATUM_MINIMA = {
    "random_representative": 50,
    "ordinary_road": 50,
    "fallback": 30,
    "parallel_highway_ground": 30,
    "ramp_interchange": 30,
    "topology_gap": 30,
}


def sample_order_sha256(frame: pd.DataFrame) -> str:
    """Hash the sorted Gate order universe, independent of parquet row order.

    The text contract is one ``date|order_id`` pair per line with no terminal
    newline.  Reusing this helper in the runner and the Gate avoids a report
    asserting a sample identity that was never independently checked.
    """

    missing = {"date", "order_id"} - set(frame.columns)
    if missing:
        raise ValueError(f"sampling manifest missing columns: {sorted(missing)}")
    keys = frame.loc[:, ["date", "order_id"]].astype(str)
    if keys.duplicated().any():
        raise ValueError("sampling manifest contains duplicate date/order_id keys")
    keys = keys.sort_values(["date", "order_id"], kind="stable")
    payload = "\n".join(f"{date}|{order_id}" for date, order_id in keys.itertuples(index=False, name=None))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number not in {float("inf"), float("-inf")} else None


def _evidence_check(
    observed: Any,
    operator: str,
    threshold: Any,
    passed: bool,
    evidence_path: str,
) -> dict[str, Any]:
    return {
        "status": "PASS" if bool(passed) else "FAIL",
        "observed": observed,
        "operator": operator,
        "threshold": threshold,
        "evidence_path": evidence_path,
    }


def _load_manual_review_evidence(output: Path, summary: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    configured = summary.get("manual_review_audit_path")
    path = Path(str(configured)) if configured else output / "manual_review" / "development_review_audit.json"
    if not path.is_absolute():
        path = output / path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    return payload, path


def _source_identity(path: Path, full_hash: bool) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path) if full_hash else None,
        "reproducible_identifier": f"size={stat.st_size};mtime_ns={stat.st_mtime_ns}",
    }


def preflight(config: Stage0Config, repo: Path, full_hash: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    paths = {name: config.path(name, repo) for name in ("archive", "pbf", "poi", "output", "work", "seven_zip")}
    existence = {name: path.exists() for name, path in paths.items()}
    missing = [name for name in ("archive", "pbf", "poi", "seven_zip") if not existence[name]]
    paths["output"].mkdir(parents=True, exist_ok=True)
    paths["work"].mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(paths["output"])
    dependencies: dict[str, str | None] = {}
    for name in REQUIRED_PACKAGES:
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    archive_members = list_archive_members(paths["archive"], paths["seven_zip"]) if not missing else []
    configured_dates = [str(item) for values in config.section("split").values() for item in values]
    archived_dates = {item["date"] for item in archive_members}
    result = {
        **base_manifest(repo, config.digest, [paths[name] for name in ("archive", "pbf", "poi") if paths[name].exists()]),
        "status": "PASS" if not missing and all(dependencies.values()) and set(configured_dates) <= archived_dates and usage.free >= 20 * 1024**3 else "FAIL",
        "existence": existence,
        "missing_paths": missing,
        "dependencies": dependencies,
        "configured_dates": configured_dates,
        "archive_date_count": len(archived_dates),
        "missing_archive_dates": sorted(set(configured_dates) - archived_dates),
        "disk_free_gb": usage.free / 1024**3,
        "minimum_disk_free_gb": 20.0,
        "sources": {name: _source_identity(paths[name], full_hash and name != "archive") for name in ("archive", "pbf", "poi") if paths[name].exists()},
        "raw_write_policy": "read_only",
        "runtime_sec": time.perf_counter() - started,
    }
    write_manifest(paths["output"] / "reports" / "gate0_preflight.json", result)
    return result


def run_gate0(config: Stage0Config, repo: Path, dates: list[str] | None, force: bool = False) -> dict[str, Any]:
    pre = preflight(config, repo)
    if pre["status"] != "PASS":
        raise RuntimeError("Gate 0 preflight failed; inspect gate0_preflight.json")
    network = build_network(config, repo, force=force)
    poi = build_poi(config, repo, force=force)
    samples = build_inventory_and_samples(config, repo, dates=dates, force=force)
    result = {
        "status": "PASS" if all(item.get("status") == "PASS" for item in (pre, network, poi, samples)) else "FAIL",
        "preflight": pre,
        "network": network,
        "poi": poi,
        "sampling": samples,
    }
    write_manifest(config.path("output", repo) / "reports" / "gate0_report.json", result)
    return result


def gate1_readiness(config: Stage0Config, repo: Path, summary: dict[str, Any]) -> dict[str, Any]:
    """Evaluate Gate 1 from independently traceable, fail-closed evidence.

    Missing evidence is a failure, not an implicit zero.  The legacy boolean
    ``checks`` mapping is retained for callers, while ``evidence_checks``
    carries the observed value, comparison and evidence location.
    """

    output = config.path("output", repo)
    gate0_path = output / "reports" / "gate0_report.json"
    gate0 = json.loads(gate0_path.read_text(encoding="utf-8")) if gate0_path.exists() else {"status": "MISSING"}
    seed = int(config.section("sampling")["seed"])
    expected_dates = list(GATE1_DATES)
    expected_run_id = sampling_run_id(expected_dates, GATE1_ORDERS_PER_DAY, seed)
    sampling_path = (
        output / "manifests" / "sampling_runs" / expected_run_id / "sampling_manifest.parquet"
    )
    sample_manifest_error: str | None = None
    expected_sample_sha: str | None = None
    expected_sample_orders = 0
    per_date_sample_counts: dict[str, int] = {}
    try:
        sample_manifest = pd.read_parquet(sampling_path, columns=["date", "order_id"])
        sample_manifest["date"] = sample_manifest.date.astype(str)
        expected_sample_sha = sample_order_sha256(sample_manifest)
        expected_sample_orders = int(len(sample_manifest))
        per_date_sample_counts = {
            str(date): int(count)
            for date, count in sample_manifest.groupby("date").size().to_dict().items()
        }
    except (OSError, ValueError, KeyError) as error:
        sample_manifest_error = f"{type(error).__name__}:{error}"

    summary_dates = [str(value) for value in summary.get("dates", [])]
    summary_orders_per_day = summary.get("orders_per_day")
    summary_seed = summary.get("sampling_seed")
    summary_run_id = summary.get("sampling_run_id")
    summary_sample_sha = summary.get("sample_order_sha256")
    input_orders = _finite_number(summary.get("input_orders"))
    output_orders = _finite_number(summary.get("output_orders"))
    expected_total = len(expected_dates) * GATE1_ORDERS_PER_DAY
    mode_share = summary.get("matching_mode_share")
    mode_share = mode_share if isinstance(mode_share, dict) else {}
    fallback = _finite_number(summary.get("fallback_share"))
    if fallback is None and "geometric_fallback" in mode_share:
        fallback = _finite_number(mode_share.get("geometric_fallback"))

    full_attempt_share = _finite_number(summary.get("full_hmm_attempt_share"))
    full_failure_share = _finite_number(summary.get("full_hmm_failure_share"))
    if full_failure_share is None and output_orders and output_orders > 0:
        failure_count = _finite_number(summary.get("full_hmm_failure_count"))
        if failure_count is not None:
            full_failure_share = failure_count / output_orders
    local_attempt_share = _finite_number(summary.get("local_hmm_attempt_share"))
    if local_attempt_share is None and output_orders and output_orders > 0:
        local_count = _finite_number(summary.get("local_hmm_attempt_count"))
        if local_count is not None:
            local_attempt_share = local_count / output_orders

    manual, manual_path = _load_manual_review_evidence(output, summary)
    manual_completed = _finite_number(manual.get("completed_unique_reviews"))
    strict_completed = _finite_number(manual.get("strict_core_completed_reviews"))
    strict_precision = _finite_number(manual.get("strict_core_precision"))
    stratum_completed = manual.get("stratum_completed_counts")
    stratum_population = manual.get("stratum_population_counts")
    stratum_completed = stratum_completed if isinstance(stratum_completed, dict) else {}
    stratum_population = stratum_population if isinstance(stratum_population, dict) else {}
    manual_strata_observed: dict[str, int | None] = {}
    manual_strata_required: dict[str, int] = {}
    manual_strata_pass = True
    for name, minimum in GATE1_MANUAL_STRATUM_MINIMA.items():
        observed = _finite_number(stratum_completed.get(name))
        population = _finite_number(stratum_population.get(name))
        required = minimum if population is None else min(minimum, max(0, int(population)))
        manual_strata_observed[name] = None if observed is None else int(observed)
        manual_strata_required[name] = required
        manual_strata_pass = manual_strata_pass and observed is not None and observed >= required

    evidence: dict[str, dict[str, Any]] = {}
    evidence["gate0_pass"] = _evidence_check(
        gate0.get("status"), "==", "PASS", gate0.get("status") == "PASS", str(gate0_path)
    )
    evidence["fixed_dates"] = _evidence_check(
        summary_dates, "==", expected_dates, summary_dates == expected_dates, "run_summary"
    )
    evidence["fixed_orders_per_day"] = _evidence_check(
        summary_orders_per_day,
        "==",
        GATE1_ORDERS_PER_DAY,
        summary_orders_per_day == GATE1_ORDERS_PER_DAY,
        "run_summary",
    )
    evidence["fixed_sampling_seed"] = _evidence_check(
        summary_seed, "==", seed, summary_seed == seed, "run_summary"
    )
    evidence["sampling_run_id"] = _evidence_check(
        summary_run_id, "==", expected_run_id, summary_run_id == expected_run_id, str(sampling_path)
    )
    manifest_valid = (
        sample_manifest_error is None
        and expected_sample_orders == expected_total
        and per_date_sample_counts == {date: GATE1_ORDERS_PER_DAY for date in expected_dates}
    )
    evidence["sampling_manifest_complete"] = _evidence_check(
        {
            "orders": expected_sample_orders,
            "per_date": per_date_sample_counts,
            "error": sample_manifest_error,
        },
        "==",
        {"orders": expected_total, "per_date": {date: GATE1_ORDERS_PER_DAY for date in expected_dates}},
        manifest_valid,
        str(sampling_path),
    )
    evidence["sample_order_sha256"] = _evidence_check(
        summary_sample_sha,
        "==",
        expected_sample_sha,
        expected_sample_sha is not None and summary_sample_sha == expected_sample_sha,
        str(sampling_path),
    )
    evidence["order_accounting"] = _evidence_check(
        {"input": input_orders, "output": output_orders, "flag": summary.get("accounting_pass")},
        "==",
        {"input": expected_total, "output": expected_total, "flag": True},
        bool(summary.get("accounting_pass"))
        and input_orders == expected_total
        and output_orders == expected_total,
        "run_summary",
    )

    zero_count_metrics = {
        "processing_exception_count": "processing_exception_count",
        "internal_time_conservation": "internal_time_conservation_failures",
        "internal_distance_conservation": "internal_distance_conservation_failures",
        "duplicate_traversal_instance": "duplicate_traversal_instance_error_count",
        "position_aware_distance": "invalid_position_aware_distance_count",
        "inferred_dynamic_label": "observed_dynamic_label_on_inferred_edge_count",
        "hmm_path_distance_consistency": "hmm_path_distance_mismatch_count",
        "forbidden_final_movement": "final_forbidden_movement_count",
        "non_rejected_topology_gap": "non_rejected_topology_gap_order_count",
    }
    for check_name, summary_key in zero_count_metrics.items():
        value = _finite_number(summary.get(summary_key))
        evidence[check_name] = _evidence_check(
            value, "==", 0, value is not None and value == 0, "run_summary"
        )
    evidence["raw_movement_audit_available"] = _evidence_check(
        summary.get("raw_movement_audit_available"),
        "is",
        True,
        summary.get("raw_movement_audit_available") is True,
        "run_summary",
    )
    evidence["cold_hot_output_equality"] = _evidence_check(
        summary.get("cold_hot_output_equality_pass"),
        "is",
        True,
        summary.get("cold_hot_output_equality_pass") is True,
        str(summary.get("cold_hot_equality_audit_path", "run_summary")),
    )

    evidence["full_hmm_attempt_share"] = _evidence_check(
        full_attempt_share,
        "<=",
        0.20,
        full_attempt_share is not None and full_attempt_share <= 0.20,
        "performance",
    )
    evidence["full_hmm_failure_share"] = _evidence_check(
        full_failure_share,
        "<=",
        0.05,
        full_failure_share is not None and full_failure_share <= 0.05,
        "performance",
    )
    evidence["local_hmm_attempt_share_reported"] = _evidence_check(
        local_attempt_share,
        "is finite",
        True,
        local_attempt_share is not None and 0 <= local_attempt_share <= 1,
        "performance",
    )
    evidence["fallback_share"] = _evidence_check(
        fallback,
        "<=",
        0.10,
        fallback is not None and fallback <= 0.10,
        "run_summary",
    )
    peak_memory = _finite_number(summary.get("peak_memory_mb"))
    evidence["peak_memory_below_12gb"] = _evidence_check(
        peak_memory,
        "<=",
        12 * 1024,
        peak_memory is not None and peak_memory <= 12 * 1024,
        "run_summary",
    )

    manual_provenance_pass = (
        manual.get("sampling_run_id") == expected_run_id
        and manual.get("sample_order_sha256") == expected_sample_sha
        and manual.get("config_hash") == config.digest
    )
    evidence["manual_review_provenance"] = _evidence_check(
        {
            "sampling_run_id": manual.get("sampling_run_id"),
            "sample_order_sha256": manual.get("sample_order_sha256"),
            "config_hash": manual.get("config_hash"),
        },
        "==",
        {
            "sampling_run_id": expected_run_id,
            "sample_order_sha256": expected_sample_sha,
            "config_hash": config.digest,
        },
        manual_provenance_pass,
        str(manual_path),
    )
    evidence["manual_review_count"] = _evidence_check(
        manual_completed,
        ">=",
        GATE1_MINIMUM_MANUAL_REVIEWS,
        manual_completed is not None and manual_completed >= GATE1_MINIMUM_MANUAL_REVIEWS,
        str(manual_path),
    )
    evidence["manual_strict_core_count"] = _evidence_check(
        strict_completed,
        ">=",
        GATE1_MINIMUM_STRICT_CORE_REVIEWS,
        strict_completed is not None and strict_completed >= GATE1_MINIMUM_STRICT_CORE_REVIEWS,
        str(manual_path),
    )
    evidence["manual_strict_core_precision"] = _evidence_check(
        strict_precision,
        ">=",
        GATE1_MINIMUM_STRICT_CORE_PRECISION,
        strict_precision is not None and strict_precision >= GATE1_MINIMUM_STRICT_CORE_PRECISION,
        str(manual_path),
    )
    evidence["manual_review_strata"] = _evidence_check(
        manual_strata_observed,
        ">= per stratum",
        manual_strata_required,
        manual_strata_pass,
        str(manual_path),
    )
    evidence["manual_review_audit_pass"] = _evidence_check(
        {"status": manual.get("status"), "schema_errors": manual.get("schema_errors")},
        "==",
        {"status": "PASS", "schema_errors": []},
        manual.get("status") == "PASS" and manual.get("schema_errors") == [],
        str(manual_path),
    )

    checks = {name: row["status"] == "PASS" for name, row in evidence.items()}
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "evidence_schema_version": "stage0_v5_gate1_evidence_v2",
        "evidence_checks": evidence,
        "expected_contract": {
            "dates": expected_dates,
            "orders_per_day": GATE1_ORDERS_PER_DAY,
            "sampling_seed": seed,
            "sampling_run_id": expected_run_id,
            "sample_order_sha256": expected_sample_sha,
        },
        "summary": summary,
        "gate2_allowed": all(checks.values()),
    }
    write_manifest(output / "reports" / "gate1_readiness.json", result)
    lines = [
        "# Stage 0 v5 Gate 1 readiness",
        "",
        f"**Status: {result['status']}**",
        "",
        "## Measured checks",
        "",
        *[
            f"- `{name}`: {row['status']} (observed={row['observed']!r}; "
            f"required {row['operator']} {row['threshold']!r})"
            for name, row in evidence.items()
        ],
        "",
        "## Measured summary",
        "",
        f"- Orders: {summary.get('output_orders', 0)}/{summary.get('input_orders', 0)}",
        f"- Matching modes: `{json.dumps(summary.get('matching_mode_share', {}), sort_keys=True)}`",
        f"- Quality counts: `{json.dumps(summary.get('quality_counts', {}), sort_keys=True)}`",
        f"- Topology gaps: {summary.get('topology_gap_count', 'MISSING')}",
        f"- Mean inferred-distance share: {summary.get('mean_inferred_distance_share', 'MISSING')}",
        "",
        "Gate 2 is blocked unless every measured check is PASS.",
    ]
    (output / "reports" / "gate1_readiness_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def require_test_freeze(config: Stage0Config, repo: Path, dates: list[str]) -> None:
    """Hard-stop any Test materialization or matching before the freeze gate."""
    test_dates = {str(value) for value in config.section("split")["test"]}
    if not test_dates.intersection(map(str, dates)):
        return
    freeze_path = config.path("output", repo) / "manifests" / "stage0_freeze_manifest.json"
    if not freeze_path.exists():
        raise RuntimeError("Test dates are locked until stage0_freeze_manifest.json exists")
    manifest = json.loads(freeze_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN" or manifest.get("config_hash") != config.digest:
        raise RuntimeError("Test dates require a FROZEN manifest for the current config hash")


def freeze(config: Stage0Config, repo: Path) -> dict[str, Any]:
    output = config.path("output", repo)
    gate1_path = output / "reports" / "gate1_readiness.json"
    if not gate1_path.exists() or json.loads(gate1_path.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("freeze requires a measured Gate 1 PASS")
    tests = subprocess.run([str(config.path("work", repo) / ".venv" / "Scripts" / "python.exe"), "-m", "pytest", "-q", "stage0/tests"], cwd=repo, capture_output=True, text=True)
    if tests.returncode != 0:
        raise RuntimeError("freeze requires Stage 0 tests to pass")
    manifest = {
        **base_manifest(repo, config.digest, [config.source, output / "network" / "network_manifest.json"]),
        "status": "FROZEN",
        "schema_version": config.values["schema_version"],
        "network_config": config.section("network"),
        "matcher_config": config.section("candidate"),
        "hmm_config": config.section("hmm"),
        "quality_config": config.section("quality"),
        "sampling_config": config.section("sampling"),
        "test_command": "python -m pytest -q stage0/tests",
        "test_stdout": tests.stdout[-4000:],
    }
    write_manifest(output / "manifests" / "stage0_freeze_manifest.json", manifest)
    return manifest
