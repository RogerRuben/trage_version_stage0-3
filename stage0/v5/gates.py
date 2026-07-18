"""Gate 0 preflight and Gate 1 evidence-derived readiness."""

from __future__ import annotations

import importlib.metadata
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .archive import build_inventory_and_samples, list_archive_members
from .config import Stage0Config, sha256_file
from .manifest import base_manifest, write_manifest
from .network import build_network
from .poi import build_poi


REQUIRED_PACKAGES = ("numpy", "pandas", "pyarrow", "shapely", "geopandas", "scipy", "networkx", "pyproj", "osmium")


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
    output = config.path("output", repo)
    gate0_path = output / "reports" / "gate0_report.json"
    gate0 = json.loads(gate0_path.read_text(encoding="utf-8")) if gate0_path.exists() else {"status": "MISSING"}
    mode_share = summary.get("matching_mode_share", {})
    fallback = float(mode_share.get("geometric_fallback", 0.0))
    rejected = int(summary.get("quality_counts", {}).get("rejected", 0))
    total = max(int(summary.get("output_orders", 0)), 1)
    checks = {
        "gate0_pass": gate0.get("status") == "PASS",
        "order_accounting": bool(summary.get("accounting_pass")),
        "time_conservation": int(summary.get("time_conservation_failures", 1)) == 0,
        "distance_conservation": int(summary.get("distance_conservation_failures", 1)) == 0,
        "fallback_share_below_25pct": fallback <= 0.25,
        "rejected_share_below_50pct": rejected / total <= 0.5,
        "peak_memory_below_12gb": float(summary.get("peak_memory_mb", float("inf"))) <= 12 * 1024,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
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
        *[f"- `{name}`: {'PASS' if passed else 'FAIL'}" for name, passed in checks.items()],
        "",
        "## Measured summary",
        "",
        f"- Orders: {summary.get('output_orders', 0)}/{summary.get('input_orders', 0)}",
        f"- Matching modes: `{json.dumps(summary.get('matching_mode_share', {}), sort_keys=True)}`",
        f"- Quality counts: `{json.dumps(summary.get('quality_counts', {}), sort_keys=True)}`",
        f"- Topology gaps: {summary.get('topology_gap_count', 0)}",
        f"- Mean inferred-distance share: {summary.get('mean_inferred_distance_share', 0.0):.6f}",
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
