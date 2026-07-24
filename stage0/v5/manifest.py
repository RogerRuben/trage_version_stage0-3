"""Resume-safe partition and artifact manifests."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

from .config import git_sha, sha256_file


def dependency_versions(names: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    return versions


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def base_manifest(repo: Path, config_hash: str, inputs: list[Path]) -> dict[str, Any]:
    input_rows = []
    for path in inputs:
        stat = path.stat()
        input_rows.append({
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256_file(path) if stat.st_size <= 1024**3 else None,
            "reproducible_identifier": f"size={stat.st_size};mtime_ns={stat.st_mtime_ns}",
        })
    return {
        "schema_version": "stage0_v5_manifest_v1",
        "created_unix": time.time(),
        "git_sha": git_sha(repo),
        "config_hash": config_hash,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "inputs": input_rows,
        "dependencies": dependency_versions(
            ["numpy", "pandas", "pyarrow", "shapely", "geopandas", "scipy", "networkx", "pyproj", "osmium"]
        ),
    }


def partition_complete(path: Path, expected: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "PASS" and all(payload.get(key) == value for key, value in expected.items())
