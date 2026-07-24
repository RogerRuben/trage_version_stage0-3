"""Append-only run registry with canonical-run uniqueness checks."""

from __future__ import annotations

import csv
import hashlib
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path


FIELDS = [
    "run_id", "stage", "commit", "config_hash", "input_manifest_hash", "seed",
    "started_at", "finished_at", "status", "canonical", "supersedes_run_id",
    "audit_status", "output_path",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def new_run_id(pipeline_version: str, seed: int) -> str:
    suffix = uuid.uuid4().hex[:10]
    return f"{pipeline_version}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-s{seed}-{suffix}"


def combined_manifest_hash(hashes: list[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(hashes):
        digest.update(value.encode("ascii"))
    return digest.hexdigest()


class RunRegistry:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=FIELDS).writeheader()

    def rows(self) -> list[dict[str, str]]:
        with self.path.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def assert_unique_canonical_success(self, stage: str, config_hash: str, supersedes: str = "") -> None:
        active = [
            row for row in self.rows()
            if row["stage"] == stage
            and row["config_hash"] == config_hash
            and row["canonical"].lower() == "true"
            and row["status"] == "SUCCESS"
            and row["run_id"] != supersedes
        ]
        if active:
            raise RuntimeError(
                f"Canonical successful run already exists for {stage}/{config_hash[:12]}: "
                f"{[row['run_id'] for row in active]}"
            )

    def append(self, row: dict[str, object]) -> None:
        unknown = set(row) - set(FIELDS)
        if unknown:
            raise ValueError(f"Unknown run-registry fields: {sorted(unknown)}")
        complete = {field: row.get(field, "") for field in FIELDS}
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=FIELDS).writerow(complete)

