"""Strict artifact-manifest loading, hashing, and dependency validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema
import pandas as pd
import yaml


class ManifestError(RuntimeError):
    """Raised when artifact lineage is absent, ambiguous, or invalid."""


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ManifestError(f"YAML root must be a mapping: {path}")
    return value


def config_sha256(path: Path) -> str:
    return canonical_json_sha256(load_yaml(path))


def parquet_row_count(path: Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


def schema_hash(path: Path) -> str | None:
    if path.suffix.lower() not in {".parquet", ".csv"}:
        return None
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        return hashlib.sha256(str(pq.ParquetFile(path).schema_arrow).encode("utf-8")).hexdigest()
    frame = pd.read_csv(path, nrows=0)
    return canonical_json_sha256(list(frame.columns))


def file_record(path: Path, role: str, workspace: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(workspace.resolve()).as_posix()
    except ValueError as exc:
        raise ManifestError(f"Artifact file must be inside workspace: {resolved}") from exc
    rows: int | None = None
    if path.suffix.lower() == ".parquet":
        rows = parquet_row_count(path)
    elif path.suffix.lower() == ".csv":
        rows = int(sum(1 for _ in path.open("r", encoding="utf-8-sig", errors="replace")) - 1)
    return {
        "role": role,
        "path": relative,
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
        "row_count": max(0, rows) if rows is not None else None,
        "schema_hash": schema_hash(resolved),
    }


@dataclass(frozen=True)
class ArtifactManifest:
    path: Path
    data: dict[str, Any]

    @property
    def artifact_id(self) -> str:
        return str(self.data["artifact_id"])

    @property
    def status(self) -> str:
        return str(self.data["status"])

    @property
    def audit_status(self) -> str:
        return str(self.data["audit"]["status"])

    @property
    def digest(self) -> str:
        return sha256_file(self.path)

    def validate_files(self, workspace: Path) -> list[str]:
        errors: list[str] = []
        for record in self.data.get("files", []):
            path = workspace / record["path"]
            if not path.is_file():
                errors.append(f"missing:{record['path']}")
                continue
            if path.stat().st_size != int(record["size_bytes"]):
                errors.append(f"size_mismatch:{record['path']}")
            if sha256_file(path) != record["sha256"]:
                errors.append(f"hash_mismatch:{record['path']}")
        return errors


def load_manifest(path: Path, schema_path: Path, workspace: Path) -> ArtifactManifest:
    if not path.is_file():
        raise ManifestError(f"Explicit manifest does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    try:
        jsonschema.validate(data, schema)
    except jsonschema.ValidationError as exc:
        raise ManifestError(f"Manifest schema violation in {path}: {exc.message}") from exc
    manifest = ArtifactManifest(path=path, data=data)
    errors = manifest.validate_files(workspace)
    if errors:
        raise ManifestError(f"Manifest file validation failed for {path}: {errors[:10]}")
    return manifest


def require_canonical_input(manifest: ArtifactManifest) -> None:
    if manifest.status != "canonical":
        raise ManifestError(
            f"Canonical run rejected {manifest.status} input {manifest.artifact_id}"
        )
    if manifest.audit_status != "PASS":
        raise ManifestError(
            f"Canonical run rejected unaudited input {manifest.artifact_id}: {manifest.audit_status}"
        )


def write_manifest(path: Path, data: dict[str, Any], schema_path: Path) -> None:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(data, schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

