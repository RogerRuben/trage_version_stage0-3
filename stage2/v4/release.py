"""Fail-closed binding to the frozen Stage 1 v3 release."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .config import Stage2V4Config
from .contracts import Stage2V4ContractError
from .io import canonical_json_bytes, sha256_bytes, sha256_file


@dataclass(frozen=True)
class Stage1ReleaseBinding:
    release_manifest: dict[str, Any]
    model_manifest: dict[str, Any]
    output_summary: dict[str, Any]
    bucket_manifests: tuple[Path, ...]
    resolved_release_commit: str


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage2V4ContractError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise Stage2V4ContractError(f"{label} must be a JSON object: {path}")
    return value


def _expect(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise Stage2V4ContractError(
            f"Stage 1 release mismatch for {label}: "
            f"expected {expected!r}, got {actual!r}"
        )


def _resolve_tag(repository: Path, tag: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-list", "-n", "1", tag],
        capture_output=True,
        text=True,
        check=False,
    )
    resolved = result.stdout.strip()
    if result.returncode != 0 or not resolved:
        message = result.stderr.strip() or "tag did not resolve"
        raise Stage2V4ContractError(f"cannot resolve frozen tag {tag!r}: {message}")
    return resolved


def _parquet_schema_sha256(path: Path) -> str:
    schema = pq.read_schema(path)
    fields = [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": bool(field.nullable),
        }
        for field in schema
    ]
    return sha256_bytes(canonical_json_bytes(fields))


def validate_release_manifest_payload(
    manifest: dict[str, Any],
    config: Stage2V4Config,
) -> None:
    expected = config.section("stage1_release")
    identity = manifest.get("stage1_identity")
    production = manifest.get("production")
    frozen_inputs = manifest.get("frozen_inputs")
    stage0 = manifest.get("stage0_release")
    if not all(isinstance(value, dict) for value in (identity, production, frozen_inputs, stage0)):
        raise Stage2V4ContractError("Stage 1 release manifest has incomplete identity sections")

    comparisons = {
        "release_tag": (manifest.get("release_tag"), expected["release_tag"]),
        "engineering_status": (
            manifest.get("engineering_status"),
            expected["engineering_status"],
        ),
        "stage1 config SHA": (
            identity.get("config_sha256"),
            expected["config_sha256"],
        ),
        "Stage 1 code SHA": (identity.get("code_sha"), expected["code_sha"]),
        "Stage 1 model ID": (identity.get("model_id"), expected["model_id"]),
        "model schema": (
            identity.get("model_schema_version"),
            expected["model_schema_version"],
        ),
        "output summary schema": (
            identity.get("output_summary_schema_version"),
            expected["output_summary_schema_version"],
        ),
        "bucket count": (frozen_inputs.get("bucket_count"), expected["bucket_count"]),
        "accepted order count": (
            frozen_inputs.get("accepted_order_count"),
            expected["accepted_order_count"],
        ),
        "completed bucket count": (
            production.get("completed_bucket_count"),
            expected["bucket_count"],
        ),
        "traversal label count": (
            production.get("traversal_label_count"),
            expected["traversal_label_count"],
        ),
        "model bundle SHA": (
            production.get("model_bundle_sha256"),
            expected["model_bundle_sha256"],
        ),
        "output manifest aggregate SHA": (
            production.get("output_manifest_aggregate_sha256"),
            expected["output_manifest_aggregate_sha256"],
        ),
    }
    for label, (actual, wanted) in comparisons.items():
        _expect(actual, wanted, label)

    expected_stage0 = expected["stage0_release"]
    for key in ("tag", "commit", "source_content_hash", "freeze_manifest_sha256"):
        _expect(stage0.get(key), expected_stage0[key], f"Stage 0 {key}")


def _validate_model_manifest(
    model_root: Path,
    manifest: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    comparisons = {
        "model schema": (manifest.get("schema_version"), expected["model_schema_version"]),
        "model ID": (manifest.get("model_id"), expected["model_id"]),
        "model config SHA": (manifest.get("config_sha"), expected["config_sha256"]),
        "model code SHA": (manifest.get("stage1_code_sha"), expected["code_sha"]),
        "label schema": (
            manifest.get("label_schema_version"),
            expected["label_schema_version"],
        ),
    }
    for label, (actual, wanted) in comparisons.items():
        _expect(actual, wanted, label)

    manifest_core = {key: value for key, value in manifest.items() if key != "model_id"}
    _expect(
        manifest.get("model_id"),
        sha256_bytes(canonical_json_bytes(manifest_core)),
        "self-derived model ID",
    )
    file_map = {
        "directed_edge_catalog": "directed_edge_catalog.parquet",
        "lcs": "lcs_histograms.parquet",
        "metadata": "histogram_metadata.json",
        "reference": "reference_histograms.parquet",
        "rts": "rts_histograms.parquet",
        "support_counts": "support_counts.parquet",
    }
    recorded = manifest.get("files")
    if not isinstance(recorded, dict):
        raise Stage2V4ContractError("Stage 1 model manifest has no file identities")
    for key, filename in file_map.items():
        path = model_root / filename
        if not path.is_file():
            raise Stage2V4ContractError(f"missing Stage 1 model file: {path}")
        _expect(sha256_file(path), recorded.get(key), f"model file {filename}")


def bind_stage1_release(
    release_path: str | Path,
    output_root: str | Path,
    input_root: str | Path,
    model_root: str | Path,
    config: Stage2V4Config,
    *,
    repository: str | Path | None = None,
) -> Stage1ReleaseBinding:
    """Validate every frozen identity needed before Stage 2 reads label rows."""

    release_source = Path(release_path)
    output = Path(output_root)
    source_input = Path(input_root)
    models = Path(model_root)
    expected = config.section("stage1_release")

    if sha256_file(release_source) != expected["release_manifest_sha256"]:
        raise Stage2V4ContractError("Stage 1 release manifest file SHA mismatch")
    release = _read_json(release_source, "Stage 1 release manifest")
    validate_release_manifest_payload(release, config)

    repo = Path(repository) if repository is not None else Path(__file__).resolve().parents[2]
    resolved_commit = _resolve_tag(repo, expected["release_tag"])
    _expect(resolved_commit, expected["release_commit"], "release tag commit")

    model_manifest = _read_json(models / "model_manifest.json", "Stage 1 model manifest")
    _validate_model_manifest(models, model_manifest, expected)

    summary = _read_json(output / "stage1_v3_summary.json", "Stage 1 output summary")
    summary_checks = {
        "output summary schema": (
            summary.get("schema_version"),
            expected["output_summary_schema_version"],
        ),
        "summary status": (summary.get("engineering_status"), "PASS"),
        "summary model ID": (summary.get("model_id"), expected["model_id"]),
        "summary config SHA": (summary.get("config_sha"), expected["config_sha256"]),
        "summary code SHA": (summary.get("stage1_code_sha"), expected["code_sha"]),
        "summary bucket count": (summary.get("bucket_count"), expected["bucket_count"]),
    }
    for label, (actual, wanted) in summary_checks.items():
        _expect(actual, wanted, label)

    paths = tuple(sorted(output.glob("split=*/date=*/bucket=*/manifest.json")))
    _expect(len(paths), expected["bucket_count"], "physical bucket manifest count")
    model_bucket_identities = model_manifest.get("input_bucket_identities")
    if not isinstance(model_bucket_identities, dict):
        raise Stage2V4ContractError("model manifest has no frozen input bucket identities")
    _expect(len(model_bucket_identities), expected["bucket_count"], "model bucket identities")

    output_schemas = expected["output_schema_hashes"]
    input_schemas = expected["input_schema_hashes"]
    for path in paths:
        manifest = _read_json(path, "Stage 1 output bucket manifest")
        relative = path.parent.relative_to(output)
        values = {part.split("=", 1)[0]: part.split("=", 1)[1] for part in relative.parts}
        bucket_key = f"{values['split']}/{values['date']}/{int(values['bucket']):05d}"
        frozen_bucket = model_bucket_identities.get(bucket_key)
        if not isinstance(frozen_bucket, dict):
            raise Stage2V4ContractError(f"unexpected Stage 1 output bucket: {bucket_key}")
        bucket_checks = {
            "bucket schema": (
                manifest.get("schema_version"),
                expected["output_bucket_schema_version"],
            ),
            "bucket status": (manifest.get("engineering_status"), "PASS"),
            "bucket model": (manifest.get("model_id"), expected["model_id"]),
            "bucket config": (manifest.get("config_sha"), expected["config_sha256"]),
            "bucket code": (manifest.get("stage1_code_sha"), expected["code_sha"]),
            "bucket input identity": (
                manifest.get("input_bucket_sha"),
                frozen_bucket.get("bucket_sha"),
            ),
        }
        for label, (actual, wanted) in bucket_checks.items():
            _expect(actual, wanted, f"{bucket_key} {label}")
        for product, wanted in output_schemas.items():
            _expect(
                manifest.get("output_schema_hashes", {}).get(product),
                wanted,
                f"{bucket_key} {product} schema manifest",
            )
            _expect(
                _parquet_schema_sha256(path.parent / f"{product}.parquet"),
                wanted,
                f"{bucket_key} {product} physical schema",
            )
        input_bucket = (
            source_input
            / f"split={values['split']}"
            / f"date={values['date']}"
            / f"bucket={int(values['bucket']):05d}"
        )
        for product, wanted in input_schemas.items():
            _expect(
                manifest.get("input_schema_hashes", {}).get(product),
                wanted,
                f"{bucket_key} {product} input schema manifest",
            )
            _expect(
                _parquet_schema_sha256(input_bucket / f"{product}.parquet"),
                wanted,
                f"{bucket_key} {product} physical input schema",
            )

    return Stage1ReleaseBinding(
        release_manifest=release,
        model_manifest=model_manifest,
        output_summary=summary,
        bucket_manifests=paths,
        resolved_release_commit=resolved_commit,
    )
