"""Bucket-streaming fit and transform orchestration for Stage 1 v3."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Sequence

import pandas as pd

from .aggregation import aggregate_order_labels
from .input_adapter import (
    BucketRef,
    Stage0Bucket,
    derive_movement_direction_context,
    iter_stage0_buckets,
    load_stage0_bucket,
    load_stage0_fit_bucket,
    load_stage0_fit_route_parts,
)
from .freeze import load_stage0_freeze_manifest
from .io import (
    atomic_output_directory,
    atomic_write_json,
    atomic_write_parquet,
    bucket_input_identity,
    canonical_json_bytes,
    parquet_column_names,
    parquet_row_count,
    parquet_schema_sha256,
    sha256_bytes,
    sha256_file,
    stage1_v3_code_identity,
)
from .models import (
    Stage1V3Models,
    load_model_bundle,
    write_model_bundle,
)
from .primitives import build_interval_labels, build_traversal_primitives
from .preflight import validate_preflight_for_fit
from .references import (
    apply_percentile_labels,
    apply_reference_labels,
    fit_label_histograms,
    fit_reference_histograms,
)
from .schema import (
    ContractError,
    OUTPUT_BUCKET_SCHEMA_VERSION,
    OUTPUT_PRIMARY_KEYS,
    OUTPUT_REQUIRED_COLUMNS,
    OUTPUT_SUMMARY_SCHEMA_VERSION,
)
from .support import (
    apply_directed_support,
    build_directed_edge_catalog,
    fit_directed_support_from_observations,
)

if TYPE_CHECKING:
    from .config import Stage1V3Config


OUTPUT_PRODUCTS = (
    "interval_labels",
    "traversal_labels",
    "route_sequence_context",
    "movement_context",
    "order_labels",
    "order_label_quality",
)


def _current_rss_mb() -> float:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss / (1024 * 1024))
    except (ImportError, OSError):
        return float("nan")


def _execution_guard(
    config: "Stage1V3Config",
    *,
    allow_review_candidate: bool,
) -> None:
    from .config import validate_config

    validate_config(config)
    status = str(config.data.get("status", ""))
    if status != "frozen_for_execution" and not allow_review_candidate:
        raise ContractError(
            "Stage1 v3 config is not frozen_for_execution; pass the explicit "
            "review-candidate override only for controlled engineering tests"
        )


def _verified_code_identity(expected: str | None) -> str:
    actual = stage1_v3_code_identity()
    if expected is not None and str(expected).strip() != actual:
        raise ContractError(
            "provided Stage1 code identity differs from the executable source tree"
        )
    return actual


def _require_expected_dates(
    refs: Sequence[BucketRef],
    expected: dict[str, Sequence[str]],
) -> None:
    actual: dict[str, set[str]] = {}
    for ref in refs:
        actual.setdefault(ref.split, set()).add(ref.date)
    failures: list[str] = []
    for split, dates in expected.items():
        expected_dates = set(dates)
        actual_dates = actual.get(split, set())
        if actual_dates != expected_dates:
            failures.append(
                f"{split}: missing={sorted(expected_dates - actual_dates)}, "
                f"unexpected={sorted(actual_dates - expected_dates)}"
            )
    if failures:
        raise ContractError("Stage1 v3 input dates are incomplete: " + "; ".join(failures))


def _single_upstream_identity(refs: Sequence[BucketRef]) -> dict[str, str]:
    identity: dict[str, str] = {}
    for field in ("code_sha", "config_sha", "tiles_sha"):
        values = {str(ref.manifest.get(field, "")).strip() for ref in refs}
        if "" in values or len(values) != 1:
            raise ContractError(
                f"Stage0 input must have one non-empty {field}; found {sorted(values)}"
            )
        identity[field] = next(iter(values))
    return identity


def _bucket_key(ref: BucketRef) -> str:
    return f"{ref.split}/{ref.date}/{ref.bucket:05d}"


def _input_identities(refs: Sequence[BucketRef]) -> dict[str, Any]:
    return {
        _bucket_key(ref): {
            **bucket_input_identity(ref.path),
            "upstream_code_sha": str(ref.manifest["code_sha"]),
            "upstream_config_sha": str(ref.manifest["config_sha"]),
            "upstream_tiles_sha": str(ref.manifest["tiles_sha"]),
            "accepted_core_count": int(ref.manifest["accepted_core_count"]),
            "product_row_counts": {
                str(name): int(count)
                for name, count in sorted(
                    ref.manifest["product_row_counts"].items()
                )
            },
        }
        for ref in refs
    }


def _source_manifest_id(identities: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(identities))


def _require_uniform_input_schemas(identities: dict[str, Any]) -> None:
    """Require schema evidence, not identical physical nullable encodings."""

    for identity in identities.values():
        schemas = identity.get("schemas")
        if not isinstance(schemas, dict):
            raise ContractError("input bucket identity is missing schema hashes")
        counts = identity.get("product_row_counts")
        if not isinstance(counts, dict):
            raise ContractError("input bucket identity is missing product row counts")
        if set(schemas) != set(counts):
            raise ContractError(
                "input bucket schema evidence and product counts differ"
            )
        if any(not str(value).strip() for value in schemas.values()):
            raise ContractError("input bucket contains an empty schema hash")


def _require_global_order_uniqueness(refs: Sequence[BucketRef]) -> None:
    """Reject order leakage before any train statistic is fitted."""

    seen: dict[str, str] = {}
    for ref in refs:
        accepted = int(ref.manifest["accepted_core_count"])
        if accepted == 0:
            continue
        path = ref.path / "order_base.parquet"
        try:
            frame = pd.read_parquet(path, columns=["order_id"])
        except Exception as exc:
            raise ContractError(f"cannot inspect order identity from {path}: {exc}") from exc
        if len(frame) != accepted or frame["order_id"].isna().any():
            raise ContractError(f"invalid order_base identity rows in {path}")
        keys = frame["order_id"].astype(str)
        if keys.duplicated(keep=False).any():
            raise ContractError(f"duplicate order_id inside {path}")
        partition = _bucket_key(ref)
        for order_id in keys:
            previous = seen.get(order_id)
            if previous is not None:
                raise ContractError(
                    f"order_id {order_id!r} appears in {previous} and {partition}"
                )
            seen[order_id] = partition


def _primitive_batches(
    refs: Iterable[BucketRef],
    config: "Stage1V3Config",
    *,
    reference_models: Stage1V3Models | None = None,
    reference_fit_manifest_id: str | None = None,
    leave_one_out_reference: bool = False,
    trusted_fit: bool = False,
) -> Iterator[pd.DataFrame]:
    for ref in refs:
        bucket = (
            load_stage0_fit_bucket(ref, config)
            if trusted_fit
            else load_stage0_bucket(ref, config)
        )
        primitives = build_traversal_primitives(
            bucket.link_interval_observations,
            bucket.link_traversals,
            bucket.route_parts,
            config,
        )
        primitives.insert(0, "date", ref.date)
        primitives.insert(0, "split", ref.split)
        if reference_models is not None:
            primitives = apply_reference_labels(
                primitives,
                reference_models.reference,
                config,
                reference_fit_manifest_id=(
                    reference_fit_manifest_id or reference_models.model_id
                ),
                reference_model_id=reference_models.model_id,
                leave_one_out=leave_one_out_reference,
            )
        yield primitives


def fit_stage1_v3(
    input_root: str | Path,
    model_root: str | Path,
    stage0_freeze_manifest: str | Path,
    validated_preflight: str | Path,
    config: "Stage1V3Config",
    *,
    stage1_code_sha: str | None = None,
    resume: bool = True,
    allow_review_candidate: bool = False,
) -> dict[str, Any]:
    """Fit train-only pace references and label CDFs without full-data concat."""

    started = time.perf_counter()
    initial_rss_mb = _current_rss_mb()
    _execution_guard(config, allow_review_candidate=allow_review_candidate)
    code_sha = _verified_code_identity(stage1_code_sha)
    freeze = load_stage0_freeze_manifest(stage0_freeze_manifest, config)
    all_refs = list(iter_stage0_buckets(input_root, config))
    _require_expected_dates(
        all_refs,
        {
            "train": config.train_dates,
            "validation": config.validation_dates,
            "test": (config.test_date,),
        },
    )
    _require_global_order_uniqueness(all_refs)
    preflight_records = validate_preflight_for_fit(
        validated_preflight,
        input_root,
        all_refs,
        config,
    )
    validated_input_manifest_id = sha256_bytes(
        canonical_json_bytes(
            [preflight_records[_bucket_key(ref)] for ref in all_refs]
        )
    )
    upstream_identity = _single_upstream_identity(all_refs)
    freeze.validate_bucket_identity(upstream_identity)
    identities = _input_identities(all_refs)
    _require_uniform_input_schemas(identities)
    source_id = _source_manifest_id(identities)
    refs = [ref for ref in all_refs if ref.split == "train"]

    destination = Path(model_root)
    if destination.exists():
        if not resume:
            raise ContractError(f"Stage1 v3 model root already exists: {destination}")
        models = load_model_bundle(destination, config)
        manifest = models.manifest
        if (
            manifest.get("source_manifest_id") != source_id
            or manifest.get("stage1_code_sha") != code_sha
            or manifest.get("validated_input_manifest_id")
            != validated_input_manifest_id
            or manifest.get("upstream_identity") != upstream_identity
            or manifest.get("stage0_freeze_identity", {}).get("manifest_sha")
            != freeze.manifest_sha
        ):
            raise ContractError(
                "existing Stage1 v3 models do not match input/config/code identity"
            )
        execution_path = destination / "fit_execution.json"
        if execution_path.is_file():
            return {**manifest, "execution": _read_json(execution_path)}
        return manifest

    edge_catalog = build_directed_edge_catalog(
        load_stage0_fit_route_parts(ref)
        for ref in refs
    )
    support = fit_directed_support_from_observations(
        (
            load_stage0_fit_bucket(
                ref, config
            ).link_interval_observations
            for ref in refs
        ),
        edge_catalog,
        config,
    )
    reference = fit_reference_histograms(
        _primitive_batches(refs, config, trusted_fit=True),
        config,
    )

    reference_shell = Stage1V3Models(
        reference=reference,
        lcs=reference,
        rts=reference,
        support=support,
        manifest={"model_id": source_id},
    )
    normalization = fit_label_histograms(
        _primitive_batches(
            refs,
            config,
            reference_models=reference_shell,
            reference_fit_manifest_id=source_id,
            leave_one_out_reference=True,
            trusted_fit=True,
        ),
        config,
    )
    manifest = write_model_bundle(
        destination,
        reference=reference,
        lcs=normalization["lcs"],
        rts=normalization["rts"],
        support=support,
        config=config,
        source_manifest_id=source_id,
        input_bucket_identities=identities,
        upstream_identity=upstream_identity,
        stage0_freeze_identity={
            "manifest_sha": freeze.manifest_sha,
            "git_commit_sha": freeze.manifest["git_commit_sha"],
            "config_sha": freeze.config_sha,
            "pbf_sha": freeze.manifest["pbf_sha"],
            "valhalla_tiles_sha": freeze.tiles_sha,
            "fixed600_sample_sha": freeze.manifest["fixed600_sample_sha"],
            "fixed600_summary_sha": freeze.manifest["fixed600_summary_sha"],
        },
        stage1_code_sha=code_sha,
        validated_input_manifest_id=validated_input_manifest_id,
    )
    execution = {
        "schema_version": "stage1_v3_fit_execution.1",
        "runtime_s": time.perf_counter() - started,
        "initial_rss_mb": initial_rss_mb,
        "peak_rss_mb_observed": max(initial_rss_mb, _current_rss_mb()),
        "resumed": False,
    }
    atomic_write_json(destination / "fit_execution.json", execution)
    return {**manifest, "execution": execution}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read Stage1 v3 manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"manifest must be a JSON object: {path}")
    return value


def _with_partition(
    frame: pd.DataFrame,
    ref: BucketRef,
) -> pd.DataFrame:
    result = frame.copy()
    for name, value in (("date", ref.date), ("split", ref.split)):
        if name in result:
            if not result[name].astype(str).eq(str(value)).all():
                raise ContractError(f"{name} column conflicts with bucket partition")
        else:
            result.insert(0, name, value)
    return result


def _movement_context(bucket: Stage0Bucket, ref: BucketRef) -> pd.DataFrame:
    movements = getattr(bucket, "turn_movements", pd.DataFrame()).copy()
    if movements.empty:
        return pd.DataFrame(
            columns=[
                "split",
                "date",
                "order_id",
                "movement_sequence",
                "from_edge_uid",
                "observed_from_directed_edge_uid",
                "via_node",
                "to_edge_uid",
                "observed_to_directed_edge_uid",
                "movement_direction_mapping_available",
                "movement_lineage_only",
                "movement_source",
                "movement_quality",
                "iis_available",
                "iis_unavailable_reason",
            ]
        )
    keep = [
        "order_id",
        "movement_sequence",
        "from_edge_uid",
        "via_node",
        "to_edge_uid",
        "movement_source",
        "movement_quality",
    ]
    missing = sorted(set(keep) - set(movements.columns))
    if missing:
        raise ContractError(f"turn_movements missing context columns: {missing}")
    result = movements[keep].copy()
    result = derive_movement_direction_context(
        result,
        bucket.route_parts,
    )
    result["iis_available"] = False
    result["iis_unavailable_reason"] = (
        "STAGE0_V6_MOVEMENT_DYNAMIC_EVIDENCE_NOT_AVAILABLE"
    )
    return _with_partition(result, ref)


def _route_sequence_context(
    bucket: Stage0Bucket,
    ref: BucketRef,
    models: Stage1V3Models,
) -> pd.DataFrame:
    columns = [
        "order_id",
        "route_sequence",
        "canonical_edge_uid",
        "observed_directed_edge_uid",
        "observed_from_node",
        "observed_to_node",
        "observed_direction",
        "canonical_mapping_available",
        "route_lineage_status",
        "sequence_feature_mask",
        "synthetic_reverse_edge",
        "osm_direction_disagreement",
        "mapping_status",
        "osm_oneway",
        "length_m",
        "canonical_length_m",
        "canonical_highway",
        "road_class",
        "bridge",
        "tunnel",
    ]
    result = bucket.route_parts[columns].rename(
        columns={"length_m": "route_part_length_m"}
    )
    train_edges = set(
        models.support.edge_catalog["observed_directed_edge_uid"].astype(str)
    )
    mapped = result["observed_directed_edge_uid"].notna()
    result["directed_edge_model_scope"] = "unmapped"
    result.loc[
        mapped
        & result["observed_directed_edge_uid"].astype(str).isin(train_edges),
        "directed_edge_model_scope",
    ] = "train_seen"
    result.loc[
        mapped
        & ~result["observed_directed_edge_uid"].astype(str).isin(train_edges),
        "directed_edge_model_scope",
    ] = "evaluation_unseen"
    return _with_partition(result, ref)


def _output_bucket_target(output_root: Path, ref: BucketRef) -> Path:
    return (
        output_root
        / f"split={ref.split}"
        / f"date={ref.date}"
        / f"bucket={ref.bucket:05d}"
    )


def _validate_output_frames(
    products: dict[str, pd.DataFrame],
    ref: BucketRef,
) -> None:
    if set(products) != set(OUTPUT_PRODUCTS):
        raise ContractError("Stage1 v3 output product set is incomplete")
    for product in OUTPUT_PRODUCTS:
        frame = products[product]
        missing = OUTPUT_REQUIRED_COLUMNS[product] - set(frame.columns)
        if missing:
            raise ContractError(
                f"{product} output is missing columns: {sorted(missing)}"
            )
        unexpected = set(frame.columns) - OUTPUT_REQUIRED_COLUMNS[product]
        if unexpected:
            raise ContractError(
                f"{product} output has undeclared columns: {sorted(unexpected)}"
            )
        keys = list(OUTPUT_PRIMARY_KEYS[product])
        if frame[keys].isna().any().any():
            raise ContractError(f"{product} output has null primary-key values")
        if frame.duplicated(keys, keep=False).any():
            raise ContractError(f"{product} output has duplicate primary keys")
        for name, value in (("split", ref.split), ("date", ref.date)):
            if len(frame) and not frame[name].astype(str).eq(str(value)).all():
                raise ContractError(
                    f"{product}.{name} conflicts with its partition"
                )


def _resume_matches(
    target: Path,
    *,
    ref: BucketRef,
    input_identity: dict[str, Any],
    model_id: str,
    config_sha: str,
    stage1_code_sha: str | None = None,
    stage0_freeze_manifest_sha: str,
    stage0_release: dict[str, Any],
) -> bool:
    if not target.exists():
        return False
    manifest_path = target / "manifest.json"
    if not manifest_path.is_file():
        raise ContractError(f"incomplete Stage1 v3 bucket exists: {target}")
    manifest = _read_json(manifest_path)
    expected = {
        "schema_version": OUTPUT_BUCKET_SCHEMA_VERSION,
        "label_schema_version": "stage1_label_schema_v3",
        "scientific_status": "NOT_VALIDATED",
        "split": ref.split,
        "date": ref.date,
        "bucket": ref.bucket,
        "input_bucket_sha": input_identity["bucket_sha"],
        "model_id": model_id,
        "config_sha": config_sha,
        "stage1_code_sha": stage1_code_sha,
        "stage0_freeze_manifest_sha": stage0_freeze_manifest_sha,
        "stage0_release": stage0_release,
    }
    if manifest.get("engineering_status") != "PASS" or any(
        manifest.get(name) != value for name, value in expected.items()
    ):
        raise ContractError(f"Stage1 v3 resume identity mismatch: {target}")
    if manifest.get("input_file_hashes") != input_identity.get("files"):
        raise ContractError(f"Stage1 v3 resume input file identity mismatch: {target}")
    if manifest.get("input_schema_hashes") != input_identity.get("schemas"):
        raise ContractError(f"Stage1 v3 resume input schema identity mismatch: {target}")
    hashes = manifest.get("output_file_hashes")
    schemas = manifest.get("output_schema_hashes")
    counts = manifest.get("product_row_counts")
    expected_products = set(OUTPUT_PRODUCTS)
    for name, value in (
        ("output_file_hashes", hashes),
        ("output_schema_hashes", schemas),
        ("product_row_counts", counts),
    ):
        if not isinstance(value, dict) or set(value) != expected_products:
            raise ContractError(
                f"Stage1 v3 bucket manifest has invalid {name}: {target}"
            )
    actual_parquet = {path.stem for path in target.glob("*.parquet")}
    if actual_parquet != expected_products:
        raise ContractError(
            f"Stage1 v3 output product set mismatch: {target}"
        )
    for product in OUTPUT_PRODUCTS:
        path = target / f"{product}.parquet"
        if not path.is_file() or sha256_file(path) != hashes.get(product):
            raise ContractError(f"Stage1 v3 output hash mismatch: {path}")
        if parquet_schema_sha256(path) != schemas.get(product):
            raise ContractError(f"Stage1 v3 output schema mismatch: {path}")
        if parquet_row_count(path) != counts.get(product):
            raise ContractError(f"Stage1 v3 output row count mismatch: {path}")
        columns = set(parquet_column_names(path))
        missing = OUTPUT_REQUIRED_COLUMNS[product] - columns
        unexpected = columns - OUTPUT_REQUIRED_COLUMNS[product]
        if missing or unexpected:
            raise ContractError(
                f"Stage1 v3 output columns differ from the contract in {path}: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
    return True


def _transform_bucket(
    ref: BucketRef,
    bucket: Stage0Bucket,
    models: Stage1V3Models,
    config: "Stage1V3Config",
) -> dict[str, pd.DataFrame]:
    interval_labels = build_interval_labels(
        bucket.link_interval_observations,
        bucket.link_traversals,
        bucket.route_parts,
        config,
    )
    traversal_labels = build_traversal_primitives(
        bucket.link_interval_observations,
        bucket.link_traversals,
        bucket.route_parts,
        config,
    )
    traversal_labels = apply_reference_labels(
        traversal_labels,
        models.reference,
        config,
        reference_fit_manifest_id=str(models.manifest["source_manifest_id"]),
        reference_model_id=models.model_id,
        leave_one_out=ref.split == "train",
    )
    traversal_labels = apply_percentile_labels(
        traversal_labels,
        {"lcs": models.lcs, "rts": models.rts},
        config,
    )
    traversal_labels = apply_directed_support(
        traversal_labels,
        models.support,
        config,
    )
    interval_labels = _with_partition(interval_labels, ref)
    traversal_labels = _with_partition(traversal_labels, ref)
    order_labels, order_quality = aggregate_order_labels(
        traversal_labels,
        bucket.route_parts,
        bucket.link_traversals,
        bucket.order_base,
        config,
    )
    return {
        "interval_labels": interval_labels,
        "traversal_labels": traversal_labels,
        "route_sequence_context": _route_sequence_context(
            bucket, ref, models
        ),
        "movement_context": _movement_context(bucket, ref),
        "order_labels": order_labels,
        "order_label_quality": order_quality,
    }


def transform_stage1_v3(
    input_root: str | Path,
    model_root: str | Path,
    output_root: str | Path,
    stage0_freeze_manifest: str | Path,
    config: "Stage1V3Config",
    *,
    stage1_code_sha: str | None = None,
    resume: bool = True,
    allow_review_candidate: bool = False,
) -> dict[str, Any]:
    """Apply immutable train models to every bucket with exact resume identity."""

    started = time.perf_counter()
    peak_rss_mb = _current_rss_mb()
    _execution_guard(config, allow_review_candidate=allow_review_candidate)
    code_sha = _verified_code_identity(stage1_code_sha)
    freeze = load_stage0_freeze_manifest(stage0_freeze_manifest, config)
    models = load_model_bundle(model_root, config)
    if models.manifest.get("stage1_code_sha") != code_sha:
        raise ContractError("transform code SHA differs from fitted model code SHA")
    upstream_value = models.manifest.get("upstream_identity")
    if not isinstance(upstream_value, dict):
        raise ContractError("model manifest is missing its Stage0 upstream identity")
    expected_upstream = {
        str(key): str(value) for key, value in upstream_value.items()
    }
    if models.manifest.get("stage0_freeze_identity", {}).get(
        "manifest_sha"
    ) != freeze.manifest_sha:
        raise ContractError("model was fitted against a different Stage0 freeze")
    freeze.validate_bucket_identity(expected_upstream)

    refs = list(iter_stage0_buckets(input_root, config))
    _require_expected_dates(
        refs,
        {
            "train": config.train_dates,
            "validation": config.validation_dates,
            "test": (config.test_date,),
        },
    )
    if _single_upstream_identity(refs) != expected_upstream:
        raise ContractError("transform input mixes a different Stage0 identity")
    transform_identities = _input_identities(refs)
    _require_uniform_input_schemas(transform_identities)
    fitted_identities = models.manifest.get("input_bucket_identities")
    if (
        not isinstance(fitted_identities, dict)
        or transform_identities != fitted_identities
    ):
        raise ContractError(
            "transform inputs differ byte-for-byte from the frozen model inputs"
        )

    root = Path(output_root)
    transformed = skipped = 0
    row_counts = {product: 0 for product in OUTPUT_PRODUCTS}
    for ref in refs:
        identity = transform_identities[_bucket_key(ref)]
        target = _output_bucket_target(root, ref)
        if target.exists():
            if not resume:
                raise ContractError(f"Stage1 v3 output bucket already exists: {target}")
            if _resume_matches(
                target,
                ref=ref,
                input_identity=identity,
                model_id=models.model_id,
                config_sha=config.digest,
                stage1_code_sha=code_sha,
                stage0_freeze_manifest_sha=freeze.manifest_sha,
                stage0_release=models.manifest["stage0_release"],
            ):
                skipped += 1
                prior = _read_json(target / "manifest.json")
                for product, count in prior["product_row_counts"].items():
                    row_counts[product] += int(count)
                peak_rss_mb = max(peak_rss_mb, _current_rss_mb())
                continue

        bucket = load_stage0_bucket(ref, config)
        products = _transform_bucket(ref, bucket, models, config)
        _validate_output_frames(products, ref)
        with atomic_output_directory(target) as temporary:
            schemas: dict[str, str] = {}
            files: dict[str, str] = {}
            counts: dict[str, int] = {}
            for product in OUTPUT_PRODUCTS:
                frame = products[product]
                path = temporary / f"{product}.parquet"
                atomic_write_parquet(frame, path)
                schemas[product] = parquet_schema_sha256(path)
                files[product] = sha256_file(path)
                counts[product] = int(len(frame))
                row_counts[product] += int(len(frame))
            manifest = {
                "schema_version": OUTPUT_BUCKET_SCHEMA_VERSION,
                "label_schema_version": config.schema_version,
                "engineering_status": "PASS",
                "scientific_status": "NOT_VALIDATED",
                "split": ref.split,
                "date": ref.date,
                "bucket": ref.bucket,
                "input_bucket_sha": identity["bucket_sha"],
                "input_file_hashes": identity["files"],
                "input_schema_hashes": identity["schemas"],
                "model_id": models.model_id,
                "stage0_freeze_manifest_sha": models.manifest[
                    "stage0_freeze_identity"
                ]["manifest_sha"],
                "stage0_release": models.manifest["stage0_release"],
                "config_sha": config.digest,
                "stage1_code_sha": code_sha,
                "product_row_counts": counts,
                "output_schema_hashes": schemas,
                "output_file_hashes": files,
            }
            atomic_write_json(temporary / "manifest.json", manifest)
        transformed += 1
        peak_rss_mb = max(peak_rss_mb, _current_rss_mb())

    summary = {
        "schema_version": OUTPUT_SUMMARY_SCHEMA_VERSION,
        "engineering_status": "PASS",
        "scientific_status": "NOT_VALIDATED",
        "model_id": models.model_id,
        "source_manifest_id": models.manifest["source_manifest_id"],
        "stage0_freeze_manifest_sha": models.manifest[
            "stage0_freeze_identity"
        ]["manifest_sha"],
        "stage0_release": models.manifest["stage0_release"],
        "config_sha": config.digest,
        "stage1_code_sha": code_sha,
        "bucket_count": len(refs),
        "transformed_bucket_count": transformed,
        "resumed_bucket_count": skipped,
        "product_row_counts": row_counts,
        "runtime_s": time.perf_counter() - started,
        "peak_rss_mb": peak_rss_mb,
        "dates": {
            "train": list(config.train_dates),
            "validation": list(config.validation_dates),
            "test": [config.test_date],
        },
    }
    atomic_write_json(root / "stage1_v3_summary.json", summary)
    return summary
