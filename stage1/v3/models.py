"""Immutable Stage 1 v3 reference/CDF model bundle."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .io import (
    atomic_output_directory,
    atomic_write_json,
    atomic_write_parquet,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from .references import COHORT_LEVELS, SparseCohortHistograms
from .schema import ContractError
from .support import DirectedSupportModel

if TYPE_CHECKING:
    from .config import Stage1V3Config


MODEL_FILES = {
    "reference": "reference_histograms.parquet",
    "lcs": "lcs_histograms.parquet",
    "rts": "rts_histograms.parquet",
    "directed_edge_catalog": "directed_edge_catalog.parquet",
    "support_counts": "support_counts.parquet",
    "metadata": "histogram_metadata.json",
}


@dataclass(frozen=True)
class Stage1V3Models:
    reference: SparseCohortHistograms
    lcs: SparseCohortHistograms
    rts: SparseCohortHistograms
    support: DirectedSupportModel
    manifest: dict[str, Any]

    @property
    def model_id(self) -> str:
        return str(self.manifest["model_id"])


def _histogram_metadata(
    reference: SparseCohortHistograms,
    lcs: SparseCohortHistograms,
    rts: SparseCohortHistograms,
) -> dict[str, Any]:
    return {
        name: {
            "edges": model.edges.tolist(),
            "invalid_count": int(model.invalid_count),
            "underflow_count": int(model.underflow_count),
            "overflow_count": int(model.overflow_count),
        }
        for name, model in {
            "reference": reference,
            "lcs": lcs,
            "rts": rts,
        }.items()
    }


def write_model_bundle(
    target: str | Path,
    *,
    reference: SparseCohortHistograms,
    lcs: SparseCohortHistograms,
    rts: SparseCohortHistograms,
    support: DirectedSupportModel,
    config: "Stage1V3Config",
    source_manifest_id: str,
    input_bucket_identities: dict[str, Any],
    upstream_identity: dict[str, str],
    stage0_freeze_identity: dict[str, Any],
    stage1_code_sha: str,
) -> dict[str, Any]:
    """Atomically publish a content-addressed, train-only model bundle."""

    destination = Path(target)
    with atomic_output_directory(destination) as temporary:
        atomic_write_parquet(
            reference.to_frame(), temporary / MODEL_FILES["reference"]
        )
        atomic_write_parquet(lcs.to_frame(), temporary / MODEL_FILES["lcs"])
        atomic_write_parquet(rts.to_frame(), temporary / MODEL_FILES["rts"])
        atomic_write_parquet(
            support.edge_catalog,
            temporary / MODEL_FILES["directed_edge_catalog"],
        )
        atomic_write_parquet(
            support.counts,
            temporary / MODEL_FILES["support_counts"],
        )
        metadata = _histogram_metadata(reference, lcs, rts)
        atomic_write_json(temporary / MODEL_FILES["metadata"], metadata)
        file_hashes = {
            name: sha256_file(temporary / filename)
            for name, filename in MODEL_FILES.items()
        }
        manifest_core = {
            "schema_version": "stage1_v3_models.2",
            "label_schema_version": config.schema_version,
            "engineering_status": "PASS",
            "scientific_status": "NOT_VALIDATED",
            "fit_dates": list(config.reference_fit_dates),
            "config_sha": config.digest,
            "stage1_code_sha": str(stage1_code_sha),
            "source_manifest_id": str(source_manifest_id),
            "source_schema_sha": sha256_bytes(
                canonical_json_bytes(
                    {
                        key: value["schemas"]
                        for key, value in sorted(
                            input_bucket_identities.items()
                        )
                    }
                )
            ),
            "input_bucket_identities": input_bucket_identities,
            "upstream_identity": dict(sorted(upstream_identity.items())),
            "stage0_freeze_identity": stage0_freeze_identity,
            "stage0_release": config.section("stage0_release"),
            "directed_edge_identity": "observed_directed_edge_uid",
            "support_fit_scope": "train_only",
            "support_definition": config.section("support"),
            "directed_edge_count": int(len(support.edge_catalog)),
            "synthetic_reverse_graph_edge_count": int(
                support.edge_catalog["synthetic_reverse_edge"].sum()
            ),
            "cohort_fallback": [
                "edge_time_weekday",
                "edge_peak",
                "edge",
                "highway_time_weekday",
                "highway",
                "global",
            ],
            "cohort_definition": config.section("cohort_reference"),
            "timezone": str(config.section("time")["timezone"]),
            "global_support_count": {
                "reference": reference.support("global", "global"),
                "lcs": lcs.support("global", "global"),
                "rts": rts.support("global", "global"),
            },
            "maximum_histogram_bin_width": {
                "reference": float(np.diff(reference.edges).max()),
                "lcs": float(np.diff(lcs.edges).max()),
                "rts": float(np.diff(rts.edges).max()),
            },
            "train_reference_application": "leave_one_out",
            "validation_test_reference_application": "full_train_frozen",
            "raw_cdf_application": "full_train_empirical_self_rank_for_train",
            "files": file_hashes,
            "known_limitations": [
                "LCS and RTS thresholds remain review candidates",
                "IIS and PMIS are unavailable",
                "GNS requires a separately frozen static edge extension",
                "no cross-dimension core composite is emitted",
            ],
        }
        model_id = sha256_bytes(canonical_json_bytes(manifest_core))
        manifest = {**manifest_core, "model_id": model_id}
        atomic_write_json(temporary / "model_manifest.json", manifest)
    return manifest


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read Stage1 v3 model metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"Stage1 v3 model metadata must be an object: {path}")
    return value


def load_model_bundle(
    root: str | Path,
    config: "Stage1V3Config",
) -> Stage1V3Models:
    source = Path(root)
    manifest = _read_json(source / "model_manifest.json")
    if manifest.get("schema_version") != "stage1_v3_models.2":
        raise ContractError("unsupported Stage1 v3 model schema")
    if manifest.get("label_schema_version") != config.schema_version:
        raise ContractError("model label schema differs from the requested schema")
    if manifest.get("engineering_status") != "PASS":
        raise ContractError("Stage1 v3 model engineering status is not PASS")
    if manifest.get("scientific_status") != "NOT_VALIDATED":
        raise ContractError("Stage1 v3 model scientific status is invalid")
    if manifest.get("config_sha") != config.digest:
        raise ContractError("model config SHA differs from the requested config")
    if tuple(manifest.get("fit_dates", [])) != config.reference_fit_dates:
        raise ContractError("model fit dates differ from frozen train dates")
    for name in ("model_id", "stage1_code_sha", "source_manifest_id"):
        if not isinstance(manifest.get(name), str) or not manifest[name].strip():
            raise ContractError(f"model manifest has invalid {name}")
    if manifest.get("cohort_fallback") != list(COHORT_LEVELS):
        raise ContractError("model cohort fallback contract is invalid")
    if manifest.get("train_reference_application") != "leave_one_out":
        raise ContractError("train RTS reference application must be leave-one-out")
    if (
        manifest.get("validation_test_reference_application")
        != "full_train_frozen"
    ):
        raise ContractError("validation/test reference application is invalid")
    if (
        manifest.get("raw_cdf_application")
        != "full_train_empirical_self_rank_for_train"
    ):
        raise ContractError("raw CDF application policy is invalid")
    if manifest.get("timezone") != config.section("time")["timezone"]:
        raise ContractError("model timezone differs from config")
    if manifest.get("cohort_definition") != config.section("cohort_reference"):
        raise ContractError("model cohort definition differs from config")
    if manifest.get("stage0_release") != config.section("stage0_release"):
        raise ContractError("model Stage0 release identity differs from config")
    if manifest.get("support_fit_scope") != "train_only":
        raise ContractError("support counts must be fitted from Train only")
    if manifest.get("support_definition") != config.section("support"):
        raise ContractError("model support definition differs from config")
    input_identities = manifest.get("input_bucket_identities")
    if not isinstance(input_identities, dict) or not input_identities:
        raise ContractError("model manifest lacks input bucket identities")
    for bucket_key, identity in input_identities.items():
        if not isinstance(bucket_key, str) or not bucket_key.strip():
            raise ContractError("model input identity has an invalid bucket key")
        if not isinstance(identity, dict):
            raise ContractError(
                f"model input identity is not a mapping: {bucket_key}"
            )
        schemas = identity.get("schemas")
        if not isinstance(schemas, dict) or not schemas:
            raise ContractError(
                f"model input identity lacks schema hashes: {bucket_key}"
            )
    if manifest.get("source_manifest_id") != sha256_bytes(
        canonical_json_bytes(input_identities)
    ):
        raise ContractError("model source manifest identity is inconsistent")
    expected_source_schema_sha = sha256_bytes(
        canonical_json_bytes(
            {
                key: value["schemas"]
                for key, value in sorted(input_identities.items())
            }
        )
    )
    if manifest.get("source_schema_sha") != expected_source_schema_sha:
        raise ContractError("model source schema identity is inconsistent")
    for name in ("global_support_count", "maximum_histogram_bin_width"):
        values = manifest.get(name)
        if not isinstance(values, dict) or set(values) != {
            "reference",
            "lcs",
            "rts",
        }:
            raise ContractError(f"model manifest has invalid {name}")
    upstream = manifest.get("upstream_identity")
    if not isinstance(upstream, dict) or set(upstream) != {
        "code_sha",
        "config_sha",
        "tiles_sha",
    }:
        raise ContractError("model manifest has invalid Stage0 upstream identity")
    freeze_identity = manifest.get("stage0_freeze_identity")
    required_freeze = {
        "manifest_sha",
        "git_commit_sha",
        "config_sha",
        "pbf_sha",
        "valhalla_tiles_sha",
        "fixed600_sample_sha",
        "fixed600_summary_sha",
    }
    if (
        not isinstance(freeze_identity, dict)
        or not required_freeze.issubset(freeze_identity)
        or any(
            not isinstance(freeze_identity[name], str)
            or not freeze_identity[name].strip()
            for name in required_freeze
        )
    ):
        raise ContractError("model manifest has invalid Stage0 freeze identity")
    if (
        str(upstream["config_sha"]) != str(freeze_identity["config_sha"])
        or str(upstream["tiles_sha"])
        != str(freeze_identity["valhalla_tiles_sha"])
    ):
        raise ContractError("model Stage0 upstream and freeze identities differ")
    expected_files = manifest.get("files")
    if not isinstance(expected_files, dict) or set(expected_files) != set(
        MODEL_FILES
    ):
        raise ContractError("model manifest is missing file hashes")
    for name, filename in MODEL_FILES.items():
        path = source / filename
        if not path.is_file():
            raise ContractError(f"model file is missing: {path}")
        if sha256_file(path) != expected_files.get(name):
            raise ContractError(f"model file hash mismatch: {path}")

    metadata = _read_json(source / MODEL_FILES["metadata"])

    def read_histogram(name: str) -> SparseCohortHistograms:
        details = metadata.get(name)
        if not isinstance(details, dict):
            raise ContractError(f"missing histogram metadata for {name}")
        try:
            edges = np.asarray(details.get("edges"), dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"invalid histogram edges for {name}") from exc
        counts: dict[str, int] = {}
        for field in ("invalid_count", "underflow_count", "overflow_count"):
            value = details.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ContractError(
                    f"invalid histogram metadata {name}.{field}"
                )
            counts[field] = value
        frame = pd.read_parquet(source / MODEL_FILES[name])
        result = SparseCohortHistograms.from_frame(
            edges,
            frame,
            invalid_count=counts["invalid_count"],
            underflow_count=counts["underflow_count"],
            overflow_count=counts["overflow_count"],
        )
        if name == "reference":
            settings = config.section("reference")
            expected_edges = np.geomspace(
                float(settings["histogram_min_sec_per_m"]),
                float(settings["histogram_max_sec_per_m"]),
                int(settings["histogram_bins"]) + 1,
            )
        else:
            expected_edges = np.linspace(
                0.0,
                1.0,
                int(config.section("normalization")["raw_bins"]) + 1,
            )
        if not np.array_equal(result.edges, expected_edges):
            raise ContractError(f"{name} histogram edges differ from config")
        return result

    manifest_core = {key: value for key, value in manifest.items() if key != "model_id"}
    expected_model_id = sha256_bytes(canonical_json_bytes(manifest_core))
    if manifest.get("model_id") != expected_model_id:
        raise ContractError("Stage1 v3 model manifest identity mismatch")
    loaded = {
        "reference": read_histogram("reference"),
        "lcs": read_histogram("lcs"),
        "rts": read_histogram("rts"),
    }
    support = DirectedSupportModel(
        edge_catalog=pd.read_parquet(
            source / MODEL_FILES["directed_edge_catalog"]
        ),
        counts=pd.read_parquet(source / MODEL_FILES["support_counts"]),
    )
    if (
        int(manifest.get("directed_edge_count", -1))
        != len(support.edge_catalog)
        or int(manifest.get("synthetic_reverse_graph_edge_count", -1))
        != int(support.edge_catalog["synthetic_reverse_edge"].sum())
    ):
        raise ContractError("directed edge catalog metadata mismatch")
    for name, model in loaded.items():
        support_value = manifest["global_support_count"].get(name)
        if (
            isinstance(support_value, bool)
            or not isinstance(support_value, int)
            or support_value != model.support("global", "global")
        ):
            raise ContractError(f"{name} global support metadata mismatch")
        width_value = manifest["maximum_histogram_bin_width"].get(name)
        if (
            isinstance(width_value, bool)
            or not isinstance(width_value, (int, float))
            or not np.isfinite(float(width_value))
            or not np.isclose(
                float(width_value),
                float(np.diff(model.edges).max()),
                atol=0.0,
                rtol=1e-15,
            )
        ):
            raise ContractError(f"{name} histogram width metadata mismatch")
    return Stage1V3Models(
        reference=loaded["reference"],
        lcs=loaded["lcs"],
        rts=loaded["rts"],
        support=support,
        manifest=manifest,
    )
