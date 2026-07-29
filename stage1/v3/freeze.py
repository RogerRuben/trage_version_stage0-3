"""Validation of the immutable Stage 0 to Stage 1 hand-off identity."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

from .io import sha256_file
from .schema import ContractError

if TYPE_CHECKING:
    from .config import Stage1V3Config


@dataclass(frozen=True)
class Stage0FreezeIdentity:
    path: Path
    manifest_sha: str
    manifest: Mapping[str, Any]

    @property
    def code_identity(self) -> str:
        return str(self.manifest["code_identities"][0])

    @property
    def config_sha(self) -> str:
        return str(self.manifest["config_sha"])

    @property
    def tiles_sha(self) -> str:
        return str(self.manifest["valhalla_tiles_sha"])

    def validate_bucket_identity(self, identity: dict[str, str]) -> None:
        expected = {
            "code_sha": self.code_identity,
            "config_sha": self.config_sha,
            "tiles_sha": self.tiles_sha,
        }
        if identity != expected:
            raise ContractError(
                "Stage0 bucket identity differs from the FROZEN hand-off: "
                f"expected={expected}, actual={identity}"
            )


def _nonempty_string(manifest: dict[str, Any], name: str) -> str:
    value = manifest.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"Stage0 freeze manifest has invalid {name!r}")
    return value


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def load_stage0_freeze_manifest(
    path: str | Path,
    config: "Stage1V3Config",
) -> Stage0FreezeIdentity:
    source = Path(path).resolve()
    try:
        manifest = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(
            f"cannot read Stage0 freeze manifest {source}: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise ContractError("Stage0 freeze manifest must be a JSON object")
    if manifest.get("schema_version") != "stage0_v6_freeze_manifest.1":
        raise ContractError("unsupported Stage0 freeze manifest schema")
    if manifest.get("freeze_status") != "FROZEN":
        raise ContractError("Stage1 v3 requires freeze_status=FROZEN")
    if manifest.get("product_schema_version") != "stage1_input_v1":
        raise ContractError(
            "Stage0 freeze manifest does not describe stage1_input_v1"
        )

    for name in (
        "git_commit_sha",
        "config_sha",
        "pbf_sha",
        "valhalla_tiles_sha",
        "fixed600_sample_sha",
        "fixed600_summary_sha",
    ):
        _nonempty_string(manifest, name)
    identities = manifest.get("code_identities")
    if (
        not isinstance(identities, list)
        or len(identities) != 1
        or not isinstance(identities[0], str)
        or not identities[0].strip()
    ):
        raise ContractError(
            "Stage0 freeze manifest must contain one code identity"
        )
    release = config.section("stage0_release")
    content_hash = str(release["stage0_source_content_hash"])
    if f"content.{content_hash}" not in identities[0]:
        raise ContractError(
            "Stage0 freeze code identity does not match the final tag content hash"
        )
    if manifest.get("working_tree_clean") is not True:
        # Production recorded a dirty commit SHA, but its executable source was
        # content-addressed and independently confirmed equal to the final tag.
        # The exact tag/commit/content tuple is frozen in Stage1 configuration.
        if not content_hash:
            raise ContractError(
                "dirty Stage0 freeze requires a verified source content hash"
            )
    if tuple(manifest.get("train_dates", [])) != config.train_dates:
        raise ContractError("Stage0 frozen train dates differ from Stage1 v3")
    if tuple(manifest.get("validation_dates", [])) != config.validation_dates:
        raise ContractError(
            "Stage0 frozen validation dates differ from Stage1 v3"
        )
    if str(manifest.get("test_date")) != config.test_date:
        raise ContractError("Stage0 frozen test date differs from Stage1 v3")
    coverage = manifest.get("coverage")
    if not isinstance(coverage, dict) or coverage.get("status") != "PASS":
        raise ContractError("Stage0 frozen coverage report is not PASS")
    verification = manifest.get("verification")
    if (
        not isinstance(verification, dict)
        or verification.get("status") != "PASS"
    ):
        raise ContractError("Stage0 frozen input verification is not PASS")

    return Stage0FreezeIdentity(
        path=source,
        manifest_sha=sha256_file(source),
        manifest=_deep_freeze(manifest),
    )
