import json
from pathlib import Path

import pytest

from canonical_pipeline.manifest import ManifestError, load_manifest, require_canonical_input
from canonical_pipeline.preflight import validate_config, validate_field_registry


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_manifest_is_valid_but_rejected_as_canonical_input():
    manifest = load_manifest(
        ROOT / "artifacts/exploratory/legacy_stage0_4_2026_07.manifest.json",
        ROOT / "config/artifact_manifest.schema.json",
        ROOT,
    )
    assert manifest.status == "exploratory"
    with pytest.raises(ManifestError):
        require_canonical_input(manifest)


def test_field_registry_has_no_undefined_availability():
    assert validate_field_registry(
        ROOT / "docs/pipeline_contract/field_availability_registry.csv"
    ) == []


def test_canonical_config_contract_preflight_passes():
    import yaml

    config = yaml.safe_load((ROOT / "config/pipeline_canonical.yaml").read_text(encoding="utf-8"))
    assert validate_config(config, ROOT) == []

