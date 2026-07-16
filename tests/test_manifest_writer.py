import pytest

from scripts.write_artifact_manifest import split_dates, validate_registration_status


def test_split_dates_ignores_empty_items():
    assert split_dates("20161020,,20161023") == ["20161020", "20161023"]


def test_canonical_registration_requires_pass():
    with pytest.raises(ValueError, match="requires audit status PASS"):
        validate_registration_status("canonical", "DIAGNOSTIC_PASS")
    validate_registration_status("exploratory", "DIAGNOSTIC_PASS")
