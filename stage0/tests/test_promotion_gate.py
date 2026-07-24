from scripts.write_artifact_manifest import validate_registration_status


def test_canonical_manifest_requires_pass():
    try:
        validate_registration_status("canonical", "DIAGNOSTIC_PASS")
    except ValueError:
        pass
    else:
        raise AssertionError("canonical registration accepted a diagnostic audit")


def test_exploratory_manifest_can_record_diagnostic_pass():
    validate_registration_status("exploratory", "DIAGNOSTIC_PASS")
