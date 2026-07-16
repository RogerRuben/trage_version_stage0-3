from scripts.write_artifact_manifest import split_dates


def test_split_dates_ignores_empty_items():
    assert split_dates("20161020,,20161023") == ["20161020", "20161023"]
