from __future__ import annotations

import json
from pathlib import Path

from stage2.v5.config import load_config
from stage2.v5.protocol_runner import materialize_protocol_config


def test_rolling_protocols_are_strict_and_reuse_frozen_upstream(tmp_path: Path) -> None:
    base = Path("stage2/config/stage2_v5.json")
    expected_evaluation = {
        "fold_1": ["20161022", "20161023"],
        "fold_2": ["20161024", "20161025"],
        "fold_3": ["20161026", "20161027"],
    }
    for name, evaluation in expected_evaluation.items():
        path = tmp_path / f"{name}.json"
        materialize_protocol_config(base, path, name)
        config = load_config(path)
        split = config.section("split")
        assert split["evaluation_dates"] == evaluation
        assert split["legacy_test_dates"] == []
        all_dates = [
            *split["train_dates"], *split["validation_model_dates"],
            *split["calibration_dates"], *split["evaluation_dates"],
        ]
        assert all_dates == sorted(all_dates)
        assert len(all_dates) == len(set(all_dates))


def test_legacy_protocol_is_never_named_unseen_test(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    identity = materialize_protocol_config(Path("stage2/config/stage2_v5.json"), path, "legacy")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert identity["protocol_name"] == "legacy_frozen_benchmark"
    assert payload["split"]["legacy_test_dates"] == ["20161031"]
    assert payload["split"]["evaluation_dates"] == []
