"""One fixed-protocol check; no native simulation, model or production reads."""
import json
from pathlib import Path
import shutil

import pytest

from stage4.analysis.scheme_a_two_scale_check import CONFIG, load_protocol


def test_two_scale_protocol_keeps_frozen_conditions_and_one_short_window(tmp_path):
    root = Path(__file__).resolve().parents[2]
    base = Path("stage4/config/scheme_a_city_v1.json")
    (tmp_path / base).parent.mkdir(parents=True)
    shutil.copyfile(root / base, tmp_path / base)
    original = json.loads((root / CONFIG).read_text(encoding="utf-8"))
    (tmp_path / CONFIG).write_text(json.dumps(original), encoding="utf-8")
    cfg = load_protocol(tmp_path)
    assert cfg["requested_q_a"] == .1 and cfg["passenger_acceptance_rate"] == .7
    assert cfg["patience_s"] == 300 and cfg["current_top_k"] == 20
    assert cfg["groups"] == ["TWO_SCALE_NATIVE_CHECK"] and cfg["full_day"] is False
    assert (cfg["window_start_s"], cfg["measurement_end_s"], cfg["admission_end_s"],
            cfg["physical_drain_limit_s"]) == (29700, 30600, 30900, 37800)
    assert cfg["coarse_reference_day_end_s"] == 86434
    assert cfg["automatic_retry"] is False
    (tmp_path / CONFIG).write_text(json.dumps(dict(original, patience_s=600)), encoding="utf-8")
    with pytest.raises(ValueError, match="scientific conditions"):
        load_protocol(tmp_path)
    (tmp_path / CONFIG).write_text(json.dumps(dict(original, full_day=True)), encoding="utf-8")
    with pytest.raises(ValueError, match="fixed 15-minute"):
        load_protocol(tmp_path)
