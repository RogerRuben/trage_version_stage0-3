"""Fixed full-day scope and exact completed-group reuse, without native replay."""
import json
from pathlib import Path
import shutil

import pytest

from stage4.analysis.scheme_a_two_scale_check import load_protocol as load_short_protocol
from stage4.analysis.scheme_a_two_scale_experiment import (
    CONFIG, GROUPS, PRODUCTS, COMPLETE, _group_binding,
    completed_group, group_configuration, load_protocol,
)


def test_three_group_protocol_and_completed_only_resume(tmp_path):
    root = Path(__file__).resolve().parents[2]
    for path in (Path("stage4/config/scheme_a_city_v1.json"),
                 Path("stage4/config/scheme_a_two_scale_v2.json"), CONFIG):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / path, tmp_path / path)
    short = load_short_protocol(tmp_path)
    cfg = load_protocol(tmp_path)
    assert cfg["full_day"] is True and cfg["window_start_s"] == 0
    assert (cfg["measurement_end_s"], cfg["admission_end_s"], cfg["physical_drain_limit_s"]) == (86434, 86760, 100800)
    assert cfg["scenario_timeout_s"] == 10800 and cfg["progress_interval_s"] == 300
    assert short["window_start_s"] == 29700 and short["scenario_timeout_s"] == 900
    for key in ("requested_q_a", "profile_id", "passenger_acceptance_rate", "supply_seed",
                "forecast_seed", "patience_s", "planning_horizon_s", "current_top_k",
                "coarse_value_wall_limit_s", "coarse_value_max_pricing_rounds"):
        assert cfg[key] == short[key]
    choices = [group_configuration(cfg, group) for group in GROUPS]
    assert [(c["layout_mode"], c["policy"]) for c in choices] == [
        ("hotspot", "SERVICE_PRESERVING"), ("joint", "SERVICE_PRESERVING"), ("joint", "CHAIN_DEFER")]
    assert len({c["disk_route_cache"] for c in choices}) == 3
    binding = dict(code_sha="code", inputs_sha256={"source": "input"},
        config_sha256="configuration", layout_sha256="layout")
    expected = _group_binding(binding, cfg, GROUPS[0])
    directory = tmp_path / "complete_group"
    directory.mkdir()
    for product in PRODUCTS:
        (directory / product).touch()
    summary = dict(status=COMPLETE, full_day=True, execution_binding=expected,
        code_sha=binding["code_sha"], inputs_sha256=binding["inputs_sha256"],
        layout_sha256=binding["layout_sha256"], group=GROUPS[0],
        layout_mode="hotspot", policy="SERVICE_PRESERVING")
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    assert completed_group(directory, expected) == summary
    with pytest.raises(ValueError, match="no automatic retry"):
        completed_group(directory, dict(expected, code_sha="different_code"))
    (directory / "summary.json").write_text(json.dumps(dict(summary, status="STOPPED")), encoding="utf-8")
    with pytest.raises(ValueError, match="no automatic retry"):
        completed_group(directory, expected)
    overrides = json.loads((tmp_path / CONFIG).read_text(encoding="utf-8"))
    (tmp_path / CONFIG).write_text(json.dumps(dict(overrides, coarse_value_max_pricing_rounds=7)), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot change inherited"):
        load_protocol(tmp_path)
