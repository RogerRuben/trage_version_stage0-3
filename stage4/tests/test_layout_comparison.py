"""Minimal prepared-position integration checks, without native replay."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from stage4.analysis import scheme_a_two_scale_experiment as experiment
from stage4.analysis.scheme_a_two_scale_check import _sha, load_frozen_layout
from stage4.dispatch.scheme_a_city_adapter import apply_av_layout
from stage4.fleetpy_adapter.research_world import SupplyPlanRow

CONFIG = Path("stage4/config/scheme_a_layout_comparison_300_600_v1.json")


def test_layout_groups_change_only_positions_under_identical_control(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    for path in ("stage4/config/scheme_a_city_v1.json", "stage4/config/scheme_a_two_scale_v2.json",
                 *(f"stage4/config/scheme_a_two_scale_city_{v}.json" for v in ("v2", "v3", "v4", "v5", "v5_common")),
                 CONFIG.as_posix()):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / path, target)
    cfg = experiment.load_protocol(tmp_path, CONFIG)
    choices = [experiment.group_configuration(cfg, group) for group in experiment.LAYOUT_RULES]
    assert [c["layout_mode"] for c in choices] == ["all_hot", "compatible_hot", "opportunity_joint"]
    assert all(c["policy"] == "SERVICE_PRESERVING" for c in choices)
    assert all((c["max_match_wait_s"], c["max_pickup_eta_s"], c["requested_q_a"],
                c["profile_id"], c["passenger_acceptance_rate"]) == (300, 600, .1, "C", .7) for c in choices)
    assert all(c["idle_movement_mode"] == "COMMON_HISTORICAL_DEFICIT"
               and c["current_empty_distance_mode"] == "PRIMARY_FACE_MM" for c in choices)
    for before, after in zip(choices, choices[1:]):
        assert {key for key in before if before[key] != after[key]} == {"group", "layout_mode", "disk_route_cache"}
    monkeypatch.setattr(experiment, "input_binding", lambda *args: {})
    monkeypatch.setattr(experiment, "output_root", lambda *args: Path("out"))
    calls = []
    monkeypatch.setattr(experiment, "_run_child", lambda root, fp, config, cfg, group, binding:
        calls.append(group) or dict(total_runner_wall_s=0, completed_orders=1))
    state = experiment.run(tmp_path, config=CONFIG)
    assert calls == state["group_order"] == list(experiment.LAYOUT_RULES)
    assert state["completed_group_count"] == 3
    with pytest.raises(ValueError, match="of this protocol"):
        experiment.run(tmp_path, config=CONFIG, group="CHAIN_LAYOUT_BASELINE")


def test_prepared_layout_loading_and_injection_preserve_all_admission_rows(tmp_path):
    for name in ("stage4/config/scheme_a_city_v1.json",
                 "stage4/input/replay_foundation/stage4_order_replay_base.parquet",
                 "routes", "remaining", "controlled", "source"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    modes = ("all_hot", "compatible_hot", "opportunity_joint")
    layouts = {mode: {"a": {"position": [108.9 + i * .01, 34.2]},
                      "b": {"position": [108.8 + i * .01, 34.3]}} for i, mode in enumerate(modes)}
    payload = dict(status="LAYOUTS_PREPARED_300_600", placements=layouts, records=[{}, {}],
        binding=dict(waiting_contract=dict(max_match_wait_s=300, max_pickup_eta_s=600),
                     inputs_sha256={"source": _sha(tmp_path / "source")}))
    (tmp_path / "layouts.json").write_text(json.dumps(payload), encoding="utf-8")
    cfg = dict(layout_product="PREPARED_300_600", source_layout="layouts.json",
        source_layout_sha256=_sha(tmp_path / "layouts.json"), expected_layout_bins=2, expected_av_slots=2,
        max_match_wait_s=300, max_pickup_eta_s=600, source_routes="routes",
        remaining_time_model="remaining", controlled_source_config="controlled")
    loaded, protected = load_frozen_layout(tmp_path, cfg)
    assert loaded == payload and "routes" in protected and "source" in protected
    original = tuple(SupplyPlanRow(vid, index, vid, kind, "C" if kind == "AV" else "HV",
        17 + index, 901 + index, 108.7, 34.1, "STOP_ADMISSION_FINISH_COMMITTED")
        for index, (vid, kind) in enumerate((("a", "AV"), ("b", "AV"), ("h", "HV"))))
    for mode in modes:
        supply = SimpleNamespace(_plan=original, _by_vehicle={}, _accounting={})
        supply.plan = lambda: supply._plan
        assert apply_av_layout(SimpleNamespace(supply=supply), loaded["placements"][mode]) == 2
        assert supply._plan[-1] == original[-1]
        for before, after in zip(original, supply._plan):
            assert replace(after, initial_lon_wgs84=before.initial_lon_wgs84,
                           initial_lat_wgs84=before.initial_lat_wgs84) == before
        with pytest.raises(ValueError, match="exact original"):
            apply_av_layout(SimpleNamespace(supply=supply), dict(layouts[mode], extra=layouts[mode]["a"]))
    (tmp_path / "source").write_bytes(b"changed")
    with pytest.raises(ValueError, match="source changed"):
        load_frozen_layout(tmp_path, cfg)
