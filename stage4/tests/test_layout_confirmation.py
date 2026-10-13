"""Six-case parameter plumbing only; no native replay, routing or inference."""
import json
from pathlib import Path
import shutil
from types import SimpleNamespace as NS

import pandas as pd
import pytest

from stage4.analysis import scheme_a_layout_prepare as preparation
from stage4.analysis import scheme_a_two_scale_experiment as experiment
from stage4.analysis.scheme_a_layout_confirmation import CONFIG, CONDITIONS, GROUPS, MODES, load_design
from stage4.analysis.scheme_a_two_scale_check import _sha, load_frozen_layout
from stage4.fleetpy_adapter import research_world as world


def fixture_root(tmp_path):
    source = Path(__file__).resolve().parents[2]
    for path in ("stage4/config/scheme_a_city_v1.json", "stage4/config/scheme_a_two_scale_v2.json",
            *(f"stage4/config/scheme_a_two_scale_city_{version}.json" for version in ("v2", "v3", "v4", "v5", "v5_common")),
            CONFIG.as_posix()):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / path, target)
    return tmp_path


def prepared_fixture(root, name):
    cfg = preparation.load_config(root, CONFIG, condition=name)
    positions = {mode: {"a": dict(position=[108.9 + index * .01, 34.2]),
        "b": dict(position=[108.8 + index * .01, 34.3])} for index, mode in enumerate(MODES)}
    payload = dict(status="LAYOUTS_PREPARED_300_600", placements=positions, records=[dict(bin_s=0)],
        type_label_sha256="fixture", supply_contract=dict(requested_q_a=cfg["requested_q_a"], supply_seed=cfg["supply_seed"]),
        binding=dict(confirmation_condition=name, waiting_contract=dict(max_match_wait_s=300, max_pickup_eta_s=600),
            inputs_sha256={CONFIG.as_posix(): _sha(root / CONFIG)}))
    path = root / cfg["source_layout"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    for relative in ("stage4/input/replay_foundation/stage4_order_replay_base.parquet", cfg["source_routes"],
            cfg["remaining_time_model"], cfg["controlled_source_config"]):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fixture source")
    return payload


def test_three_explicit_conditions_keep_controls_and_derive_new_dimensions(tmp_path):
    root = fixture_root(tmp_path)
    assert load_design(root)["planned_native_groups"] == 6
    with pytest.raises(ValueError, match="explicit condition"):
        preparation.load_config(root, CONFIG)
    for name, q, seed in CONDITIONS:
        prepared_fixture(root, name)
        cfg = experiment.load_protocol(root, CONFIG, condition=name)
        assert (cfg["requested_q_a"], cfg["supply_seed"], cfg["profile_id"], cfg["passenger_acceptance_rate"]) == (q, seed, "C", .7)
        assert (cfg["expected_av_slots"], cfg["expected_layout_bins"]) == (2, 1)  # Not the old 851/47 constants.
        assert (cfg["max_match_wait_s"], cfg["max_pickup_eta_s"], cfg["solver_time_limit_s"]) == (300, 600, 10.)
        assert cfg["idle_movement_mode"] == "COMMON_HISTORICAL_DEFICIT"
        assert cfg["current_empty_distance_mode"] == "PRIMARY_FACE_MM"
        left, right = [experiment.group_configuration(cfg, group) for group in GROUPS]
        assert {key for key in left if left[key] != right[key]} == {"group", "layout_mode", "disk_route_cache"}
        with pytest.raises(ValueError, match="not part"):
            experiment.group_configuration(cfg, "ALL_HOT_COMMON_CONTROL")


def test_pair_coordinator_plumbing_selects_exactly_six_mock_children(tmp_path, monkeypatch):
    root = fixture_root(tmp_path)
    calls = []
    monkeypatch.setattr(experiment, "input_binding", lambda *args: {})
    monkeypatch.setattr(experiment, "_run_child", lambda root, fleetpy, config, cfg, group, binding:
        calls.append((cfg["confirmation_condition"], group, cfg["supply_seed"])) or dict(total_runner_wall_s=0, completed_orders=1))
    for name, _, _ in CONDITIONS:
        prepared_fixture(root, name)
        state = experiment.run(root, config=CONFIG, condition=name)
        assert state["group_order"] == list(GROUPS) and state["completed_group_count"] == 2
        assert state["sequential_single_child"] and not state["automatic_retry"]
    assert calls == [(name, group, seed) for name, _, seed in CONDITIONS for group in GROUPS]


def test_prepared_pair_accepts_two_modes_and_rejects_different_label_seed(tmp_path):
    root = fixture_root(tmp_path)
    name = CONDITIONS[0][0]
    payload = prepared_fixture(root, name)
    cfg = experiment.load_protocol(root, CONFIG, condition=name)
    loaded, _ = load_frozen_layout(root, cfg)
    assert set(loaded["placements"]) == set(MODES)
    payload["supply_contract"]["supply_seed"] = 20260824
    source = root / cfg["source_layout"]
    source.write_text(json.dumps(payload), encoding="utf-8")
    cfg["source_layout_sha256"] = _sha(source)
    with pytest.raises(ValueError, match="supply does not match"):
        load_frozen_layout(root, cfg)
    with pytest.raises(ValueError, match="different q/label seed"):
        experiment.load_protocol(root, CONFIG, condition=name)


def test_episode_factory_uses_explicit_label_seed_without_changing_default(tmp_path, monkeypatch):
    monkeypatch.setattr(world, "BASE_CONFIG_REL", Path("base.json"))
    monkeypatch.setattr(world, "CONTROLLED_CONFIG_REL", Path("controlled.json"))
    monkeypatch.setattr(world, "ORDER_BASE_REL", Path("requests"))
    monkeypatch.setattr(world, "ROUTES_REL", Path("routes"))
    base_config = dict(source_scenario="fixture", measurement_end_s=3600, physical_drain_limit_s=7200,
        patience_s=300, version="fixture")
    (tmp_path / "base.json").write_text(json.dumps(base_config), encoding="utf-8")
    (tmp_path / "controlled.json").write_text(json.dumps(dict(supply_seed=20260824, version="fixture")), encoding="utf-8")
    scenario = tmp_path / "stage4/output/final_experiments/fixture/scenario_config.json"
    scenario.parent.mkdir(parents=True)
    scenario.write_text(json.dumps(dict(runtime_configuration=dict(benchmark_start_time="2016-10-31T00:00:00+08:00",
        passenger_acceptance_seed=20260827))), encoding="utf-8")
    monkeypatch.setattr(world.pq, "read_schema", lambda *args: NS(names=list(world.DECISION_BASE_COLUMNS)))
    monkeypatch.setattr(world, "_read_projection", lambda path, *args, **kwargs:
        pd.DataFrame(dict(request_time=[pd.Timestamp("2016-10-31T00:00:10+08:00")])) if path.name == "requests" else pd.DataFrame())
    seeds = []
    monkeypatch.setattr(world, "build_controlled_fleet", lambda *args, **kwargs: seeds.append(kwargs) or "fixture fleet")
    monkeypatch.setattr(world, "episode_from_frames", lambda *args, **kwargs: NS(requests=None, metadata=kwargs["metadata"]))
    monkeypatch.setattr(world, "ExecutionTruthLookup", lambda *args, **kwargs: None)
    original = world.load_research_episode(tmp_path)
    explicit = world.load_research_episode(tmp_path, requested_q_a=.05, supply_seed=20261013)
    assert [(row["requested_q_a"], row["seed"]) for row in seeds] == [(.1, 20260824), (.05, 20261013)]
    assert original.metadata["supply_seed"] == 20260824 and explicit.metadata["supply_seed"] == 20261013
