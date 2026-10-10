from copy import deepcopy

import pytest

from stage4.dispatch.scheme_a_joint_planner import CoarseHistoricalScenarios, SchemeAJointPlanner


def settings():
    return dict(planning_horizon_s=1800, horizon_s=3600, step_s=30, patience_s=300,
        coarse_reference_update_s=300, solver_decision_limit_s=10.0,
        reposition_interval_s=900, reposition_max_moves=50, reposition_radius_m=2000,
        reposition_top_k=3, reposition_max_eta_s=300)


def test_coarse_refresh_preserves_exact_timestamps_and_never_reveals_forecasts():
    cohorts = [dict(scenario_id="history", weight=1.0, tasks=[dict(job_id="old",
        source_kind="FORECAST", release_s=20), dict(job_id="inside", source_kind="FORECAST",
        release_s=1799), dict(job_id="outside", source_kind="FORECAST", release_s=1801)])]
    store = CoarseHistoricalScenarios(cohorts, settings())
    assert [t["job_id"] for t in store.view(0)[0]["tasks"]] == ["old", "inside"]
    assert [t["job_id"] for t in store.view(30)[0]["tasks"]] == ["inside"]
    assert store.refresh_count == 1
    assert [t["job_id"] for t in store.view(300)[0]["tasks"]] == ["inside", "outside"]
    assert store.refresh_count == 2
    assert cohorts[0]["tasks"][1]["release_s"] == 1799


def test_joint_layout_keeps_HV_positions_and_pending_keeps_original_deadline():
    class Provider:
        def connection(self, origin, target, allowed):
            assert target in allowed
            return dict(origin_id=origin, target_job_id=target, supported=False,
                compatible_profiles=[], travel_time_s=None, empty_distance_m=None)

    planner = SchemeAJointPlanner(Provider(), [dict(scenario_id="h", weight=1.0, tasks=[])], settings())
    templates = [dict(resource_id="H", profile_id="HV", location_id="S1", admission_end_s=1800),
        dict(resource_id="C", profile_id="C", location_id="S1", admission_end_s=1800)]
    sites = [dict(site_id="S1"), dict(site_id="S2")]
    p = planner.layout_problem(templates, sites, ["S1", "S2"], mode="CHAIN_JOINT")
    assert [a["location_id"] for a in p["actions"] if a["resource_id"] == "H"] == ["S1"]
    assert [a["location_id"] for a in p["actions"] if a["resource_id"] == "C"] == ["S1", "S2"]
    bad = dict(job_id="not-revealed", release_s=60, deadline_s=360, source_kind="ACTUAL_PENDING")
    with pytest.raises(ValueError, match="revealed"):
        planner.plan_epoch([], [bad], 30)
    bad = deepcopy(bad)
    bad.update(release_s=0, deadline_s=301)
    with pytest.raises(ValueError, match="300s"):
        planner.plan_epoch([], [bad], 30)
