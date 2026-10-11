"""Necessary two-clock integration checks; no native scenario or Actor."""
from pathlib import Path
from types import SimpleNamespace as NS

import pandas as pd
import pytest

from stage4.analysis.scheme_a_two_scale_experiment import group_configuration, load_protocol
from stage4.dispatch.routing_v4 import PickupEtaBudget
from stage4.dispatch.solver import LexicographicResult
from stage4.dispatch.waiting_contract import assignment_feasible, pickup_eta_limit_s
from stage4.tests.test_controlled_exit import controller, START
from stage4.tests.test_research_native_bridge import fixture as bridge_fixture, activate
from stage4.tests.test_city_pickup_support import ETA, Routes, AllowedJoin, case, SNAPSHOT
from stage4.dispatch.city_pickup_support import CityPickupSupportValidator


CFG = dict(patience_mode="MATCH_DEADLINE_AND_PICKUP_CAP", max_match_wait_s=300,
    max_pickup_eta_s=600)


def test_independent_clocks_and_v4_configuration_preserve_original_conditions():
    assert not assignment_feasible({}, 290., 300., 60.)
    assert assignment_feasible(CFG, 290., 300., 600.)
    assert not assignment_feasible(CFG, 290., 300., 600.01)
    assert not assignment_feasible(CFG, 300., 300., 0.)
    assert not assignment_feasible(CFG, 290., 300., -1.)
    budget = PickupEtaBudget(pickup_eta_limit_s(CFG, 10), 290, 120,
        pickup_exclusion_reason="PICKUP_CAP")
    assert budget.rejection(600.) is None and budget.rejection(600.01) == "PICKUP_CAP"
    root = Path(__file__).resolve().parents[2]
    old = load_protocol(root, Path("stage4/config/scheme_a_two_scale_city_v3.json"))
    new = load_protocol(root, Path("stage4/config/scheme_a_two_scale_city_v4.json"))
    for key in ("patience_s", "requested_q_a", "passenger_acceptance_rate", "solver_time_limit_s",
                "planning_horizon_s", "current_top_k", "forecast_seed", "idle_movement_mode"):
        assert old[key] == new[key]
    assert new["max_match_wait_s"] == 300 and new["max_pickup_eta_s"] == 600
    assert not new["post_assignment_cancellation"] and new["max_total_wait_s"] is None
    assert group_configuration(old, "HOTSPOT_BASELINE")["disk_route_cache"] != (
        group_configuration(new, "HOTSPOT_BASELINE")["disk_route_cache"])


@pytest.mark.parametrize("kind", ["HV", "AV"])
def test_critical_order_can_be_assigned_before_deadline_with_pickup_after_it(kind):
    c, runtime, request, eta = controller(kind)
    c.config.update(CFG)
    c.match_deadline_mode = True
    request.sim_time_s = -200  # At t=90 it has waited 290 seconds.
    request.request_time = START + pd.Timedelta(seconds=-200)
    c.user_request(request.native_request, 90)
    c.request_meta[0].update(research_route_compatible=True, research_base_eligible=True)
    c.flexibility_adapter = NS(policy="SERVICE_PRESERVING", solve=lambda *a:
        LexicographicResult((0,), 0., 1, 1, 0))
    c.time_trigger(90)
    assert len(c.assignment_rows) == 1 and c.request_meta[0]["entered_critical"]
    assert c.assignment_rows[0]["pickup_eta_s"] == 20  # Ten seconds left to match, not to arrive.
    assert eta.budgets[0][0].remaining_patience_s == 600
    assert c.epoch_rows[-1]["critical_matched"] == 1 and not c.expired_rids


def test_actual_native_commitment_uses_same_cap_and_truth_is_still_after_commit():
    cfg = dict(CFG, max_match_wait_s=10)  # Synthetic source has a ten-second matching deadline.
    episode, bridge, c, calls = bridge_fixture(waiting_config=cfg)
    activate(bridge, c, 10)
    record = next(r for r in bridge.requests.values() if r.order_id == "inside")
    from stage4.fleetpy_adapter.valhalla_time_adapter import PickupEstimate
    with pytest.raises(ValueError, match="waiting contract"):
        c._assign(c.runtime_by_vid[0], record, PickupEstimate(601., 601., 100., 1., 0, False), 10)
    assert not episode.requests.was_committed("inside") and episode.execution_truth._values is None
    c._assign(c.runtime_by_vid[0], record, PickupEstimate(600., 600., 100., 1., 0, False), 10)
    assert episode.requests.was_committed("inside") and bridge.diagnostics()["deferred_truth_calls"] == 1
    assert episode._commitments["slot-0"].predicted_pickup_time_s == 610.
    assert calls[0][3] == 600.  # Native physical pickup leg, not merely a loosened accounting label.


def test_c_scalar_validator_does_not_reapply_old_remaining_arrival_deadline():
    eta = ETA()
    c, arc = case(eta)
    c.config = dict(CFG)
    validator = CityPickupSupportValidator(Routes(eta), geometry_timestamp=SNAPSHOT,
        routing_budget_s=10., seam_evidence=AllowedJoin())
    assert validator(c, [arc], 62220.)[0].pickup_eta_s == 200.  # Only 180 seconds left to match.
    c.request_meta[100]["pickup_deadline_s"] = 62220.
    assert validator(c, [arc], 62220.) == []  # Assignment deadline itself remains strict.
