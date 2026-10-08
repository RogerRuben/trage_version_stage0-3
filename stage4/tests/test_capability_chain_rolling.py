"""Only information boundary and executed-clock accounting tests for the adapter."""
from stage4.analysis.capability_chain_rolling import HistoricalForecastReference, first_actions, visible_tasks


def test_reveal_boundary_and_history_forecasts_are_separate():
    actual = [dict(job_id="A1", release_s=0, deadline_s=300), dict(job_id="A2", release_s=90, deadline_s=390)]
    assert [t["job_id"] for t in visible_tasks(actual, 30, {})] == ["A1"]
    assert visible_tasks(actual, 30, {"A1": {}}) == []
    assert visible_tasks(actual, 400, {}) == []
    cfg = dict(step_s=30, horizon_s=1800, coarse_reference_update_s=300)
    cohorts = [dict(scenario_id="20161017", weight=1.0, tasks=[
        dict(job_id="F1", release_s=45), dict(job_id="F2", release_s=660)])]
    reference = HistoricalForecastReference(cohorts, cfg)
    assert [t["job_id"] for t in reference.view(30)[0]["tasks"]] == ["F1", "F2"]
    assert [t["job_id"] for t in reference.view(60)[0]["tasks"]] == ["F2"]
    assert reference.update_count == 1
    reference.view(300)
    assert reference.update_count == 2
    # No actual replay list is accepted by the historical-reference interface.
    assert [t["job_id"] for t in visible_tasks(actual + [dict(job_id="A_hidden", release_s=1200, deadline_s=1500)], 30, {})] == ["A1"]


def test_busy_resource_not_teleported_into_current_pickup_or_relocation():
    class NoQueries:
        def connection(self, *args):
            raise AssertionError("busy resource queried a current pickup")
        def relocation(self, *args):
            raise AssertionError("busy resource queried an idle movement")
    resources = [dict(resource_id="H1", profile_id="HV", location_id="A_done_destination",
        ready_s=900, admission_end_s=1800, last_relocation_s=-900, relocation_count=0)]
    pending = [dict(job_id="A_waiting", release_s=0, deadline_s=300, compatible_profiles=["HV"])]
    actions = first_actions(resources, pending, NoQueries(), 30, dict(step_s=30))
    assert len(actions) == 1 and actions[0]["kind"] == "BUSY" and actions[0]["ready_s"] == 900


def test_relocation_stop_does_not_invent_a_next_customer_or_outgoing_edge():
    import pandas as pd
    import pyarrow as pa
    from stage4.analysis.capability_chain_instances import JoinEvidence
    from stage4.dispatch.controlled_routes import MOVEMENT_COLUMNS, CONTROL_COLUMNS

    class Router:
        _boundary = pa.table(dict(stage3_edge_uid=["E1", "E2"],
            intersection_complex_uid=["C1", "C1"], boundary_role=["INCOMING", "INTERNAL"]))
        _movements = pa.table({c:pa.array([], type=pa.bool_() if c == "restriction_enforcement_certified" else pa.string()) for c in MOVEMENT_COLUMNS})
        _controls = pa.table(dict(intersection_complex_uid=["C1"], roundabout_evidence_present=[False],
            signalized_research=[True], control_basis=["POSITIVE_OSM_OR_GAPFILL"]))
        def small_geometry(self, uids):
            return {}
        def _bearings_for(self, uids):
            return {}
    evidence = JoinEvidence.__new__(JoinEvidence)
    evidence.router = Router()
    evidence.tokens = {}
    evidence.selected = pd.DataFrame()
    evidence.overlay = pd.DataFrame(columns=["canonical_edge_uid", "physical_forward_stage3_edge_uid"])
    evidence.overlay_forward = {}
    evidence.simulation_date = "20161024"
    # The physical empty route stops within the complex. There is no outgoing
    # traversal and therefore no completed/invented maneuver or customer label.
    result = evidence.evaluate(None, None, dict(edges=[dict(stage3_edge_uid="E1"), dict(stage3_edge_uid="E2")]), 80)
    assert result["supported"] and result["join_encounter_count"] == 0
    assert result["compatible_profiles"] == ["HV", "C"]
