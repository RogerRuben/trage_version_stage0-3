from types import SimpleNamespace as NS

import pandas as pd

from stage3.odd_tod.research_compatibility import ResearchMovement, evaluate_research_compatibility
from stage4.analysis.flexibility_prepare import MovementClassifier
from stage4.analysis.symmetric_research_prepare import HistoricalResearchParser


def classifier():
    c = MovementClassifier.__new__(MovementClassifier)
    c.complexes = {"j": dict(signalized_research=False,
        control_basis="RESEARCH_ASSUMED_NON_SIGNAL", roundabout_evidence_present=False)}
    c.lookup = {}
    # Travel reversed f from east to junction (270deg), then leave south (180deg).
    c.bearings = {"f": (90., 90.), "g": (180., 180.)}
    return c


def test_reverse_role_and_bearing_preserve_unsignalized_left_not_forward_movement():
    c = classifier()
    boundary = pd.DataFrame([("f", "j", "OUTGOING"), ("g", "j", "OUTGOING")],
        columns=["stage3_edge_uid", "intersection_complex_uid", "boundary_role"])
    overlay = pd.DataFrame([dict(canonical_edge_uid="c", physical_forward_stage3_edge_uid="f")])
    parser = HistoricalResearchParser(boundary, overlay, c)
    source = pd.DataFrame([dict(date="20161031", order_id="o", route_sequence=0,
        canonical_edge_uid="c", route_token_type="HISTORICAL_REVERSE_OVERLAY", resolved_stage3_edge_uid=None),
        dict(date="20161031", order_id="o", route_sequence=1, canonical_edge_uid="g",
        route_token_type="FULL_NETWORK_EDGE", resolved_stage3_edge_uid="g")])
    identity, classified, _, encounters = parser.parse(source)
    assert identity.loc["o", "direction_supported"]
    assert source.iloc[0].route_token_type == "HISTORICAL_REVERSE_OVERLAY"
    assert encounters.iloc[0].incoming_stage3_edge_uid == "RESEARCH_REVERSE:c:R"
    movements = classified["o"][1]
    assert movements[0].maneuver == "LEFT"
    nc = dict(NC=1., ML=0., MR=0., CG=0., SC=0., U=0.)
    assert not evaluate_research_compatibility("C", movements, nc).compatible
    assert evaluate_research_compatibility("M", movements, nc).compatible
    assert evaluate_research_compatibility("A", movements, nc).compatible
    assert evaluate_research_compatibility("HV", movements, nc).compatible


def test_unsupported_reverse_is_common_support_failure_not_guessed_forward():
    c = classifier()
    b = pd.DataFrame(columns=["stage3_edge_uid", "intersection_complex_uid", "boundary_role"])
    overlay = pd.DataFrame([dict(canonical_edge_uid="c", physical_forward_stage3_edge_uid=None)])
    p = HistoricalResearchParser(b, overlay, c)
    source = pd.DataFrame([dict(date="20161031", order_id="o", route_sequence=0,
        canonical_edge_uid="c", route_token_type="HISTORICAL_REVERSE_OVERLAY", resolved_stage3_edge_uid=None)])
    identity, _, _, _ = p.parse(source)
    assert not identity.loc["o", "direction_supported"]
    assert identity.loc["o", "unsupported_reverse_token_count"] == 1


def test_A_equals_HV_and_certified_prohibition_is_common():
    traffic = dict(NC=0., ML=0., MR=0., CG=0., SC=1., U=0.)
    for maneuver in ("LEFT", "RIGHT", "STRAIGHT", "UTURN", "ROUNDABOUT"):
        m = (ResearchMovement(maneuver, False, "RESEARCH_ASSUMPTION"),)
        assert evaluate_research_compatibility("A", m, traffic).compatible
        assert evaluate_research_compatibility("HV", m, traffic).compatible
    m = (ResearchMovement("LEFT", True, "POSITIVE_EVIDENCE", True),)
    assert all(not evaluate_research_compatibility(k, m, traffic).compatible for k in ("C", "M", "A", "HV"))
