from pathlib import Path

import numpy as np
import pandas as pd

from stage3.odd_tod.capability_envelope import (
    AUTHORIZED_BASE, DYNAMIC_DIMS, M3_SHA256, PHASE_STATUS, PI, Q_TAIL, SPEED_CAPS, TRAIN_DATES, VALIDATION_DATES, apply_mid_cdf,
    boundary_road_class_diversity, build_profiles, build_static_reference, dynamic_caps, movement_rules,
    movement_compatibility, verify_categorical_nestedness,
    parse_route_complex_encounters, quantile_higher, resolve_route_tokens,
    route_eqc, static_caps, validate_s3_date, verify_nestedness,
    weighted_mid_cdf_reference,
)
from stage3.odd_tod.network_foundation import Stage3S2AError, sha256_file


ROOT = Path(__file__).resolve().parents[2]


def test_s3_authorized_base_and_m3_checkpoint():
    assert AUTHORIZED_BASE == "c9b6bcdf136ee11fc2863609218a198f60a332c8"
    assert sha256_file(ROOT / "stage2/output_v5_2/development/M3/epoch_004.pt") == "965fc491cd77256f7889961d89932ec6be709bab04adcca358ac1b49f47c2cde"


def test_s2b_final_hashes_and_stage2_contract_are_bound():
    import json
    release = json.loads((ROOT / "stage3/docs/odd_tod/s2b/stage3_s2b_final_release_manifest.json").read_text(encoding="utf-8"))
    assert release["phase_status"] == "STAGE3_S2B_INTERSECTION_COMPLEX_FROZEN"
    for item in release["final_outputs"].values():
        assert sha256_file(ROOT / item["path"]) == item["sha256"]
    assert (ROOT / "stage2/docs/v5_2/stage2_v5_2_final_release_manifest.json").is_file()
    assert (ROOT / "stage2/docs/v5_2/stage2_v5_2_to_stage3_contract.md").is_file()


def test_stage2_m3_only_no_retraining_or_checkpoint_reselection():
    source = (ROOT / "stage3/odd_tod/capability_envelope.py").read_text(encoding="utf-8")
    assert M3_SHA256 in source
    assert "epoch_004.pt" in source
    assert "optimizer.step" not in source and "model.train(" not in source


def test_dates_and_scope_are_exact():
    assert TRAIN_DATES == tuple(f"201610{d:02d}" for d in range(9, 25))
    assert VALIDATION_DATES == ("20161025", "20161026", "20161027")
    assert PHASE_STATUS == "STAGE3_S3_CAPABILITY_ENVELOPE_FROZEN"


def test_historical_identity_is_typed_without_forward_projection_or_imputation():
    route = pd.DataFrame({
        "date": ["20161009"] * 3, "order_id": ["o"] * 3, "route_sequence": [0, 1, 2],
        "canonical_edge_uid": ["full", "reverse", "unknown"], "observed_direction": ["F", "R", "F"],
        "observed_directed_edge_uid": ["full", "reverse", "unknown"],
    })
    mapping = pd.DataFrame({"canonical_edge_uid": ["full"], "canonical_traversal_direction": ["F"], "stage3_edge_uid": ["s3e_full"], "mapping_status": ["EXACT_VALHALLA"]})
    overlay = pd.DataFrame({"canonical_edge_uid": ["reverse"], "canonical_traversal_direction": ["R"], "physical_forward_stage3_edge_uid": ["s3e_forward_only"], "av_routability_status": ["AV_ROUTABILITY_VIOLATION"], "historical_direction_status": ["HISTORICAL_DIRECTION_OVERLAY"], "missing_identity": [False]})
    result = resolve_route_tokens(route, mapping, overlay)
    assert list(result["route_token_type"]) == ["FULL_NETWORK_EDGE", "HISTORICAL_REVERSE_OVERLAY", "UNRESOLVED"]
    assert result.loc[1, "resolved_stage3_edge_uid"] is None or pd.isna(result.loc[1, "resolved_stage3_edge_uid"])
    assert result.loc[1, "physical_forward_stage3_edge_uid"] == "s3e_forward_only"
    assert result.loc[2, "resolution_status"] == "UNRESOLVED"


def _parser_fixture(gap=False):
    token_type = ["FULL_NETWORK_EDGE"] * 6
    edges = ["in1", "internal", "out1", "in2", "out2", "tail"]
    if gap: token_type[2] = "UNRESOLVED"; edges[2] = None
    typed = pd.DataFrame({"date": ["20161009"] * 6, "order_id": ["o"] * 6, "route_sequence": range(6), "route_token_type": token_type, "resolved_stage3_edge_uid": edges})
    boundary = pd.DataFrame([
        ["in1", "c", "INCOMING"], ["internal", "c", "INTERNAL"], ["out1", "c", "OUTGOING"],
        ["in2", "c", "INCOMING"], ["out2", "c", "OUTGOING"], ["tail", "z", "INTERNAL"],
    ], columns=["stage3_edge_uid", "intersection_complex_uid", "boundary_role"])
    lookup = pd.DataFrame({"intersection_complex_uid": ["c", "c"], "incoming_stage3_edge_uid": ["in1", "in2"], "outgoing_stage3_edge_uid": ["out1", "out2"]})
    return typed, boundary, lookup


def test_route_parser_internal_zero_internal_and_repeated_occurrences():
    result = parse_route_complex_encounters(*_parser_fixture())
    assert len(result) == 2
    assert list(result["internal_edge_count"]) == [1, 0]
    assert list(result["movement_occurrence_index"]) == [0, 1]
    assert set(result["movement_lookup_status"]) == {"MATCHED_TOPOLOGICAL_MOVEMENT"}


def test_route_parser_does_not_splice_across_broken_identity():
    typed, boundary, lookup = _parser_fixture(gap=True)
    result = parse_route_complex_encounters(typed, boundary, lookup)
    assert len(result) == 1
    assert result.iloc[0]["incoming_stage3_edge_uid"] == "in2"


def test_static_reference_unique_not_demand_weighted_and_exact_dimensions():
    encounters = pd.DataFrame({"intersection_complex_uid": ["a", "a", "b"]})
    complexes = pd.DataFrame({
        "intersection_complex_uid": ["a", "b"], "external_physical_connection_count": [3, 4],
        "topological_movement_count": [2, 5], "road_class_diversity": [1, 2], "internal_length_m": [4.0, 8.0],
        "signal_state": ["SIGNALIZED", "UNKNOWN_CONTROL"], "roundabout_evidence_present": [False, False],
        "grade_separation_evidence_present": [False, True],
    })
    boundary = pd.DataFrame({
        "stage3_edge_uid": ["a_in", "a_out", "a_internal", "b_in"],
        "intersection_complex_uid": ["a", "a", "a", "b"],
        "boundary_role": ["INCOMING", "OUTGOING", "INTERNAL", "INCOMING"],
    })
    edges = pd.DataFrame({
        "stage3_edge_uid": ["a_in", "a_out", "a_internal", "b_in"],
        "valhalla_road_class": [1, 2, 99, 3],
    })
    result = build_static_reference(encounters, complexes, boundary, edges)
    assert len(result) == 2
    assert {"intersection_complex_uid", "A_c", "M_c", "D_c", "L_c", "signal_state", "roundabout_evidence_present", "grade_separation_evidence_present", "train_encounter_count"}.issubset(result.columns)
    assert result.set_index("intersection_complex_uid").loc["a", "train_encounter_count"] == 2
    assert result.set_index("intersection_complex_uid").loc["a", "D_c"] == 2
    assert result.set_index("intersection_complex_uid").loc["b", "D_c"] == 1
    assert result.set_index("intersection_complex_uid").loc["a", "s2b_internal_road_class_diversity_qa"] == 1


def test_boundary_road_class_diversity_excludes_internal_edges():
    boundary = pd.DataFrame({
        "stage3_edge_uid": ["in", "out", "internal"], "intersection_complex_uid": ["c"] * 3,
        "boundary_role": ["INCOMING", "OUTGOING", "INTERNAL"],
    })
    edges = pd.DataFrame({"stage3_edge_uid": ["in", "out", "internal"], "valhalla_road_class": [1, 2, 9]})
    result = boundary_road_class_diversity(boundary, edges).iloc[0]
    assert result["boundary_road_class_diversity"] == 2
    assert result["boundary_edge_count"] == 2


def test_higher_quantile_and_static_support_gate():
    assert quantile_higher([1, 2, 3, 4], .75) == 4
    try: static_caps(pd.DataFrame({"A_c": [1], "M_c": [1], "D_c": [1], "L_c": [1]}))
    except Stage3S2AError as exc: assert "support gate" in str(exc)
    else: raise AssertionError("support gate not enforced")


def test_mid_cdf_large_ties_and_time_weighting():
    ref = weighted_mid_cdf_reference([0, 0, 1], [1, 3, 4], "stop")
    assert list(ref["value"]) == [0, 1]
    assert np.allclose(ref["mid_cdf"], [.25, .75])
    assert np.allclose(apply_mid_cdf([0, .5, 1, 2], ref), [.25, .5, .75, 1.0])


def test_eqc_time_weighting_strict_tail_route_order_and_seconds():
    tokens = pd.DataFrame({
        "date": ["20161009"] * 4, "order_id": ["o"] * 4, "route_sequence": [0, 1, 2, 3],
        "travel_time_p50_s": [1.0, 2.0, 3.0, 4.0],
        **{f"pred_{d}": [0.0, 1.0, 0.0, 1.0] for d in ("crawl", "stop", "speed_cv", "acceleration_rms")},
    })
    reference = pd.concat([weighted_mid_cdf_reference([0, 1], [1, 1], d) for d in ("crawl", "stop", "speed_cv", "acceleration_rms")])
    out = route_eqc(tokens, reference).iloc[0]
    assert np.isclose(out["crawl_E"], .55)
    assert out["crawl_Q"] == 0.0  # mid-CDF maximum is .75, strict > .90
    tail_ref = pd.DataFrame({"dimension": ["crawl"], "value": [1.0], "predicted_time_weight_s": [1.0], "mid_cdf": [.95], "total_predicted_time_weight_s": [1.0]})
    all_ref = pd.concat([tail_ref] + [weighted_mid_cdf_reference([0, 1], [1, 1], d) for d in ("stop", "speed_cv", "acceleration_rms")])
    out = route_eqc(tokens, all_ref).iloc[0]
    assert np.isclose(out["crawl_Q"], .6)
    assert out["crawl_C"] == 4.0


def test_dynamic_support_gate_and_one_route_one_sample_quantile():
    frame = pd.DataFrame({f"{d}_{m}": np.arange(4000) for d in ("crawl", "stop", "speed_cv", "acceleration_rms") for m in ("E", "Q", "C")})
    caps = dynamic_caps(frame)
    assert caps["C"]["crawl"]["E"] == quantile_higher(np.arange(4000), .75)
    try: dynamic_caps(frame.iloc[:3999])
    except Stage3S2AError: pass
    else: raise AssertionError("dynamic support gate not enforced")


def test_profile_constants_rules_nestedness_and_noncompensation():
    assert PI == {"C": .75, "M": .90, "A": .975}
    assert SPEED_CAPS == {"C": 60, "M": 80, "A": 120}
    assert Q_TAIL == .90
    rules = movement_rules()
    assert rules["C"]["LEFT"]["UNKNOWN_CONTROL"] == "UNKNOWN"
    assert rules["M"]["LEFT"] == "COMPATIBLE"
    assert rules["C"]["UTURN"] == rules["M"]["UTURN"] == "INCOMPATIBLE"
    assert rules["A"]["UTURN"] == "COMPATIBLE"
    assert all(rules[p]["UNKNOWN"] == "UNKNOWN" for p in rules)
    static = {p: {d: i for d in ("A_c", "M_c", "D_c", "L_c")} for i, p in enumerate(("C", "M", "A"), 1)}
    dynamic = {p: {d: {m: i for m in ("E", "Q", "C")} for d in ("crawl", "stop", "speed_cv", "acceleration_rms")} for i, p in enumerate(("C", "M", "A"), 1)}
    profile = build_profiles(static, dynamic); verify_nestedness(profile)
    assert profile["non_compensatory"] is True
    assert profile["s4_authorized"] is False and profile["next_phase_authorized"] is False


def test_categorical_rules_and_nestedness_are_programmatic():
    for profile in ("C", "M", "A"):
        assert movement_compatibility(profile, "STRAIGHT") == "COMPATIBLE"
        assert movement_compatibility(profile, "RIGHT") == "COMPATIBLE"
        assert movement_compatibility(profile, "UNKNOWN") == "UNKNOWN"
    assert movement_compatibility("C", "LEFT", "SIGNALIZED") == "COMPATIBLE"
    assert movement_compatibility("C", "LEFT", "STOP_OR_YIELD_CONTROLLED") == "INCOMPATIBLE"
    assert movement_compatibility("C", "LEFT", "UNKNOWN_CONTROL") == "UNKNOWN"
    assert movement_compatibility("M", "LEFT", "UNKNOWN_CONTROL") == "COMPATIBLE"
    assert movement_compatibility("A", "LEFT", "UNKNOWN_CONTROL") == "COMPATIBLE"
    assert movement_compatibility("C", "UTURN") == "INCOMPATIBLE"
    assert movement_compatibility("M", "UTURN") == "INCOMPATIBLE"
    assert movement_compatibility("A", "UTURN") == "COMPATIBLE"
    assert movement_compatibility("C", "STRAIGHT", roundabout=True) == "INCOMPATIBLE"
    assert movement_compatibility("M", "STRAIGHT", roundabout=True) == "COMPATIBLE"
    assert movement_compatibility("A", "LEFT", restriction_enforcement_certified=True, movement_legality_state="CERTIFIED_PROHIBITED") == "INCOMPATIBLE"
    verify_categorical_nestedness()


def test_profile_claim_boundaries_speed_static_and_dynamic_contract():
    static = {p: {d: i for d in ("A_c", "M_c", "D_c", "L_c")} for i, p in enumerate(("C", "M", "A"), 1)}
    dynamic = {p: {d: {m: i for m in ("E", "Q", "C")} for d in DYNAMIC_DIMS} for i, p in enumerate(("C", "M", "A"), 1)}
    profiles = build_profiles(static, dynamic)
    assert profiles["quantile_method"] == "higher"
    assert profiles["dynamic_cdf"].startswith("global Train")
    for profile in profiles["profiles"]:
        assert profile["grade_bridge_tunnel_rule"] == "DESCRIPTIVE_ONLY"
        assert "not safety" in profile["claim_boundary"]
        assert set(profile["static_caps"]) == {"external_physical_connection_count", "topological_movement_count", "road_class_diversity", "internal_length_m"}
        assert set(profile["dynamic_caps"]) == set(DYNAMIC_DIMS)


def test_decision_time_only_and_no_realized_future_inputs():
    source = (ROOT / "stage3/odd_tod/capability_envelope.py").read_text(encoding="utf-8")
    assert '"decision_time_only": True' in source
    assert '"realized_future_time_used": False' in source
    forbidden_prediction_columns = ("actual_arrival_time", "actual_link_time", "actual_cross_time", "future_observed_traffic")
    for name in forbidden_prediction_columns:
        assert name not in source


def test_no_route_fui_fallback_or_stage4_output():
    source = (ROOT / "stage3/odd_tod/capability_envelope.py").read_text(encoding="utf-8").lower()
    assert "fallback_routing" in source and '"fallback_routing": false' in source
    assert '"stage4": false' in source
    assert "route_fui" not in source


def test_test31_is_hard_rejected():
    try: validate_s3_date("20161031", ["20161031"])
    except Stage3S2AError as exc: assert "Test31" in str(exc)
    else: raise AssertionError("Test31 accepted")
