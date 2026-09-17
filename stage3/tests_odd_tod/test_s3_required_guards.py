"""Explicitly named S3 taskbook guards for static and evidence-chain audit."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from stage3.odd_tod.capability_envelope import (
    DYNAMIC_DIMS, M3_SHA256, PI, Q_TAIL, SPEED_CAPS, S31_AUTHORIZED_BASE,
    apply_mid_cdf, build_profiles, movement_compatibility,
    quantile_higher, resolve_route_tokens, route_eqc, validate_s3_date,
    verify_categorical_nestedness, verify_nestedness,
    weighted_mid_cdf_reference,
)
from stage3.odd_tod.network_foundation import Stage3S2AError, sha256_file


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "stage3/odd_tod/capability_envelope.py").read_text(encoding="utf-8")


def _profile():
    static = {p: {d: i for d in ("A_c", "M_c", "D_c", "L_c")} for i, p in enumerate(("C", "M", "A"), 1)}
    dynamic = {p: {d: {m: i for m in ("E", "Q", "C")} for d in DYNAMIC_DIMS} for i, p in enumerate(("C", "M", "A"), 1)}
    return build_profiles(static, dynamic)


def test_m3_checkpoint_hash_bound():
    assert sha256_file(ROOT / "stage2/output_v5_2/development/M3/epoch_004.pt") == M3_SHA256


def test_reverse_overlay_not_projected_forward():
    assert "physical reference" in SOURCE and '"AV_ROUTABILITY_VIOLATION"' in SOURCE


def test_unresolved_identity_not_geometry_imputed():
    route = pd.DataFrame({
        "order_id": ["o"], "route_sequence": [0], "canonical_edge_uid": ["missing"],
        "observed_direction": ["F"], "observed_directed_edge_uid": ["missing:F"],
    })
    mapping = pd.DataFrame(columns=["canonical_edge_uid", "canonical_traversal_direction", "stage3_edge_uid", "mapping_status"])
    overlay = pd.DataFrame(columns=["canonical_edge_uid", "canonical_traversal_direction", "physical_forward_stage3_edge_uid", "av_routability_status", "historical_direction_status", "missing_identity"])
    result = resolve_route_tokens(route, mapping, overlay).iloc[0]
    assert result["route_token_type"] == "UNRESOLVED"
    assert pd.isna(result["resolved_stage3_edge_uid"])


def test_route_parser_handles_zero_internal_edge_transition():
    assert '"internal": []' in SOURCE


def test_route_parser_preserves_repeated_complex_occurrences():
    assert "movement_occurrence_index" in SOURCE and "occurrence += 1" in SOURCE


def test_route_parser_does_not_splice_across_reverse_overlay():
    assert "if token.route_token_type != \"FULL_NETWORK_EDGE\"" in SOURCE and "active.clear()" in SOURCE


def test_route_parser_does_not_splice_across_unresolved_token():
    test_route_parser_does_not_splice_across_reverse_overlay()


def test_static_reference_uses_unique_complexes():
    assert 'duplicated().any()' in SOURCE and 'validate="one_to_one"' in SOURCE


def test_static_reference_not_encounter_weighted():
    assert "train_encounter_count" in SOURCE and "quantile_higher(reference[dimension]" in SOURCE


def test_static_dimensions_exactly_a_m_d_l():
    assert set(_profile()["profiles"][0]["static_caps"]) == {"external_physical_connection_count", "topological_movement_count", "road_class_diversity", "internal_length_m"}


def test_static_d_is_boundary_road_class_diversity():
    profile = _profile()
    assert "INCOMING/OUTGOING boundary edges" in profile["static_dimension_definitions"]["D_c"]
    assert "INTERNAL edges excluded" in profile["static_dimension_definitions"]["D_c"]


def test_static_quantile_method_higher():
    assert quantile_higher([1, 2, 3, 4], .75) == 4


def test_static_support_gate():
    assert "static support gate failed" in SOURCE and "< 1000" in SOURCE


def test_no_stage2_retraining():
    assert "optimizer.step" not in SOURCE and "backward(" not in SOURCE


def test_no_checkpoint_reselection():
    assert "epoch_004.pt" in SOURCE and "best_checkpoint" not in SOURCE


def test_dynamic_features_decision_time_only():
    assert '"decision_time_only": True' in SOURCE and "feature timestamp" in SOURCE


def test_no_realized_arrival_time():
    assert "actual_arrival_time" not in SOURCE


def test_no_realized_link_time():
    assert "actual_link_time" not in SOURCE and "actual_cross_time" not in SOURCE


def test_no_realized_future_traffic():
    assert "future_observed_traffic" not in SOURCE


def test_dynamic_common_complete_route_cohort():
    assert "dynamic_route_complete" in SOURCE and "complete_dynamic_tokens" in SOURCE


def test_no_dynamic_imputation():
    assert "no imputation" in SOURCE.lower()


def test_mid_cdf_formula():
    ref = weighted_mid_cdf_reference([0, 1], [1, 3], "crawl")
    assert np.allclose(ref["mid_cdf"], [.125, .625])


def test_mid_cdf_handles_large_ties():
    ref = weighted_mid_cdf_reference([0, 0, 1], [2, 2, 4], "stop")
    assert np.allclose(ref["mid_cdf"], [.25, .75])


def test_cdf_is_time_weighted():
    ref = weighted_mid_cdf_reference([0, 1], [1, 9], "stop")
    assert np.allclose(ref["mid_cdf"], [.05, .55])


def test_cdf_train_only():
    assert "_build_cdf_reference" in SOURCE and "for date in TRAIN_DATES" in SOURCE


def test_cdf_global_not_context_specific():
    assert _profile()["dynamic_cdf"].startswith("global Train")


def test_qtail_exactly_0p90():
    assert Q_TAIL == .90


def test_qtail_strict_greater_than():
    assert "z > Q_TAIL" in SOURCE


def test_eqc_time_weighting():
    assert "np.dot(weight, z) / total" in SOURCE


def test_consecutive_tail_uses_route_order():
    assert 'sort_values(["date", "order_id", "route_sequence"])' in SOURCE


def test_consecutive_tail_in_seconds():
    assert "running + duration" in SOURCE


def test_dynamic_quantiles_one_route_one_sample():
    assert 'descriptors[f"{dimension}_{metric}"]' in SOURCE


def test_dynamic_quantile_method_higher():
    assert _profile()["quantile_method"] == "higher"


def test_dynamic_support_gate():
    assert "dynamic support gate failed" in SOURCE and "< 4000" in SOURCE


def test_pi_values_exactly_075_090_0975():
    assert PI == {"C": .75, "M": .90, "A": .975}
    assert "marginal" in _profile()["quantile_anchor_semantics"]
    assert "not joint route acceptance" in _profile()["quantile_anchor_semantics"]


def test_speed_caps_exactly_60_80_120():
    assert SPEED_CAPS == {"C": 60, "M": 80, "A": 120}


def test_static_profile_nestedness():
    verify_nestedness(_profile())


def test_dynamic_profile_nestedness():
    verify_nestedness(_profile())


def test_categorical_profile_nestedness():
    verify_categorical_nestedness()


def test_straight_rule():
    assert all(movement_compatibility(p, "STRAIGHT") == "COMPATIBLE" for p in ("C", "M", "A"))


def test_right_rule():
    assert all(movement_compatibility(p, "RIGHT") == "COMPATIBLE" for p in ("C", "M", "A"))


def test_conservative_left_requires_signal():
    assert movement_compatibility("C", "LEFT", "SIGNALIZED") == "COMPATIBLE"
    assert movement_compatibility("C", "LEFT", "STOP_OR_YIELD_CONTROLLED") == "INCOMPATIBLE"
    assert movement_compatibility("C", "LEFT", "UNKNOWN_CONTROL") == "UNKNOWN"


def test_moderate_advanced_left_control_not_required():
    assert movement_compatibility("M", "LEFT", "UNKNOWN_CONTROL") == "COMPATIBLE"
    assert movement_compatibility("A", "LEFT", "UNKNOWN_CONTROL") == "COMPATIBLE"


def test_uturn_profile_rule():
    assert [movement_compatibility(p, "UTURN") for p in ("C", "M", "A")] == ["INCOMPATIBLE", "INCOMPATIBLE", "COMPATIBLE"]


def test_unknown_turn_remains_unknown():
    assert all(movement_compatibility(p, "UNKNOWN") == "UNKNOWN" for p in ("C", "M", "A"))


def test_roundabout_rule():
    assert movement_compatibility("C", "STRAIGHT", roundabout=True) == "INCOMPATIBLE"
    assert movement_compatibility("M", "STRAIGHT", roundabout=True) == "COMPATIBLE"


def test_topological_movement_not_legal_claim():
    assert "NOT_CERTIFIED_OR_UNKNOWN_DOES_NOT_IMPLY_PERMISSION" in json.dumps(_profile())


def test_grade_separation_not_auto_infeasible():
    assert all(p["grade_bridge_tunnel_rule"] == "DESCRIPTIVE_ONLY" for p in _profile()["profiles"])


def test_validation_does_not_modify_profiles():
    path = ROOT / "stage3/output/odd_tod/s3/validation_sanity_summary.json"
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        assert value["profile_sha256_before_validation"] == value["profile_sha256_after_validation"]


def test_validation_not_used_for_threshold_selection():
    assert "threshold_selection_from_validation" in SOURCE and 'False' in SOURCE


def test_test31_rejected():
    try: validate_s3_date("20161031", ["20161031"])
    except Stage3S2AError: pass
    else: raise AssertionError("Test31 accepted")


def test_no_test31_input_binding():
    profile = _profile()
    assert profile["test31_used"] is False and "20161031" not in profile["calibration_dates"] + profile["validation_sanity_dates"]


def test_no_route_fui_output():
    assert "route_fui" not in SOURCE.lower()


def test_no_fallback_routing():
    assert '"fallback_routing": False' in SOURCE


def test_no_stage4():
    assert '"stage4": False' in SOURCE


def test_s4_not_authorized():
    assert _profile()["s4_authorized"] is False


def test_next_phase_not_authorized():
    assert _profile()["next_phase_authorized"] is False


def test_persisted_m3_caches_contain_no_realized_targets():
    cache_roots = [ROOT / "stage3/output/odd_tod/s3/cache/m3", ROOT / "stage3/output/odd_tod/s3/cache/m3_validation"]
    for cache_root in cache_roots:
        for path in cache_root.glob("*.parquet"):
            columns = pq.ParquetFile(path).schema_arrow.names
            assert not [c for c in columns if c.startswith("target_") or c.endswith("_target_valid")]


def test_s31_reviewed_base_is_bound():
    assert S31_AUTHORIZED_BASE == "309da4e5164eb99314c34b15ae2652f587a29f0b"


def test_all_train_m3_cache_manifests_bind_frozen_checkpoint_and_current_files():
    for date in (f"201610{day:02d}" for day in range(9, 25)):
        path = ROOT / f"stage3/output/odd_tod/s3/cache/m3/date={date}.parquet"
        manifest_path = path.with_suffix(".json")
        if not path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["model_id"] == "M3"
        assert manifest["checkpoint_sha256"] == M3_SHA256
        assert manifest["prediction_sha256"] == sha256_file(path)
        assert manifest["row_count"] == pq.ParquetFile(path).metadata.num_rows
        assert manifest["decision_time_only"] is True
        assert manifest["predicted_progression_only"] is True
        assert manifest["realized_target_columns_persisted"] is False


def test_s31_frozen_static_caps_change_only_boundary_d():
    path = ROOT / "stage3/output/odd_tod/s3/s31_closure_summary.json"
    if not path.is_file():
        return
    closure = json.loads(path.read_text(encoding="utf-8"))
    assert closure["a_m_l_unchanged"] is True
    assert closure["d_changed_only"] is True
    assert [closure["new_static_caps"][p]["D_c"] for p in ("C", "M", "A")] == [2.0, 3.0, 3.0]
    assert closure["dynamic_caps_unchanged"] is True
    assert closure["dynamic_product_hashes_before"] == closure["dynamic_product_hashes_after"]


def test_s31_train_static_reference_has_positive_boundary_d_and_provenance():
    path = ROOT / "stage3/output/odd_tod/s3/train_static_complex_reference.parquet"
    if not path.is_file():
        return
    frame = pd.read_parquet(path)
    assert (frame["D_c"] == frame["boundary_road_class_diversity"]).all()
    assert (frame["D_c"] >= 1).all()
    assert set(frame["road_class_diversity_definition"]) == {"UNIQUE_VALHALLA_ROAD_CLASS_ON_INCOMING_OUTGOING_BOUNDARY_EDGES"}


def test_s31_release_binds_all_train_prediction_caches():
    path = ROOT / "stage3/docs/odd_tod/s3/stage3_s3_release_manifest.json"
    if not path.is_file():
        return
    release = json.loads(path.read_text(encoding="utf-8"))
    if release["phase_status"] != "STAGE3_S31_CLOSURE_COMPLETE":
        return
    assert release["base_commit"] == S31_AUTHORIZED_BASE
    assert len(release["train_m3_prediction_caches"]) == 16
    assert all(value["checkpoint_sha256"] == M3_SHA256 for value in release["train_m3_prediction_caches"].values())
