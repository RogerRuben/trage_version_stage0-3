"""Behavioral contract for S4 Test31 original-route suitability.

The fixtures in this module are intentionally tiny.  They exercise the
scientific semantics independently of the 30,000-order production run so a
future refactor cannot accidentally turn UNKNOWN into failure, mask a known
violation, or discard simultaneous causes.
"""

from pathlib import Path
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import stage3.odd_tod.original_route_suitability as s4

from stage3.odd_tod.original_route_suitability import (
    AUTHORIZED_BASE,
    NEXT_PHASE_AUTHORIZED,
    PHASE_STATUS,
    S5_AUTHORIZED,
    TEST_DATE,
    aggregate_atomic_state,
    aggregate_reason_codes,
    audit_three_state_nestedness,
    evaluate_directional_routability,
    evaluate_dynamic_checks,
    evaluate_movement_atomic_checks,
    evaluate_speed_checks,
    evaluate_static_checks,
    finalize_route_state,
    ATOMIC_COLUMNS,
    EVIDENCE_DESCRIPTOR_PATHS,
    EXPECTED_ORDER_COUNT,
    EXPECTED_ORDER_PROFILE_COUNT,
    descriptor_mismatches,
    refresh_evidence_chain,
    _stream_atomic_reconciliation,
)
from stage3.odd_tod.network_foundation import (
    Stage3S2AError,
    parquet_descriptor,
    payload_hash,
    sha256_file,
    source_descriptor,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = ROOT / "stage3/odd_tod/original_route_suitability.py"


def _by_name(rows):
    return {row["check_name"]: row for row in rows}


def _reason_set(rows, state="INCOMPATIBLE"):
    return {
        row["reason_code"]
        for row in rows
        if row["state"] == state and row.get("reason_code")
    }


def _dynamic_caps(value=1.0):
    return {
        dimension: {metric: value for metric in ("E", "Q", "C")}
        for dimension in ("crawl", "stop", "speed_cv", "acceleration_rms")
    }


def _complete_descriptor(value=0.5):
    result = {"dynamic_complete": True}
    for dimension in ("crawl", "stop", "speed_cv", "acceleration_rms"):
        for metric in ("E", "Q", "C"):
            result[f"{dimension}_{metric}"] = value
    return result


def test_s4_phase_identity_and_scope_are_exact():
    assert AUTHORIZED_BASE == "a18dcdb42bd622273eccf2347ab6c61f4fef955f"
    assert TEST_DATE == "20161031"
    assert PHASE_STATUS == "STAGE3_S4_TEST31_ORIGINAL_ROUTE_SUITABILITY_COMPLETE"
    assert S5_AUTHORIZED is False
    assert NEXT_PHASE_AUTHORIZED is False


def test_atomic_state_reducer_uses_known_violation_precedence():
    assert aggregate_atomic_state(["COMPATIBLE", "COMPATIBLE"]) == "COMPATIBLE"
    assert aggregate_atomic_state(["COMPATIBLE", "UNKNOWN"]) == "UNKNOWN"
    assert aggregate_atomic_state(["UNKNOWN", "INCOMPATIBLE"]) == "INCOMPATIBLE"
    assert aggregate_atomic_state(["INCOMPATIBLE", "UNKNOWN", "COMPATIBLE"]) == "INCOMPATIBLE"
    assert finalize_route_state(["COMPATIBLE"]) == "FEASIBLE"
    assert finalize_route_state(["UNKNOWN"]) == "UNKNOWN"
    assert finalize_route_state(["UNKNOWN", "INCOMPATIBLE"]) == "INFEASIBLE"


def test_reason_aggregation_preserves_known_and_unknown_causes_after_failure():
    atomic = [
        {"state": "INCOMPATIBLE", "reason_code": "KNOWN_REVERSE_DIRECTION_AV_UNROUTABLE"},
        {"state": "UNKNOWN", "reason_code": "UNRESOLVED_ROUTE_IDENTITY"},
        {"state": "INCOMPATIBLE", "reason_code": "SPEED_DOMAIN_CAP_EXCEEDED"},
        {"state": "UNKNOWN", "reason_code": "DYNAMIC_ROUTE_INCOMPLETE"},
    ]
    result = aggregate_reason_codes(atomic)
    assert result["known_violation_reason_codes"] == [
        "KNOWN_REVERSE_DIRECTION_AV_UNROUTABLE",
        "SPEED_DOMAIN_CAP_EXCEEDED",
    ]
    assert result["unknown_reason_codes"] == [
        "DYNAMIC_ROUTE_INCOMPLETE",
        "UNRESOLVED_ROUTE_IDENTITY",
    ]
    assert result["known_violation_count"] == 2
    assert result["unknown_requirement_count"] == 2


def test_reason_counts_are_distinct_but_atomic_occurrences_remain_available():
    atomic = [
        {"state": "INCOMPATIBLE", "reason_code": "SPEED_DOMAIN_CAP_EXCEEDED", "route_sequence": 1},
        {"state": "INCOMPATIBLE", "reason_code": "SPEED_DOMAIN_CAP_EXCEEDED", "route_sequence": 7},
        {"state": "UNKNOWN", "reason_code": "SPEED_DOMAIN_UNKNOWN", "route_sequence": 8},
        {"state": "UNKNOWN", "reason_code": "SPEED_DOMAIN_UNKNOWN", "route_sequence": 9},
    ]
    result = aggregate_reason_codes(atomic)
    assert result["known_violation_count"] == 1
    assert result["unknown_requirement_count"] == 1
    # Cause aggregation must not mutate or deduplicate the scientific atomic product.
    assert len(atomic) == 4


def test_direction_reverse_is_known_violation_and_unresolved_is_unknown():
    typed = pd.DataFrame(
        {
            "route_sequence": [0, 1, 2],
            "route_token_type": [
                "FULL_NETWORK_EDGE",
                "HISTORICAL_REVERSE_OVERLAY",
                "UNRESOLVED",
            ],
            "resolved_stage3_edge_uid": ["s3e_ok", None, None],
            "physical_forward_stage3_edge_uid": [None, "s3e_provenance_only", None],
        }
    )
    rows = evaluate_directional_routability(typed)
    assert len(rows) == 1
    assert aggregate_atomic_state(row["state"] for row in rows) == "INCOMPATIBLE"
    assert _reason_set(rows) == {"KNOWN_REVERSE_DIRECTION_AV_UNROUTABLE"}
    assert not _reason_set(rows, "UNKNOWN")
    assert any(row.get("stage3_edge_uid") is None for row in rows if row["route_sequence"] == 1)


def test_direction_all_unresolved_is_unknown_not_infeasible():
    typed = pd.DataFrame(
        {
            "route_sequence": [0],
            "route_token_type": ["UNRESOLVED"],
            "resolved_stage3_edge_uid": [None],
            "physical_forward_stage3_edge_uid": [None],
        }
    )
    rows = evaluate_directional_routability(typed)
    assert aggregate_atomic_state(row["state"] for row in rows) == "UNKNOWN"
    assert not _reason_set(rows)


def test_speed_uses_only_resolved_full_network_edges_and_never_forward_projection():
    typed = pd.DataFrame(
        {
            "route_sequence": [0, 1, 2],
            "route_token_type": ["FULL_NETWORK_EDGE", "HISTORICAL_REVERSE_OVERLAY", "UNRESOLVED"],
            "resolved_stage3_edge_uid": ["s3e_real", None, None],
            "physical_forward_stage3_edge_uid": [None, "s3e_must_not_be_used", None],
        }
    )
    speed = pd.DataFrame(
        {
            "stage3_edge_uid": ["s3e_real", "s3e_must_not_be_used"],
            "speed_domain_value_kmh": [90.0, 20.0],
        }
    )
    rows = evaluate_speed_checks(typed, speed, "C", 60.0)
    assert _reason_set(rows) == {"SPEED_DOMAIN_CAP_EXCEEDED"}
    assert {row.get("stage3_edge_uid") for row in rows} <= {"s3e_real", None}


def test_speed_known_exceedance_precedes_another_resolved_edge_unknown():
    typed = pd.DataFrame(
        {
            "route_sequence": [0, 1],
            "route_token_type": ["FULL_NETWORK_EDGE", "FULL_NETWORK_EDGE"],
            "resolved_stage3_edge_uid": ["known", "missing"],
            "physical_forward_stage3_edge_uid": [None, None],
        }
    )
    speed = pd.DataFrame({"stage3_edge_uid": ["known"], "speed_domain_value_kmh": [90.0]})
    rows = evaluate_speed_checks(typed, speed, "C", 60.0)
    assert aggregate_atomic_state(row["state"] for row in rows) == "INCOMPATIBLE"
    assert _reason_set(rows) == {"SPEED_DOMAIN_CAP_EXCEEDED"}
    assert _reason_set(rows, "UNKNOWN") == {"SPEED_DOMAIN_UNKNOWN"}


def test_speed_with_no_full_network_edge_is_vacuously_compatible():
    typed = pd.DataFrame(
        {
            "route_sequence": [0],
            "route_token_type": ["HISTORICAL_REVERSE_OVERLAY"],
            "resolved_stage3_edge_uid": [None],
            "physical_forward_stage3_edge_uid": ["forward_provenance"],
        }
    )
    speed = pd.DataFrame(
        {"stage3_edge_uid": ["forward_provenance"], "speed_domain_value_kmh": [200.0]}
    )
    rows = evaluate_speed_checks(typed, speed, "C", 60.0)
    assert aggregate_atomic_state(row["state"] for row in rows) == "COMPATIBLE"


def test_static_checks_are_four_independent_noncompensatory_dimensions():
    complex_row = {
        "intersection_complex_uid": "c1",
        "A_c": 5.0,
        "M_c": 17.0,
        "D_c": 4.0,
        "L_c": 40.0,
    }
    caps = {"A_c": 4.0, "M_c": 16.0, "D_c": 3.0, "L_c": 34.0}
    rows = evaluate_static_checks(complex_row, "M", caps)
    assert len(rows) == 4
    assert _reason_set(rows) == {
        "STATIC_A_CAP_EXCEEDED",
        "STATIC_M_CAP_EXCEEDED",
        "STATIC_D_CAP_EXCEEDED",
        "STATIC_L_CAP_EXCEEDED",
    }


def test_static_missing_metric_is_unknown_without_imputation():
    complex_row = {
        "intersection_complex_uid": "c1",
        "A_c": 3.0,
        "M_c": 8.0,
        "D_c": np.nan,
        "L_c": 9.0,
    }
    rows = _by_name(
        evaluate_static_checks(
            complex_row,
            "C",
            {"A_c": 4.0, "M_c": 9.0, "D_c": 2.0, "L_c": 10.0},
        )
    )
    assert rows["STATIC_D"]["state"] == "UNKNOWN"
    assert rows["STATIC_D"]["reason_code"] == "STATIC_METRIC_UNKNOWN"
    assert all(rows[name]["state"] == "COMPATIBLE" for name in ("STATIC_A", "STATIC_M", "STATIC_L"))


def test_static_route_cannot_average_a_failing_complex_with_an_easy_one():
    caps = {"A_c": 4.0, "M_c": 9.0, "D_c": 2.0, "L_c": 10.0}
    easy = {"intersection_complex_uid": "easy", "A_c": 1, "M_c": 1, "D_c": 1, "L_c": 1}
    hard = {"intersection_complex_uid": "hard", "A_c": 5, "M_c": 1, "D_c": 1, "L_c": 1}
    rows = evaluate_static_checks(easy, "C", caps) + evaluate_static_checks(hard, "C", caps)
    assert aggregate_atomic_state(row["state"] for row in rows) == "INCOMPATIBLE"
    assert _reason_set(rows) == {"STATIC_A_CAP_EXCEEDED"}


def test_movement_lookup_absence_is_unknown_not_fabricated_infeasible():
    encounter = {
        "movement_occurrence_index": 0,
        "intersection_complex_uid": "c",
        "movement_lookup_status": "UNRESOLVED_MOVEMENT_LOOKUP",
        "route_turn_type": None,
        "signal_state": "UNKNOWN_CONTROL",
        "roundabout_evidence_present": False,
        "restriction_enforcement_certified": False,
        "movement_legality_state": "UNKNOWN",
    }
    rows = evaluate_movement_atomic_checks(encounter, "C")
    assert aggregate_atomic_state(row["state"] for row in rows) == "UNKNOWN"
    assert "MOVEMENT_LOOKUP_UNRESOLVED" in _reason_set(rows, "UNKNOWN")
    assert not _reason_set(rows)


def test_conservative_left_control_rules_and_more_capable_left_rules():
    base = {
        "movement_occurrence_index": 0,
        "intersection_complex_uid": "c",
        "movement_lookup_status": "MATCHED_TOPOLOGICAL_MOVEMENT",
        "route_turn_type": "LEFT",
        "roundabout_evidence_present": False,
        "restriction_enforcement_certified": False,
        "movement_legality_state": "UNKNOWN",
    }
    signal = evaluate_movement_atomic_checks({**base, "signal_state": "SIGNALIZED"}, "C")
    stop = evaluate_movement_atomic_checks({**base, "signal_state": "STOP_OR_YIELD_CONTROLLED"}, "C")
    unknown = evaluate_movement_atomic_checks({**base, "signal_state": "UNKNOWN_CONTROL"}, "C")
    assert aggregate_atomic_state(row["state"] for row in signal) == "COMPATIBLE"
    assert "CONSERVATIVE_LEFT_STOP_YIELD_INCOMPATIBLE" in _reason_set(stop)
    assert "CONSERVATIVE_LEFT_UNKNOWN_CONTROL" in _reason_set(unknown, "UNKNOWN")
    for profile in ("M", "A"):
        rows = evaluate_movement_atomic_checks({**base, "signal_state": "UNKNOWN_CONTROL"}, profile)
        assert aggregate_atomic_state(row["state"] for row in rows) == "COMPATIBLE"


def test_movement_does_not_short_circuit_and_preserves_three_simultaneous_causes():
    encounter = {
        "movement_occurrence_index": 3,
        "intersection_complex_uid": "c",
        "movement_lookup_status": "MATCHED_TOPOLOGICAL_MOVEMENT",
        "route_turn_type": "UTURN",
        "signal_state": "UNKNOWN_CONTROL",
        "roundabout_evidence_present": True,
        "restriction_enforcement_certified": True,
        "movement_legality_state": "CERTIFIED_PROHIBITED",
    }
    rows = evaluate_movement_atomic_checks(encounter, "C")
    assert _reason_set(rows) == {
        "UTURN_PROFILE_INCOMPATIBLE",
        "CONSERVATIVE_ROUNDABOUT_INCOMPATIBLE",
        "CERTIFIED_MOVEMENT_PROHIBITION",
    }


def test_noncertified_restriction_is_not_treated_as_prohibition():
    encounter = {
        "movement_occurrence_index": 0,
        "intersection_complex_uid": "c",
        "movement_lookup_status": "MATCHED_TOPOLOGICAL_MOVEMENT",
        "route_turn_type": "STRAIGHT",
        "signal_state": "UNKNOWN_CONTROL",
        "roundabout_evidence_present": False,
        "restriction_enforcement_certified": False,
        "movement_legality_state": "NOT_CERTIFIED",
    }
    rows = _by_name(evaluate_movement_atomic_checks(encounter, "A"))
    assert rows["RESTRICTION"]["state"] == "COMPATIBLE"
    assert rows["RESTRICTION"]["reason_code"] is None


def test_uturn_and_roundabout_rules_are_profile_nested():
    base = {
        "movement_occurrence_index": 0,
        "intersection_complex_uid": "c",
        "movement_lookup_status": "MATCHED_TOPOLOGICAL_MOVEMENT",
        "route_turn_type": "UTURN",
        "signal_state": "UNKNOWN_CONTROL",
        "roundabout_evidence_present": True,
        "restriction_enforcement_certified": False,
        "movement_legality_state": "UNKNOWN",
    }
    states = {
        profile: aggregate_atomic_state(
            row["state"] for row in evaluate_movement_atomic_checks(base, profile)
        )
        for profile in ("C", "M", "A")
    }
    assert states == {"C": "INCOMPATIBLE", "M": "INCOMPATIBLE", "A": "COMPATIBLE"}


def test_dynamic_incomplete_route_emits_all_twelve_unknown_checks_and_no_pseudo_values():
    descriptor = {
        "dynamic_complete": False,
        "missing_dynamic_token_count": 1,
        "first_missing_dynamic_route_sequence": 4,
        "missing_dynamic_field_mask": "pred_stop",
    }
    rows = evaluate_dynamic_checks(descriptor, "C", _dynamic_caps())
    assert len(rows) == 12
    assert {row["state"] for row in rows} == {"UNKNOWN"}
    assert _reason_set(rows, "UNKNOWN") == {"DYNAMIC_ROUTE_INCOMPLETE"}
    assert all(row.get("observed_value") is None for row in rows)


def test_dynamic_checks_are_all_twelve_noncompensatory_checks():
    descriptor = _complete_descriptor(0.5)
    descriptor["crawl_E"] = 1.01
    rows = evaluate_dynamic_checks(descriptor, "C", _dynamic_caps(1.0))
    assert len(rows) == 12
    assert aggregate_atomic_state(row["state"] for row in rows) == "INCOMPATIBLE"
    assert _reason_set(rows) == {"DYNAMIC_CRAWL_E_CAP_EXCEEDED"}
    assert sum(row["state"] == "COMPATIBLE" for row in rows) == 11


def test_dynamic_equal_to_cap_is_compatible():
    rows = evaluate_dynamic_checks(_complete_descriptor(1.0), "A", _dynamic_caps(1.0))
    assert len(rows) == 12
    assert {row["state"] for row in rows} == {"COMPATIBLE"}


def test_three_state_nestedness_allows_capability_relaxation_transitions():
    rows = []
    for order_id, states in {
        "f": ("FEASIBLE", "FEASIBLE", "FEASIBLE"),
        "u": ("UNKNOWN", "UNKNOWN", "FEASIBLE"),
        "i": ("INFEASIBLE", "UNKNOWN", "FEASIBLE"),
        "ii": ("INFEASIBLE", "INFEASIBLE", "UNKNOWN"),
    }.items():
        rows.extend(
            {"order_id": order_id, "profile_id": profile, "original_route_state": state}
            for profile, state in zip(("C", "M", "A"), states)
        )
    audit = audit_three_state_nestedness(pd.DataFrame(rows))
    assert audit["pass"] is True
    assert audit["capability_regression_count"] == 0
    assert audit["feasible_to_unknown_count"] == 0


def test_three_state_nestedness_rejects_feasible_to_unknown_or_infeasible():
    rows = []
    for order_id, states in {
        "f_to_u": ("FEASIBLE", "UNKNOWN", "FEASIBLE"),
        "f_to_i": ("FEASIBLE", "INFEASIBLE", "INFEASIBLE"),
        "u_to_i": ("UNKNOWN", "INFEASIBLE", "FEASIBLE"),
    }.items():
        rows.extend(
            {"order_id": order_id, "profile_id": profile, "original_route_state": state}
            for profile, state in zip(("C", "M", "A"), states)
        )
    audit = audit_three_state_nestedness(pd.DataFrame(rows))
    assert audit["pass"] is False
    assert audit["capability_regression_count"] == 3
    assert audit["feasible_to_unknown_count"] == 1


def test_source_contains_no_fallback_or_route_search_implementation():
    source = SOURCE_PATH.read_text(encoding="utf-8").lower()
    forbidden = (
        "k_shortest",
        "shortest_path(",
        "valhalla_route",
        "route_search(",
        "fallback_candidate",
        "stage4_dispatch(",
    )
    assert not [token for token in forbidden if token in source]


def test_source_does_not_fit_test31_cdf_or_profile_thresholds():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert "weighted_mid_cdf_reference(" not in source
    assert "quantile_higher(" not in source
    assert "dynamic_caps(" not in source
    assert "static_caps(" not in source


def test_source_descriptors_detect_stale_hashes(tmp_path):
    """The primitive used by the bundle must make a stale descriptor obvious."""
    artifact = tmp_path / "artifact.json"
    artifact.write_text('{"version":1}\n', encoding="utf-8")
    descriptor = source_descriptor(artifact, tmp_path)
    assert descriptor["sha256"] == sha256_file(artifact)
    artifact.write_text('{"version":2}\n', encoding="utf-8")
    assert descriptor["sha256"] != sha256_file(artifact)
    refreshed = source_descriptor(artifact, tmp_path)
    assert refreshed["sha256"] == sha256_file(artifact)
    assert refreshed["size_bytes"] == artifact.stat().st_size


def test_parquet_descriptor_binds_row_count_schema_and_detects_staleness(tmp_path):
    artifact = tmp_path / "product.parquet"
    pd.DataFrame({"order_id": ["a"], "state": ["FEASIBLE"]}).to_parquet(artifact, index=False)
    descriptor = parquet_descriptor(artifact, tmp_path)
    assert descriptor["row_count"] == 1
    assert descriptor["sha256"] == sha256_file(artifact)
    assert descriptor["schema"] == str(pq.ParquetFile(artifact).schema_arrow)
    pd.DataFrame({"order_id": ["a", "b"], "state": ["FEASIBLE", "UNKNOWN"]}).to_parquet(artifact, index=False)
    assert descriptor["sha256"] != sha256_file(artifact)
    refreshed = parquet_descriptor(artifact, tmp_path)
    assert refreshed["row_count"] == 2
    assert refreshed["schema_sha256"] == descriptor["schema_sha256"]


def test_evidence_bundle_lists_every_mutable_descriptor_that_must_be_refreshed():
    """Guard the provenance chain even when the large products remain ignored."""
    source = SOURCE_PATH.read_text(encoding="utf-8")
    for label in (
        '"release_manifest"',
        '"test_evidence"',
        '"suitability_summary"',
        '"prediction_manifest"',
        '"input_manifest"',
    ):
        assert label in source
    # Verification must inspect the evidence bundle itself, not only release
    # descriptors; otherwise update_test_evidence can silently stale its SHA.
    verify_body = source[source.index("def verify_s4"): source.index("def update_test_evidence")]
    assert 'read_json(docs / "stage3_s4_evidence_bundle.json")' in verify_body
    assert "descriptor_mismatches(root, evidence_descriptors)" in verify_body


def test_evidence_descriptor_staleness_is_detected_and_refresh_restores_chain(tmp_path):
    """A post-finalize test-evidence update must refresh release and bundle SHA links."""
    docs = tmp_path / "stage3/docs/odd_tod/s4"
    output = tmp_path / "stage3/output/odd_tod/s4"
    docs.mkdir(parents=True)
    output.mkdir(parents=True)

    test_path = docs / "stage3_s4_test_evidence.json"
    release_path = docs / "stage3_s4_release_manifest.json"
    bundle_path = docs / "stage3_s4_evidence_bundle.json"
    test_path.write_text('{"tests":"initial"}\n', encoding="utf-8")
    (output / "test31_suitability_summary.json").write_text('{"orders":30000}\n', encoding="utf-8")
    (output / "test31_m3_predictions.json").write_text('{"model_id":"M3"}\n', encoding="utf-8")
    (output / "test31_input_manifest.json").write_text('{"date":"20161031"}\n', encoding="utf-8")
    release_path.write_text(
        json.dumps({"phase_status": PHASE_STATUS, "reports": {}, "artifact_sha256": "stale"}) + "\n",
        encoding="utf-8",
    )

    first = refresh_evidence_chain(tmp_path)
    first_descriptors = {name: first[name] for name in EVIDENCE_DESCRIPTOR_PATHS}
    assert descriptor_mismatches(tmp_path, first_descriptors) == []
    assert first["artifact_sha256"] == payload_hash(first)
    assert json.loads(bundle_path.read_text(encoding="utf-8")) == first

    # Simulate update_test_evidence changing the file after finalization.
    test_path.write_text('{"tests":"updated"}\n', encoding="utf-8")
    assert descriptor_mismatches(tmp_path, first_descriptors) == ["descriptor_hash:test_evidence"]

    refreshed = refresh_evidence_chain(tmp_path)
    refreshed_descriptors = {name: refreshed[name] for name in EVIDENCE_DESCRIPTOR_PATHS}
    assert descriptor_mismatches(tmp_path, refreshed_descriptors) == []
    assert refreshed["artifact_sha256"] == payload_hash(refreshed)
    assert refreshed["test_evidence"]["sha256"] == sha256_file(test_path)

    release = json.loads(release_path.read_text(encoding="utf-8"))
    test_label = test_path.relative_to(tmp_path).as_posix()
    assert release["reports"][test_label]["sha256"] == sha256_file(test_path)
    assert release["artifact_sha256"] == payload_hash(release)
    assert refreshed["release_manifest"]["sha256"] == sha256_file(release_path)


def test_descriptor_mismatch_reports_bad_path_missing_file_and_hash(tmp_path):
    good = tmp_path / "good.json"
    good.write_text("{}\n", encoding="utf-8")
    descriptors = {
        "bad_path": {},
        "missing": {"path": "missing.json", "sha256": "unused"},
        "stale": {"path": "good.json", "sha256": "not-the-current-hash"},
    }
    assert descriptor_mismatches(tmp_path, descriptors) == [
        "descriptor_path:bad_path",
        "descriptor_missing:missing",
        "descriptor_hash:stale",
    ]


def test_atomic_parquet_contract_is_fixed_and_cheaply_auditable(tmp_path):
    """Metadata/schema checks must not require loading the large atomic table."""
    artifact = tmp_path / "atomic.parquet"
    arrays = []
    for field in (
        pa.field("date", pa.string()),
        pa.field("order_id", pa.string()),
        pa.field("profile_id", pa.string()),
        pa.field("check_family", pa.string()),
        pa.field("check_name", pa.string()),
        pa.field("state", pa.string()),
        pa.field("observed_value", pa.float64()),
        pa.field("cap_value", pa.float64()),
        pa.field("evidence_id", pa.string()),
        pa.field("route_sequence", pa.int64()),
        pa.field("stage3_edge_uid", pa.string()),
        pa.field("intersection_complex_uid", pa.string()),
        pa.field("movement_occurrence_index", pa.int64()),
        pa.field("reason_code", pa.string()),
    ):
        arrays.append(pa.array([], type=field.type))
    pq.write_table(pa.Table.from_arrays(arrays, names=list(ATOMIC_COLUMNS)), artifact)
    parquet = pq.ParquetFile(artifact)
    assert parquet.metadata.num_rows == 0
    assert tuple(parquet.schema_arrow.names) == tuple(ATOMIC_COLUMNS)


def test_full_product_count_constants_are_frozen():
    assert EXPECTED_ORDER_COUNT == 30_000
    assert EXPECTED_ORDER_PROFILE_COUNT == 90_000


def test_verify_source_uses_parquet_metadata_for_prediction_schema():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    verify_body = source[source.index("def verify_s4"): source.index("def update_test_evidence")]
    assert "pq.ParquetFile" in verify_body
    assert "schema_arrow.names" in verify_body
    assert "realized target in prediction schema" in verify_body


def test_external_evidence_id_sort_detects_duplicate_across_batches(monkeypatch, tmp_path):
    """The exact distinct reducer must detect a duplicate split across chunks."""
    def record_batch(evidence_ids):
        frame = pd.DataFrame(
            {
                "order_id": ["o"] * len(evidence_ids),
                "profile_id": ["C"] * len(evidence_ids),
                "check_family": ["DIRECTION"] * len(evidence_ids),
                "state": ["COMPATIBLE"] * len(evidence_ids),
                "evidence_id": evidence_ids,
                "reason_code": pd.array([None] * len(evidence_ids), dtype="string"),
            }
        )
        return pa.Table.from_pandas(frame, preserve_index=False).to_batches()[0]

    class FakeParquet:
        metadata = SimpleNamespace(num_rows=4)

        def iter_batches(self, *, batch_size, columns):
            assert batch_size == 250_000
            assert columns == [
                "order_id", "profile_id", "check_family", "state", "evidence_id", "reason_code"
            ]
            yield record_batch(["shared", "z"])
            yield record_batch(["a", "shared"])

    monkeypatch.setattr(s4.pq, "ParquetFile", lambda _: FakeParquet())
    suitability = pd.DataFrame(
        {
            "order_id": ["o"],
            "profile_id": ["C"],
            **{column: ["COMPATIBLE"] for column in s4.ATOMIC_FAMILY_STATE_COLUMNS.values()},
            "dynamic_state": ["COMPATIBLE"],
            "original_route_state": ["FEASIBLE"],
            "known_violation_reason_codes": ["[]"],
            "unknown_reason_codes": ["[]"],
            "known_violation_count": [0],
            "unknown_requirement_count": [0],
            "known_violation_atomic_count": [0],
            "unknown_atomic_count": [0],
        }
    )
    atomic_path = tmp_path / "atomic.parquet"
    result = s4._stream_atomic_reconciliation(atomic_path, suitability)
    assert result["evidence_id_unique_count"] == 3
    assert result["duplicate_evidence_id_count"] == 1
    assert result["evidence_id_distinct_method"] == "memory_bounded_external_merge_sort_exact"
    assert not list(tmp_path.glob(".s4_evidence_id_sort_*"))


def _stub_verify_environment(monkeypatch, tmp_path, test_statuses):
    """Install a one-order verification world without reading production artifacts."""
    monkeypatch.setattr(s4, "EXPECTED_ORDER_COUNT", 1)
    monkeypatch.setattr(s4, "EXPECTED_ORDER_PROFILE_COUNT", 3)
    suitability = pd.DataFrame(
        {
            "order_id": ["o", "o", "o"],
            "profile_id": ["C", "M", "A"],
            **{column: ["COMPATIBLE"] * 3 for column in s4.ATOMIC_FAMILY_STATE_COLUMNS.values()},
            "dynamic_state": ["COMPATIBLE"] * 3,
            "original_route_state": ["FEASIBLE"] * 3,
            "known_violation_reason_codes": ["[]"] * 3,
            "unknown_reason_codes": ["[]"] * 3,
            "known_violation_count": [0] * 3,
            "unknown_requirement_count": [0] * 3,
            "known_violation_atomic_count": [0] * 3,
            "unknown_atomic_count": [0] * 3,
        }
    )
    release = {
        "phase_status": PHASE_STATUS,
        "artifact_sha256": "payload",
        "frozen_inputs": {},
        "products": {},
        "reports": {},
    }
    evidence = {"artifact_sha256": "payload"}
    summary = {
        "artifact_sha256": "payload",
        "frozen_hashes_before": {"profile": "file-sha", "cdf": "file-sha"},
    }
    test_evidence = {
        "artifact_sha256": "payload",
        **{name: {"status": status} for name, status in test_statuses.items()},
    }
    input_manifest = {
        "artifact_sha256": "payload",
        "route_source": {},
        "route_source_manifest": {},
    }
    prediction_manifest = {
        "artifact_sha256": "payload",
        "checkpoint_sha256": s4.M3_SHA256,
        "prediction_sha256": "file-sha",
        "route_sha256": "file-sha",
        "row_count": 1,
        "order_count": 1,
        "decision_time_only": True,
        "predicted_progression_only": True,
        "realized_future_time_used": False,
        "realized_target_columns_persisted": False,
        "prediction_only_forward": True,
        "target_arrays_constructed": False,
        "loss_or_metric_path_called": False,
        "input_bindings": {
            name: {"sha256": "file-sha"}
            for name in ("checkpoint", "model_manifest", "feature", "static", "support", "route")
        },
    }
    payloads = {
        "stage3_s4_release_manifest.json": release,
        "stage3_s4_evidence_bundle.json": evidence,
        "test31_suitability_summary.json": summary,
        "stage3_s4_test_evidence.json": test_evidence,
        "test31_input_manifest.json": input_manifest,
        "test31_m3_predictions.json": prediction_manifest,
    }
    monkeypatch.setattr(s4, "read_json", lambda path: payloads[Path(path).name])
    monkeypatch.setattr(s4, "payload_hash", lambda payload: "payload")
    monkeypatch.setattr(s4, "sha256_file", lambda path: "file-sha")
    monkeypatch.setattr(s4, "descriptor_mismatches", lambda root, descriptors: [])
    monkeypatch.setattr(
        s4,
        "refresh_evidence_chain",
        lambda root, **kwargs: evidence,
    )
    monkeypatch.setattr(s4.pd, "read_parquet", lambda *args, **kwargs: suitability.copy())
    monkeypatch.setattr(
        s4,
        "audit_three_state_nestedness",
        lambda frame: {"status": "PASS", "pass": True},
    )

    class FakeParquet:
        def __init__(self, path):
            self.is_atomic = Path(path).name == "test31_original_route_atomic_checks.parquet"
            self.schema_arrow = SimpleNamespace(
                names=list(s4.ATOMIC_COLUMNS) if self.is_atomic else ["pred_crawl"]
            )
            self.metadata = SimpleNamespace(num_rows=4 if self.is_atomic else 1)

    monkeypatch.setattr(s4.pq, "ParquetFile", FakeParquet)
    reconciliation = {
        "duplicate_evidence_id_count": 0,
        "null_evidence_id_count": 0,
        "invalid_state_count": 0,
        "invalid_reason_count": 0,
        "final_state_mismatch_count": 0,
        "known_reason_set_mismatch_count": 0,
        "unknown_reason_set_mismatch_count": 0,
        "known_distinct_count_mismatch_count": 0,
        "unknown_distinct_count_mismatch_count": 0,
        "known_atomic_count_mismatch_count": 0,
        "unknown_atomic_count_mismatch_count": 0,
        "family_state_mismatch_counts": {"dynamic_state": 0},
    }
    monkeypatch.setattr(s4, "_stream_atomic_reconciliation", lambda path, frame: reconciliation)
    written = {}
    monkeypatch.setattr(s4, "atomic_json", lambda path, payload: written.update({Path(path).name: payload}))
    return written


@pytest.mark.parametrize("status", ["PENDING_FINAL_RECORD", "FAIL"])
def test_verify_rejects_pending_or_failed_test_evidence(monkeypatch, tmp_path, status):
    written = _stub_verify_environment(
        monkeypatch,
        tmp_path,
        {"focused_tests": status, "full_tests": "PASS", "compileall": "PASS"},
    )
    with pytest.raises(Stage3S2AError, match="test evidence not PASS:focused_tests"):
        s4.verify_s4(tmp_path)
    result = written["stage3_s4_evidence_verification.json"]
    assert result["status"] == "FAIL"
    assert f"test evidence not PASS:focused_tests" in result["failures"]


def test_run_stops_at_awaiting_test_evidence_without_calling_verify(
    monkeypatch, tmp_path, capsys
):
    calls = []
    monkeypatch.setattr(s4, "prepare_test31", lambda root: calls.append("prepare"))
    monkeypatch.setattr(s4, "assess_test31", lambda root: calls.append("assess"))
    monkeypatch.setattr(s4, "finalize_s4", lambda root: calls.append("finalize"))
    monkeypatch.setattr(
        s4,
        "verify_s4",
        lambda root: (_ for _ in ()).throw(AssertionError("run must not fabricate verification")),
    )
    import stage3.odd_tod.s4_inference as inference

    monkeypatch.setattr(
        inference,
        "build_test31_predictions",
        lambda root, batch_size: calls.append(("infer", batch_size)),
    )
    assert s4.main(["--root", str(tmp_path), "run", "--batch-size", "7"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert calls == ["prepare", ("infer", 7), "assess", "finalize"]
    assert result["status"] == "AWAITING_TEST_EVIDENCE"
    assert result["s5_authorized"] is False
    assert result["next_phase_authorized"] is False


def test_stream_atomic_reconciliation_reduces_exact_states_reasons_and_ids(tmp_path):
    rows = [
        {
            "date": TEST_DATE, "order_id": "o", "profile_id": "C",
            "check_family": "DIRECTION", "check_name": "directional_routability",
            "state": "INCOMPATIBLE", "observed_value": 1.0, "cap_value": None,
            "evidence_id": "direction|o|C", "route_sequence": 2,
            "stage3_edge_uid": None, "intersection_complex_uid": None,
            "movement_occurrence_index": None,
            "reason_code": "KNOWN_REVERSE_DIRECTION_AV_UNROUTABLE",
        },
        {
            "date": TEST_DATE, "order_id": "o", "profile_id": "C",
            "check_family": "DYNAMIC_CRAWL", "check_name": "crawl_E",
            "state": "UNKNOWN", "observed_value": None, "cap_value": 0.5,
            "evidence_id": "dynamic|o|C|crawl_E", "route_sequence": None,
            "stage3_edge_uid": None, "intersection_complex_uid": None,
            "movement_occurrence_index": None,
            "reason_code": "DYNAMIC_ROUTE_INCOMPLETE",
        },
    ]
    atomic = tmp_path / "atomic.parquet"
    pd.DataFrame(rows).to_parquet(atomic, index=False)
    suitability = pd.DataFrame({
        "order_id": ["o"], "profile_id": ["C"],
        **{column: ["COMPATIBLE"] for column in (
            "speed_state", "static_A_state", "static_M_state", "static_D_state",
            "static_L_state", "movement_state", "control_state", "roundabout_state",
            "restriction_state",
        )},
        "directional_routability_state": ["INCOMPATIBLE"],
        "dynamic_state": ["UNKNOWN"], "original_route_state": ["INFEASIBLE"],
        "known_violation_reason_codes": ['["KNOWN_REVERSE_DIRECTION_AV_UNROUTABLE"]'],
        "unknown_reason_codes": ['["DYNAMIC_ROUTE_INCOMPLETE"]'],
        "known_violation_count": [1], "unknown_requirement_count": [1],
        "known_violation_atomic_count": [1], "unknown_atomic_count": [1],
    })
    audit = _stream_atomic_reconciliation(atomic, suitability)
    assert audit["duplicate_evidence_id_count"] == 0
    assert audit["evidence_id_unique_count"] == 2
    assert audit["final_state_mismatch_count"] == 0
    assert not any(audit["family_state_mismatch_counts"].values())
    assert audit["known_reason_set_mismatch_count"] == 0
    assert audit["unknown_reason_set_mismatch_count"] == 0
