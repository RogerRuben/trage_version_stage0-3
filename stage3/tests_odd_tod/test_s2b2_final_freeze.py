import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from stage3.odd_tod.network_foundation import payload_hash, sha256_file
from stage3.odd_tod.s2b2_final_freeze import (
    ALLOWED_LABELS,
    AUTHORIZED_BASE,
    EXPECTED_LABEL_COUNTS,
    PHASE_STATUS,
    adapt_route_edge_sequence,
    validate_adjudication,
)


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "stage3/docs/odd_tod/s2b"
CAL = ROOT / "stage3/output/odd_tod/s2b/calibration"
FINAL = ROOT / "stage3/output/odd_tod/s2b/final"
S2A = ROOT / "stage3/output/odd_tod/s2a"


def _json(name):
    return json.loads((DOCS / name).read_text(encoding="utf-8"))


def _parquet(name):
    return pq.read_table(FINAL / name).to_pandas()


def test_s2b2_authorized_base_and_phase_state():
    assert AUTHORIZED_BASE == "800c9e9d0747b6fe875c00e5c1d8932626b29fc0"
    report = _json("stage3_s2b_final_closure_report.json")
    assert report["authorized_base"] == AUTHORIZED_BASE
    assert report["phase_status"] == PHASE_STATUS
    assert report["s2b_complete"] is True
    assert report["s3_authorized"] is False
    assert report["next_phase_authorized"] is False


def test_human_adjudication_exact_and_hash_bound():
    path = DOCS / "stage3_s2b_completed_adjudication.csv"
    frame, descriptor = validate_adjudication(path, ROOT)
    assert len(frame) == 70
    assert set(frame["adjudication_label"]).issubset(ALLOWED_LABELS)
    assert descriptor["label_counts"] == EXPECTED_LABEL_COUNTS
    assert descriptor["sha256"] == "f1c7c0e4c4086c08c89729a2d844ff6335930a139691137746d70dc8dc43b3a0"
    assert len(descriptor["neither_cases"]) == 2
    assert all(row["reviewer_note"] for row in descriptor["neither_cases"])


def test_selected_tolerance_is_only_10m_and_baselines_not_promoted():
    complexes = _parquet("stage3_intersection_complexes.parquet")
    movements = _parquet("stage3_intersection_movements.parquet")
    membership = _parquet("stage3_intersection_node_membership.parquet")
    assert set(complexes["buffer_radius_m"]) == {10}
    assert set(complexes["tolerance_m"]) == {10}
    assert set(movements["tolerance_m"]) == {10}
    assert set(membership["tolerance_m"]) == {10}
    manifest = _json("stage3_s2b_final_release_manifest.json")
    assert manifest["selected_buffer_radius_m"] == 10
    assert manifest["rejected_baseline_radii_m"] == [5, 15, 20]
    assert manifest["spatial_candidate_overlap_condition"] == "center_distance_m <= 20"


def test_r10_source_to_final_integrity():
    source_complexes = pq.read_table(CAL / "complexes_r10.parquet").to_pandas()
    final_complexes = _parquet("stage3_intersection_complexes.parquet")
    pd.testing.assert_frame_equal(source_complexes, final_complexes.drop(columns=["buffer_radius_m"]))
    assert sha256_file(CAL / "movements_r10.parquet") == sha256_file(FINAL / "stage3_intersection_movements.parquet")
    assert sha256_file(CAL / "node_membership_r10.parquet") == sha256_file(FINAL / "stage3_intersection_node_membership.parquet")
    assert sha256_file(CAL / "movements_r10.parquet") == sha256_file(FINAL / "stage3_route_movement_lookup.parquet")


def test_semantics_are_conservative_and_poi_excluded():
    complexes = _parquet("stage3_intersection_complexes.parquet")
    movements = _parquet("stage3_intersection_movements.parquet")
    assert "UNSIGNALIZED" not in set(complexes["signal_state"])
    assert "legal_movement_count" not in complexes.columns
    assert "topological_movement_count" in complexes.columns
    assert set(movements["turn_type_semantics"]) == {"GEOMETRIC_COMPUTATIONAL_CONVENTION_NOT_TRAFFIC_LAW"}
    assert complexes["red_flags_are_qa_only"].all()
    evidence = pq.read_table(S2A / "stage3_control_evidence.parquet").to_pandas()
    assert not evidence["poi_used"].any()


def test_reverse_overlay_is_retained_not_missing_or_inserted():
    overlay = pq.read_table(S2A / "stage3_historical_direction_overlay.parquet").to_pandas()
    edges = pq.read_table(S2A / "stage3_full_network_edges.parquet", columns=["stage3_edge_uid"]).to_pandas()
    assert len(overlay) == 6502
    assert set(overlay["historical_direction_status"]) == {"HISTORICAL_DIRECTION_OVERLAY"}
    assert set(overlay["av_routability_status"]) == {"AV_ROUTABILITY_VIOLATION"}
    assert not overlay["missing_identity"].any()
    assert overlay["physical_forward_stage3_edge_uid"].notna().sum() == 6388
    assert not (set(overlay["canonical_edge_uid"]) & set(edges["stage3_edge_uid"]))


def test_edge_complex_boundary_index_is_consistent():
    index = _parquet("stage3_edge_complex_boundary_index.parquet")
    complexes = set(_parquet("stage3_intersection_complexes.parquet")["intersection_complex_uid"])
    edges = set(pq.read_table(S2A / "stage3_full_network_edges.parquet", columns=["stage3_edge_uid"]).to_pandas()["stage3_edge_uid"])
    assert len(index) == 289295
    assert set(index["boundary_role"]) == {"INCOMING", "OUTGOING", "INTERNAL"}
    assert set(index["intersection_complex_uid"]).issubset(complexes)
    assert set(index["stage3_edge_uid"]).issubset(edges)
    assert not index.duplicated(["stage3_edge_uid", "intersection_complex_uid", "boundary_role"]).any()


def test_route_adapter_matches_known_transition_and_no_transition():
    movements = _parquet("stage3_route_movement_lookup.parquet")
    complexes = _parquet("stage3_intersection_complexes.parquet")
    boundary = _parquet("stage3_edge_complex_boundary_index.parquet")
    overlay = pq.read_table(S2A / "stage3_historical_direction_overlay.parquet").to_pandas()
    known = movements.iloc[0]
    network_edges = set(pq.read_table(S2A / "stage3_full_network_edges.parquet", columns=["stage3_edge_uid"]).to_pandas()["stage3_edge_uid"])
    result = adapt_route_edge_sequence(
        [known["incoming_stage3_edge_uid"], known["outgoing_stage3_edge_uid"]],
        movements, complexes, boundary, overlay, network_edges,
    )
    assert result.iloc[0]["interface_status"] == "MATCHED_TOPOLOGICAL_MOVEMENT"
    assert result.iloc[0]["intersection_complex_uid"] == known["intersection_complex_uid"]
    incoming = boundary.loc[boundary["boundary_role"] == "INTERNAL", "stage3_edge_uid"].iloc[0]
    no_transition = adapt_route_edge_sequence([incoming], movements, complexes, boundary, overlay, network_edges)
    assert no_transition.iloc[0]["interface_status"] == "NO_COMPLEX_TRANSITION"
    empty = adapt_route_edge_sequence([], movements, complexes, boundary, overlay, network_edges)
    assert empty.empty
    assert list(empty.columns)[-1] == "interface_status"


def test_route_adapter_handles_reverse_and_unresolved_identities():
    movements = _parquet("stage3_route_movement_lookup.parquet")
    complexes = _parquet("stage3_intersection_complexes.parquet")
    boundary = _parquet("stage3_edge_complex_boundary_index.parquet")
    overlay = pq.read_table(S2A / "stage3_historical_direction_overlay.parquet").to_pandas()
    network_edges = set(pq.read_table(S2A / "stage3_full_network_edges.parquet", columns=["stage3_edge_uid"]).to_pandas()["stage3_edge_uid"])
    historical_uid = overlay.iloc[0]["canonical_edge_uid"]
    historical = adapt_route_edge_sequence([historical_uid], movements, complexes, boundary, overlay, network_edges)
    assert historical.iloc[0]["interface_status"] == "AV_ROUTABILITY_VIOLATION"
    unresolved = adapt_route_edge_sequence(["unknown_edge_identity"], movements, complexes, boundary, overlay, network_edges)
    assert unresolved.iloc[0]["interface_status"] == "UNRESOLVED_NETWORK_IDENTITY"


def test_contract_and_methodology_document_buffer_and_claim_boundaries():
    contract = (DOCS / "stage3_s2b_to_s3_contract.md").read_text(encoding="utf-8")
    method = (DOCS / "stage3_s2b_final_methodology_note.md").read_text(encoding="utf-8")
    assert "buffer radius is 10m" in contract
    assert "center distance <= 20m" in contract
    assert "topological movements, not legally certified movements" in contract
    assert "not unsignalized" in contract
    assert "POI" not in contract or "excluded" in contract
    assert "No clustering" in method


def test_scope_guards_and_evidence_are_closed():
    evidence = _json("stage3_s2b2_evidence_bundle.json")
    assert evidence["artifact_sha256"] == payload_hash(evidence)
    assert not any(evidence["scope_guards"].values())
    forbidden = [
        "tolerance_recalibration_performed", "candidate_products_recomputed", "new_tolerance_tested",
        "full_network_reexported", "speed_model_changed", "stage2_inference_performed",
        "dynamic_eqc_calibrated", "av_profile_calibrated", "test31_route_assessment_performed",
        "fallback_routing_performed", "stage4_performed", "s3_authorized", "next_phase_authorized",
    ]
    assert all(evidence["scope_guards"][key] is False for key in forbidden)
