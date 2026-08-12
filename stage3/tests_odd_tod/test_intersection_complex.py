from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from stage3.odd_tod.intersection_complex import (
    PHASE_STATUS,
    complex_uid,
    load_config,
    signed_turn,
    turn_type,
    verify_evidence,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "stage3/config/stage3_s2b_intersection_complex.json"
DOCS = ROOT / "stage3/docs/odd_tod/s2b"
OUTPUT = ROOT / "stage3/output/odd_tod/s2b/calibration"


def _config(): return load_config(CONFIG, ROOT)
def _read(name): return pq.read_table(OUTPUT / name).to_pandas()
def _comparison(): return json.loads((DOCS / "stage3_s2b_tolerance_comparison.json").read_text(encoding="utf-8"))


def test_s2b_uses_full_network_only() -> None:
    evidence = json.loads((DOCS / "stage3_s2b1_evidence_bundle.json").read_text(encoding="utf-8"))
    assert evidence["counts"]["full_nodes"] == 89_607
    assert evidence["counts"]["full_edges"] == 209_454
    assert evidence["inputs"]["edges"]["path"].endswith("stage3_full_network_edges.parquet")


def test_observed_subnetwork_not_used_as_topology_base() -> None:
    config = _config()
    assert "observed_mapping" in config["paths"]
    assert config["paths"]["edges"].endswith("stage3_full_network_edges.parquet")
    assert _comparison()["junction_candidate_count"] > 25_382


def test_candidate_tolerances_exactly_5_10_15_20() -> None:
    assert _config()["candidate_tolerances_m"] == [5, 10, 15, 20]
    assert [row["tolerance_m"] for row in _comparison()["tolerance_metrics"]] == [5, 10, 15, 20]


def test_metric_crs_epsg32649() -> None:
    assert _config()["metric_crs"] == "EPSG:32649"


def test_candidate_node_detection_deterministic() -> None:
    candidates = _read("junction_candidates.parquet")
    assert candidates["stage3_node_uid"].is_monotonic_increasing
    assert candidates["stage3_node_uid"].is_unique
    assert _comparison()["candidate_rule_trigger_counts"]


def test_complex_identity_row_order_independent() -> None:
    assert complex_uid(10, ["b", "a"]) == complex_uid(10, ["a", "b"])


def test_complex_membership_unique() -> None:
    candidate_count = _comparison()["junction_candidate_count"]
    for tolerance in (5, 10, 15, 20):
        membership = _read(f"node_membership_r{tolerance:02d}.parquet")
        candidates = membership[membership["candidate_node"]]
        assert len(candidates) == candidate_count
        assert candidates["stage3_node_uid"].is_unique
        assert not membership["stage3_node_uid"].duplicated().any()


def test_spatially_close_disconnected_nodes_not_merged() -> None:
    for tolerance in (5, 10, 15, 20):
        complexes = _read(f"complexes_r{tolerance:02d}.parquet")
        assert not complexes["RED_FLAG_DISCONNECTED_MERGE"].any()


def test_layer_bridge_tunnel_guards_are_auditable() -> None:
    config = _config()
    assert "grade separation" in config["selection_rule"][2]
    for tolerance in (5, 10, 15, 20):
        complexes = _read(f"complexes_r{tolerance:02d}.parquet")
        for column in ("RED_FLAG_LAYER_CONFLICT", "RED_FLAG_BRIDGE_SURFACE_MIX", "RED_FLAG_TUNNEL_SURFACE_MIX"):
            assert column in complexes


def test_roundabout_semantic_complex_and_no_unrelated_overmerge() -> None:
    counts = []
    for tolerance in (5, 10, 15, 20):
        complexes = _read(f"complexes_r{tolerance:02d}.parquet")
        counts.append(int(complexes["roundabout_evidence_present"].sum()))
        assert not complexes["RED_FLAG_ROUNDABOUT_MIXED_WITH_UNRELATED_JUNCTION"].any()
    assert len(set(counts)) == 1 and counts[0] > 0
    stability = _comparison()["stability"]
    assert all(row["roundabout_complex_stability"] == 1.0 for row in stability)


def test_reverse_overlay_excluded_and_not_missing() -> None:
    overlay = pq.read_table(ROOT / _config()["paths"]["reverse_overlay"]).to_pandas()
    assert len(overlay) == 6502
    assert not overlay["missing_identity"].any()
    assert set(overlay["av_routability_status"]) == {"AV_ROUTABILITY_VIOLATION"}
    full_edges = pq.read_table(ROOT / _config()["paths"]["edges"], columns=["stage3_edge_uid"]).to_pandas()
    assert not set(overlay["observed_valhalla_edge_id"].astype(str)) & set(full_edges["stage3_edge_uid"])


def test_topological_movement_not_legal_and_directed_path_required() -> None:
    for tolerance in (5, 10, 15, 20):
        complexes = _read(f"complexes_r{tolerance:02d}.parquet")
        movements = _read(f"movements_r{tolerance:02d}.parquet")
        assert "topological_movement_count" in complexes
        assert "legal_movement_count" not in complexes
        assert movements["topological_path_exists"].all()
        assert not movements["restriction_enforcement_certified"].any()
        assert not (movements["movement_legality_state"] == "CERTIFIED_PROHIBITED").any()


def test_missing_signal_not_unsignalized_and_positive_signal_aggregates() -> None:
    for tolerance in (5, 10, 15, 20):
        complexes = _read(f"complexes_r{tolerance:02d}.parquet")
        assert "UNSIGNALIZED" not in set(complexes["signal_state"])
        assert (complexes["signal_evidence_present"] == (complexes["signal_evidence_count"] > 0)).all()
        # Roundabout is the semantic display override, while the raw positive
        # signal flag/count remains retained for conflict audit.
        assert set(complexes.loc[complexes["signal_evidence_present"], "signal_state"]).issubset({"SIGNALIZED", "ROUNDABOUT"})


def test_poi_not_used() -> None:
    config = _config()
    assert "poi" not in config["paths"]
    controls = pq.read_table(ROOT / config["paths"]["controls"]).to_pandas()
    assert not controls["poi_used"].any()


def test_turn_angle_deterministic_and_thresholds_frozen() -> None:
    assert signed_turn(0, 90) == -90
    assert turn_type(30) == "STRAIGHT" and turn_type(-30) == "STRAIGHT"
    assert turn_type(31) == "LEFT" and turn_type(-31) == "RIGHT"
    assert turn_type(150) == "UTURN" and turn_type(-150) == "UTURN"
    assert _config()["turn_type_thresholds_deg"] == {"straight_max_abs": 30, "uturn_min_abs": 150}


def test_speed_domain_not_retrained_or_reselected() -> None:
    config = _config()
    assert config["frozen_speed"] == {"quantile": .85, "method": "MAP_SPEED_AND_ROAD_CLASS", "caps_kmh": [60, 80, 120]}
    for key in ("speed_model_retrained", "speed_quantile_reselected", "speed_method_reselected", "speed_domain_reinferred"):
        assert config["scope_guards"][key] is False


def test_tolerance_selection_excludes_test31_av_and_stage4() -> None:
    comparison = _comparison()
    assert comparison["selection_uses_test31"] is False
    assert comparison["selection_uses_av_feasibility"] is False
    assert comparison["selection_uses_stage4"] is False
    assert comparison["recommendation"]["forbidden_inputs_used"] == []


def test_s2b2_not_authorized_and_tolerance_not_frozen() -> None:
    config, comparison = _config(), _comparison()
    assert config["authorizations"]["s2b2"] is False
    assert config["next_phase_authorized"] is False
    assert comparison["tolerance_frozen"] is False
    assert comparison["recommendation"]["recommendation_only"] is True


def test_qa_pack_complete() -> None:
    qa = _read("qa_sample.parquet")
    visual = _read("qa_visualizations/qa_visual_index.parquet")
    assert len(qa) == len(visual) == 200
    assert not qa["sampling_uses_test31"].any()
    assert not qa["sampling_uses_av_feasibility"].any()
    assert all(Path(path).is_file() for path in visual["png_path"])


def test_evidence_chain_passes() -> None:
    result = verify_evidence(DOCS / "stage3_s2b1_evidence_bundle.json", ROOT)
    assert result == {"status": "PASS", "failures": [], "phase_status": PHASE_STATUS}
