import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from stage3.odd_tod.intersection_complex import recommend
from stage3.odd_tod.network_foundation import payload_hash


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "stage3/docs/odd_tod/s2b"
OUT = ROOT / "stage3/output/odd_tod/s2b/closure_5v10"


def _json(name):
    return json.loads((DOCS / name).read_text(encoding="utf-8"))


def test_phase_scope_and_prior_recommendation_are_closed_correctly():
    closure = _json("stage3_s2b11_5v10_closure.json")
    assert closure["s2b1_engineering"] == "PASS"
    assert closure["s2b1_tolerance_selection"] == "NOT_YET_CLOSED"
    assert closure["recommendation_status"] == "5m_RECOMMENDATION_NOT_ACCEPTED"
    assert closure["final_review_pair_m"] == [5, 10]
    assert closure["rejected_baselines_m"] == [15, 20]
    assert closure["four_tolerance_products_recomputed"] is False
    assert closure["s2b2_authorized"] is False
    assert closure["next_phase_authorized"] is False
    assert closure["artifact_sha256"] == payload_hash(closure)


def test_recommendation_function_has_no_small_radius_tiebreak():
    result = recommend([{"tolerance_m": value} for value in (5, 10, 15, 20)], [])
    assert result["recommended_tolerance_m"] is None
    assert result["recommendation_status"] == "NOT_YET_CLOSED"
    assert result["final_review_pair_m"] == [5, 10]
    assert "no smaller-radius tie-break" in result["basis"]


def test_endpoint_incompleteness_impact_is_fully_accounted():
    endpoint = _json("stage3_s2b11_5v10_closure.json")["endpoint_incompleteness"]
    assert endpoint["endpoint_incomplete_edge_count"] == 13663
    assert endpoint["missing_from_count"] == 6995
    assert endpoint["missing_to_count"] == 7029
    assert endpoint["missing_both_count"] == 361
    assert sum(row["endpoint_incomplete_count"] for row in endpoint["by_road_class"]) == 13663
    assert {row["evidence"] for row in endpoint["complete_edge_comparisons"]} == {
        "junction_candidate", "signal", "changed_05_10"
    }
    # No evidence source is enriched among incomplete edges relative to the
    # complete-edge reference population at the frozen descriptive radius.
    assert all(row["incomplete_to_complete_share_ratio"] < 1 for row in endpoint["complete_edge_comparisons"])


def test_signal_fragmentation_preserves_assignments_and_reports_buckets():
    signal = _json("stage3_s2b11_5v10_closure.json")["signal_fragmentation"]
    assert signal["r05"]["total_signal_node_assignments"] == 3792
    assert signal["r10"]["total_signal_node_assignments"] == 3792
    assert signal["r05"]["signal_nodes_per_complex_distribution"] == {"1": 1943, "2": 602, "3": 37, "4+": 133}
    assert signal["r10"]["signal_nodes_per_complex_distribution"] == {"1": 438, "2": 670, "3": 80, "4+": 441}
    assert signal["r10"]["singleton_signal_node_share"] < signal["r05"]["singleton_signal_node_share"]


def test_full_network_degree_distribution_is_conservative_and_complete():
    degree = _json("stage3_s2b11_5v10_closure.json")["degree_sanity"]
    assert degree["node_count"] == 89607
    assert sum(degree["degree_distribution"].values()) == degree["node_count"]
    assert degree["degree_max"] == 6
    assert degree["degree_ge3_count"] == 58111
    assert "endpoint-complete" in degree["definition"]


def test_adjudication_pack_has_exact_nonoverlapping_quotas_and_blank_labels():
    cases = pq.read_table(OUT / "s2b11_adjudication_cases.parquet").to_pandas()
    assert len(cases) == 70
    assert cases["complex_r10"].nunique() == 70
    assert cases["selection_stratum"].value_counts().to_dict() == {
        "signalized": 20,
        "multi_node_divided_road": 20,
        "high_degree": 10,
        "grade_separated": 10,
        "random_changed": 10,
    }
    assert (cases["adjudication_label"] == "").all()
    assert (cases["reviewer_note"] == "").all()
    assert int(cases["reused_existing_qa"].sum()) == 23
    assert (OUT / "s2b11_adjudication_sheet.csv").is_file()


def test_visual_pack_is_bound_and_complete():
    visual = pq.read_table(OUT / "s2b11_visual_index.parquet").to_pandas()
    assert len(visual) == 70
    assert set(visual["left_panel"]) == {"5m"}
    assert set(visual["right_panel"]) == {"10m"}
    assert all((ROOT / path).is_file() for path in visual["png_path"])
    assert all(size > 0 for size in visual["png_size_bytes"])
    manifest = _json("stage3_s2b11_qa_manifest.json")
    assert manifest["case_count"] == 70
    assert manifest["labels_filled"] == 0
    assert manifest["artifact_sha256"] == payload_hash(manifest)


def test_evidence_is_hash_bound_and_scope_guards_are_closed():
    evidence = _json("stage3_s2b11_evidence_bundle.json")
    assert evidence["artifact_sha256"] == payload_hash(evidence)
    assert not any(evidence["guards"].values())
    for section in ("inputs", "outputs"):
        for descriptor in evidence[section].values():
            path = Path(descriptor["path"])
            path = path if path.is_absolute() else ROOT / path
            assert path.is_file()
