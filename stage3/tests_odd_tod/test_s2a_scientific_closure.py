from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from stage3.odd_tod.s2a_scientific_closure import PHASE_STATUS, payload_hash, verify


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "stage3/docs/odd_tod/s2a"
OUTPUT = ROOT / "stage3/output/odd_tod/s2a"


def _report():
    return json.loads((DOCS / "stage3_s2a1_scientific_closure.json").read_text(encoding="utf-8"))


def test_anchor_unit_and_502_to_180_reconciliation() -> None:
    item = _report()["anchor_reconciliation"]
    assert item["unit"] == "canonical_segment_anchor"
    assert item["canonical_segment_anchor_count"] == 502
    assert item["mapped_canonical_segment_anchor_count"] == 500
    assert item["unmapped_canonical_segment_anchor_count"] == 2
    assert item["unique_full_network_directed_edges_after_mapping"] == 180
    assert item["known_full_network_speed_rows"] == 180
    assert item["directed_identity_claim_withdrawn"] is True


def test_80_and_120_are_not_empirically_identified() -> None:
    caps = {item["cap_kmh"]: item for item in _report()["cap_identification"]}
    assert caps[60]["empirically_identified"] is True
    for cap in (80, 120):
        assert caps[cap]["anchor_gt_cap_count"] == 0
        assert caps[cap]["empirically_identified"] is False
        assert "NOT_EMPIRICALLY_IDENTIFIED" in caps[cap]["interpretation"]
    assert _report()["macro_interpretation"] == "MECHANICAL_THREE_THRESHOLD_AVERAGE_NOT_THREE_PROFILE_EMPIRICAL_VALIDATION"


def test_b0_b2_full_network_disagreement_is_explicit() -> None:
    rows = {item["cap_kmh"]: item for item in _report()["b0_vs_b2_compatibility_disagreement"]}
    assert rows[60]["full_network_disagreement_count"] == 26
    assert rows[80]["full_network_disagreement_count"] == 3
    assert rows[120]["full_network_disagreement_count"] == 0
    assert all(item["full_network_denominator"] == 209_454 for item in rows.values())
    assert all(item["b2_applicable_denominator"] == 4_898 for item in rows.values())


def test_reverse_identities_are_overlay_violations_not_missing() -> None:
    overlay = pq.read_table(OUTPUT / "stage3_historical_direction_overlay.parquet").to_pandas()
    assert len(overlay) == 6_502
    assert set(overlay["canonical_traversal_direction"]) == {"R"}
    assert set(overlay["historical_direction_status"]) == {"HISTORICAL_DIRECTION_OVERLAY"}
    assert set(overlay["av_routability_status"]) == {"AV_ROUTABILITY_VIOLATION"}
    assert overlay["historical_observation_accepted"].all()
    assert not overlay["missing_identity"].any()


def test_no_frozen_selection_or_scope_change() -> None:
    report = _report()
    assert report["frozen_selection"] == {"quantile": 0.85, "method": "MAP_SPEED_AND_ROAD_CLASS", "caps_kmh": [60, 80, 120]}
    assert report["network_reexported"] is False
    assert report["speed_model_retrained"] is False
    assert report["speed_quantile_reselected"] is False
    assert report["speed_method_reselected"] is False
    assert report["av_caps_changed"] is False
    assert report["s2b_authorized"] is False
    assert report["next_phase_authorized"] is False


def test_closure_report_and_evidence_hashes_pass() -> None:
    report = _report()
    assert report["phase_status"] == PHASE_STATUS
    assert report["artifact_sha256"] == payload_hash(report)
    result = verify(DOCS / "stage3_s2a1_scientific_closure_evidence.json", ROOT)
    assert result == {"status": "PASS", "failures": [], "phase_status": PHASE_STATUS}
