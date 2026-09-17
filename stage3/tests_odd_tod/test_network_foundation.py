from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from stage3.odd_tod.network_foundation import (
    AUTO_ACCESS_BIT,
    PHASE_STATUS,
    grouped_folds,
    load_config,
    payload_hash,
    stage3_edge_uid,
    verify_evidence,
)


REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "stage3/config/stage3_s2a_network_foundation.json"
DOCS = REPO / "stage3/docs/odd_tod/s2a"
OUTPUT = REPO / "stage3/output/odd_tod/s2a"


def _config() -> dict:
    return load_config(CONFIG, REPO)


def _read(name: str) -> pd.DataFrame:
    return pq.read_table(OUTPUT / name).to_pandas()


def _validation() -> dict:
    return json.loads((DOCS / "stage3_s2a_speed_validation.json").read_text(encoding="utf-8"))


def test_s2a_binds_frozen_sources() -> None:
    config = _config()
    assert config["execution_base_commit"] == "573413e87d4fd6698bc13100425bb5aed6b6a621"
    assert config["upstream_frozen_commit"] == "e489343abb36a878cffe48e3a5eae34e6c4670c5"
    assert config["bindings"]["pbf_sha256"] == "4d918b7ed2201a8f7a75fa7fb2974679343c89a10bf83638e7bcf21ac07c1526"


def test_full_network_not_observed_subnetwork_only() -> None:
    evidence = json.loads((DOCS / "stage3_s2a_evidence_bundle.json").read_text(encoding="utf-8"))
    assert evidence["products"]["full_network_edges"]["row_count"] > 25_382
    assert evidence["products"]["full_network_nodes"]["row_count"] > 0


def test_stage3_edge_uid_is_deterministic() -> None:
    edges = _read("stage3_full_network_edges.parquet")
    sample = edges.iloc[:: max(1, len(edges) // 1000)]
    assert sample["stage3_edge_uid"].is_unique
    assert all(stage3_edge_uid(edge_id) == uid for edge_id, uid in zip(sample.valhalla_directed_edge_id, sample.stage3_edge_uid, strict=True))


def test_motor_vehicle_routability_uses_frozen_network_semantics() -> None:
    edges = _read("stage3_full_network_edges.parquet")
    assert edges["auto_routable"].all()
    assert ((edges["valhalla_access_mask"].astype(int) & AUTO_ACCESS_BIT) != 0).all()
    assert set(edges["motor_vehicle_routability_source"]) == {"frozen_valhalla_forwardaccess_auto_bit"}


def test_full_network_nodes_are_incident_to_auto_edges() -> None:
    nodes = _read("stage3_full_network_nodes.parquet")
    assert (nodes["incident_auto_edge_count"] > 0).all()


def test_observed_canonical_mapping_direction_preserved() -> None:
    mapping = _read("stage3_observed_full_network_mapping.parquet")
    assert set(mapping["canonical_traversal_direction"]) == {"F", "R"}
    assert mapping[["canonical_edge_uid", "canonical_traversal_direction"]].drop_duplicates().shape[0] == 25_382


def test_ambiguous_mapping_not_silently_resolved() -> None:
    mapping = _read("stage3_observed_full_network_mapping.parquet")
    ambiguous = mapping["candidate_count"] > 1
    assert (mapping.loc[ambiguous, "mapping_status"] == "AMBIGUOUS").all()
    assert not mapping["mapping_method"].str.contains("nearest", case=False).any()


def test_bridge_osm_enrichment_preserves_raw_stage0() -> None:
    edges = _read("stage3_full_network_edges.parquet")
    required = {"bridge_stage0", "bridge_osm", "bridge_effective", "bridge_conflict"}
    assert required.issubset(edges.columns)
    assert (edges["bridge_effective"] >= edges["bridge_osm"]).all()


def test_roundabout_provenance_is_explicit() -> None:
    edges = _read("stage3_full_network_edges.parquet")
    assert {"junction_roundabout_way", "mini_roundabout_node_exposure", "roundabout_evidence_source"}.issubset(edges.columns)
    evidenced = edges["junction_roundabout_way"] | edges["mini_roundabout_node_exposure"]
    assert edges.loc[evidenced, "roundabout_evidence_source"].notna().all()


def test_missing_signal_not_equal_unsignalized_and_poi_not_used() -> None:
    controls = _read("stage3_control_evidence.parquet")
    assert controls["positive_evidence_only"].all()
    assert not controls["missing_tag_means_negative_control"].any()
    assert not controls["poi_used"].any()


def test_restriction_reader_preserves_roles_and_forbids_geometry_guess() -> None:
    restrictions = _read("stage3_turn_restrictions.parquet")
    assert len(restrictions) == 20
    assert set(restrictions["reader"]) == {"pyosmium_role_ref_preserving"}
    for row in restrictions.itertuples(index=False):
        members = json.loads(row.members)
        assert json.loads(row.from_member) == [item for item in members if item["role"] == "from"]
        assert json.loads(row.via_members) == [item for item in members if item["role"] == "via"]
        assert json.loads(row.to_member) == [item for item in members if item["role"] == "to"]
    assert not restrictions["geometry_guessing_used"].any()


def test_speed_anchor_set_reproducible() -> None:
    validation = _validation()
    assert validation["anchor_count"] == 502
    assert validation["anchor_set_sha256"]


def test_speed_history_train_only_and_test31_never_used() -> None:
    validation = _validation()
    assert validation["train_history_dates"] == [f"201610{x:02d}" for x in range(9, 25)]
    assert "20161031" not in validation["train_history_dates"]
    assert validation["test31_used"] is False


def test_speed_quantiles_grid_and_label_semantics() -> None:
    config = _config()
    validation = _validation()
    speed = _read("stage3_speed_domain.parquet")
    assert config["speed_inference"]["candidate_quantiles"] == [0.85, 0.9, 0.95]
    assert all(float(value).is_integer() for value in validation["speed_grid_kmh"])
    assert not speed["is_verified_posted_speed_limit"].any()


def test_speed_cv_grouping_prevents_duplicate_anchor_leakage() -> None:
    groups = ["way1", "way1", "way2", "way3", "way3"]
    folds = grouped_folds(groups, 5, 20261009)
    assert folds[0] == folds[1]
    assert folds[3] == folds[4]
    assert _validation()["duplicate_group_leakage_count"] == 0


def test_speed_model_selection_uses_scenario_compatibility_metric() -> None:
    validation = _validation()
    selected = validation["selected"]
    ranked = sorted(
        validation["candidate_results"],
        key=lambda row: (
            -row["macro_scenario_compatibility_accuracy"],
            -row["within_10_kmh_accuracy"],
            -row["exact_class_accuracy"],
            row["model_complexity"],
            row["quantile"],
        ),
    )[0]
    assert selected["method"] == ranked["method"]
    assert selected["quantile"] == ranked["quantile"]


def test_speed_domain_caps_remain_60_80_120() -> None:
    assert _config()["speed_inference"]["av_caps_kmh"] == [60, 80, 120]


def test_s2b_and_later_work_not_authorized() -> None:
    config = _config()
    assert config["authorizations"]["s2b"] is False
    assert config["next_phase_authorized"] is False
    assert not any(config["scope_guards"].values())


def test_evidence_chain_passes() -> None:
    result = verify_evidence(DOCS / "stage3_s2a_evidence_bundle.json", REPO)
    assert result == {"status": "PASS", "failures": [], "phase_status": PHASE_STATUS}
