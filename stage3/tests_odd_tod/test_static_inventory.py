from __future__ import annotations

import json
from pathlib import Path

import pytest

from stage3.odd_tod.static_inventory import (
    PHASE_STATUS,
    Stage3S1InventoryError,
    _load_config,
    parse_maxspeed,
    parse_other_tags,
    verify_inventory,
)


REPO = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO / "stage3/config/stage3_s1_static_inventory.json"
INVENTORY_PATH = REPO / "stage3/docs/odd_tod/stage3_static_data_inventory.json"


def _inventory() -> dict:
    return json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))


def test_maxspeed_parser_preserves_provenance() -> None:
    assert parse_maxspeed("60") == {
        "raw": "60", "format": "numeric", "parseable": True, "values_kmh": [60.0]
    }
    assert parse_maxspeed("50 mph")["values_kmh"] == pytest.approx([80.4672])
    assert parse_maxspeed("CN:urban") == {
        "raw": "CN:urban",
        "format": "named_or_non_numeric",
        "parseable": False,
        "values_kmh": [],
    }
    assert parse_maxspeed("signals")["parseable"] is False


def test_osm_other_tags_parser() -> None:
    assert parse_other_tags('"maxspeed"=>"60","junction"=>"roundabout"') == {
        "maxspeed": "60",
        "junction": "roundabout",
    }


def test_s1_config_forbids_later_phase_work() -> None:
    config = _load_config(CONFIG_PATH, REPO)
    assert config["authorizations"]["s1"] is True
    assert all(config["authorizations"][f"s{phase}"] is False for phase in range(2, 9))
    assert config["authorizations"]["stage4"] is False
    assert not any(config["analysis_boundaries"].values())


def test_inventory_identity_and_scope_pass() -> None:
    assert verify_inventory(_inventory()) == {
        "schema_version": "stage3_s1_static_inventory_verification.1",
        "status": "PASS",
        "phase_status": PHASE_STATUS,
        "s2_authorized": False,
        "next_phase_authorized": False,
    }


def test_inventory_has_all_priority_facts() -> None:
    inventory = _inventory()
    stage0_speed = inventory["stage0_route_parts"]["speed_limit"]
    osm_speed = inventory["frozen_osm_pbf"]["maxspeed"]
    signals = inventory["frozen_osm_pbf"]["traffic_signals"]
    restrictions = inventory["frozen_osm_pbf"]["turn_restrictions"]
    assert stage0_speed["field_present"] is True
    assert stage0_speed["provenance_field_present"] is False
    assert stage0_speed["unique_edge_weighted"]["coverage_positive"]["denominator"] > 0
    assert osm_speed["base_tag_coverage"]["denominator"] > 0
    assert signals["identity_method"] == "exact OSM node ID intersection"
    assert restrictions["restriction_relation_count"] > 0


def test_speed_counts_reconcile_and_abnormal_values_are_descriptive() -> None:
    speed = _inventory()["stage0_route_parts"]["speed_limit"]
    row = speed["row_weighted"]
    edge = speed["unique_edge_weighted"]
    assert row["coverage_positive"]["count"] + row["null"]["count"] == row["coverage_positive"]["denominator"]
    assert edge["coverage_positive"]["count"] + edge["null"]["count"] == edge["coverage_positive"]["denominator"]
    assert speed["diagnostic_abnormal_definition"] == {
        "low_positive_below_kmh": 5.0,
        "high_above_kmh": 160.0,
    }


def test_turn_restriction_mapping_is_not_overclaimed() -> None:
    restrictions = _inventory()["frozen_osm_pbf"]["turn_restrictions"]
    assert restrictions["exact_current_directed_network_mapping_certified"] is False
    assert "roles/member refs" in restrictions["exact_mapping_blocker"]


def test_poi_is_corroboration_only() -> None:
    poi = _inventory()["poi"]
    assert poi["osm_identity_fields_present"] is False
    assert poi["signal_or_junction_role"] == "corroboration_only"
    assert "no direct OSM node" in poi["interpretation"]


def test_graph_scope_is_observed_subnetwork_not_complete_network() -> None:
    inventory = _inventory()
    assert "observed subnetwork" in inventory["stage0_route_parts"]["population"]["scope_note"]
    assert inventory["graph_identity"]["complete_frozen_network_edge_table_available"] is False
    assert inventory["graph_identity"]["observed_graph_ways_found_as_raw_highway_ways"]["share"] == 1.0


def test_inventory_tamper_fails_closed() -> None:
    inventory = _inventory()
    inventory["s2_authorized"] = True
    with pytest.raises(Stage3S1InventoryError, match="invalid S1 inventory"):
        verify_inventory(inventory)
