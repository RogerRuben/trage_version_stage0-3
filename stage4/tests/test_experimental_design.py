from pathlib import Path

import pytest

from stage4.dispatch.acceptance import passenger_acceptance, stable_acceptance_uniform
from stage4.dispatch.experimental_design import (
    ACCEPTANCE_LEVELS,
    ACCEPTANCE_SEED,
    FAMILIES,
    GAMMA_PRESETS,
    H_BASE_EXACT_EXPECTED,
    PATH_DIAGNOSTIC,
    build_scenario_registry,
    configuration_signature,
    validate_registry,
)

ROOT = Path(__file__).resolve().parents[2]


def _rows():
    return build_scenario_registry(ROOT)


def test_acceptance_common_random_number_sets_are_nested():
    order_ids = [f"nested-{index}" for index in range(4096)]
    draws = {
        order_id: stable_acceptance_uniform(order_id, ACCEPTANCE_SEED)
        for order_id in order_ids
    }
    accepted = {
        rate: {
            order_id
            for order_id in order_ids
            if passenger_acceptance(
                order_id, rate, ACCEPTANCE_SEED
            ).passenger_accepts_av
        }
        for rate in ACCEPTANCE_LEVELS
    }
    assert accepted[0.40] <= accepted[0.70] <= accepted[1.00]
    assert all(0.0 <= draw < 1.0 for draw in draws.values())


def test_main_structural_matrix_has_exactly_27_scenarios():
    main = [row for row in _rows() if row["experiment_block"] == "MAIN_STRUCTURAL"]
    assert len(main) == 27
    assert {row["requested_q_A"] for row in main} == {0.25, 0.50, 0.75}
    assert {row["profile_id"] for row in main} == {"C", "M", "A"}
    assert {row["acceptance_probability"] for row in main} == {0.40, 0.70, 1.00}


def test_registry_duplicates_require_exact_reuse_linkage():
    rows = _rows()
    counts = validate_registry(rows)
    source_signatures = {}
    for row in rows:
        signature = configuration_signature(row)
        reuse = row["reuse_source_scenario_id"]
        if reuse:
            source = next(item for item in rows if item["scenario_id"] == reuse)
            assert configuration_signature(source) == signature
        else:
            assert signature not in source_signatures
            source_signatures[signature] = row["scenario_id"]
    assert counts["reuse_rows"] == 1


def test_registry_q_a_uses_exact_vehicle_hour_denominator():
    for row in _rows():
        assert row["H_base_exact"] == pytest.approx(H_BASE_EXACT_EXPECTED, abs=1e-6)
        expected_av = round(row["requested_q_A"] * row["H_base_exact"] / 24.0)
        assert row["AV_vehicle_count"] == expected_av
        assert row["achieved_q_A"] == pytest.approx(
            24.0 * expected_av / row["H_base_exact"]
        )
        assert row["H_base_exact"] != pytest.approx(14369.5)


def test_gamma_presets_are_frozen_exactly():
    assert GAMMA_PRESETS["STRICT"] == {
        "static": 0.0,
        "dynamic": 0.0,
        "speed": 0.0,
    }
    assert GAMMA_PRESETS["REFERENCE"] == {
        "static": 2.145067625590382,
        "dynamic": 0.14934256810554178,
        "speed": 0.0,
    }
    assert GAMMA_PRESETS["UNCONSTRAINED"] == {
        "static": None,
        "dynamic": None,
        "speed": None,
    }


def test_speed_family_is_retained_in_every_gamma_vector():
    assert all(set(vector) == set(FAMILIES) for vector in GAMMA_PRESETS.values())
    assert "speed" in PATH_DIAGNOSTIC
    assert PATH_DIAGNOSTIC["speed"] == 0.0


def test_unique_scenario_count_stays_within_cap():
    rows = _rows()
    counts = validate_registry(rows)
    assert len(rows) == 42
    assert counts == {
        "registry_rows": 42,
        "unique_dispatch_scenarios": 41,
        "reuse_rows": 1,
    }
    assert counts["unique_dispatch_scenarios"] <= 45
    assert sum(row["experiment_block"] == "BENCHMARK" for row in rows) == 4
    assert sum(row["experiment_block"] == "COST_ROBUSTNESS" for row in rows) == 8
