from pathlib import Path

import pandas as pd

from stage4.analysis.paper_enhancement_gate_decomposition import (
    ANCHORS,
    build_observability_table,
    extract_observed_counts,
)


ROOT = Path(__file__).resolve().parents[2]


def test_gate_audit_uses_only_frozen_logged_counts() -> None:
    observed = extract_observed_counts(ROOT)
    assert tuple(observed["scenario_id"]) == ANCHORS
    assert (observed["request_count"] == 30_000).all()
    assert (observed["selected_total_assignments"] > 0).all()
    assert (observed["enabled_gamma_constraint_count_max"] == 0).all()


def test_unobserved_funnel_stages_are_not_filled() -> None:
    table = build_observability_table(extract_observed_counts(ROOT))
    unavailable = table[table["observability"].str.startswith("NOT_")]
    assert unavailable["retained_share"].isna().all()
    assert unavailable["lost_share"].isna().all()
    nominal = table[table["gate"] == "nominal_nearby_av_opportunity"]
    assert nominal["logged_value"].isna().all()
    assert set(table["scenario_id"]) == set(ANCHORS)
    assert not table["observability"].eq("INFERRED").any()
