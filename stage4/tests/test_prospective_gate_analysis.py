from pathlib import Path

import pytest

from stage4.analysis.prospective_gate_analysis import (
    load_and_validate,
    retention_table,
)


ROOT = Path(__file__).resolve().parents[2]


def test_four_anchor_outputs_reproduce_and_reconcile() -> None:
    totals, bins, reproduction = load_and_validate(ROOT)

    assert len(totals) == 4
    assert not bins.empty
    assert reproduction["canonical_reproduction_pass"].all()
    assert (reproduction["summary_difference_count"] == 0).all()
    assert reproduction["request_outcomes_exact"].all()
    assert reproduction["assignments_exact"].all()


def test_central_eligibility_conversion_declines_and_routing_is_exact() -> None:
    totals, _, _ = load_and_validate(ROOT)
    central = totals.set_index("scenario_id").loc[
        ["MAIN_Q25_M_P70", "MAIN_Q50_M_P70", "MAIN_Q75_M_P70"]
    ]
    conversion = central["eligibility_conversion_n5_over_n0"].tolist()

    assert conversion[0] > conversion[1] > conversion[2]

    retention = retention_table(totals)
    routing = retention[retention["transition"].eq("routing")]
    assert routing["retention_share"].tolist() == pytest.approx([1.0] * 4)
