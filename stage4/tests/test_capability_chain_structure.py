import pytest

from stage4.analysis.capability_chain_structure import exact_served, run_checks, subchains


def test_structural_counterexamples_and_complete_dual_bounds():
    result = run_checks()
    checks = result["checks"]
    assert checks["fractional_relaxation"]["integer_gap_jobs"] == 1
    assert checks["deployment_complementarity"]["marginal_alone"] == 0
    assert checks["deployment_complementarity"]["marginal_after_other"] == 1
    assert checks["dual_increment"]["beta"] == 0
    assert checks["dual_increment"]["ip_increment"] == 1
    assert result["maximum_columns"] <= 20 and result["maximum_nonzeros"] <= 64


def test_toy_limits_and_no_duplicate_service():
    with pytest.raises(ValueError, match="repeats"):
        subchains(("same", "same"))
    with pytest.raises(ValueError, match="integer"):
        exact_served({"C": (("a",),)}, {"C": 0.5})
    with pytest.raises(ValueError, match="1..6"):
        exact_served({str(v): (("a",),) for v in range(7)}, {str(v): 1 for v in range(7)})


def test_realized_task_overlap_is_not_hidden_by_path_rewards():
    result = exact_served({"C": subchains(("a", "b")), "H": subchains(("b", "c"))}, {"C": 1, "H": 1})
    visited = [job for row in result["selected"] for job in row["chain"]]
    assert result["served"] == 3 and len(visited) == len(set(visited))
