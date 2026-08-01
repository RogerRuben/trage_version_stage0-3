from __future__ import annotations

import numpy as np

from stage2.v5.cdf import EmpiricalCDFIndex, map_empirical_cdf, map_empirical_cdf_reference


def test_indexed_cdf_matches_reference_with_fallback() -> None:
    values = np.array([1.5, 2.5, 4.0, np.nan])
    keys = {"edge": np.array(["a", "b", "missing", "a"]), "global": np.array(["all"] * 4)}
    index = EmpiricalCDFIndex(
        levels=("edge", "global"),
        samples={"edge": {"a": np.array([1.0, 2.0]), "b": np.array([2.0])}, "global": {"all": np.array([1.0, 2.0, 3.0, 4.0])}},
        supports={"edge": {"a": 2, "b": 1}, "global": {"all": 4}},
    )
    actual = map_empirical_cdf(values, keys, index, minimum_support=2)
    expected = map_empirical_cdf_reference(values, keys, index, minimum_support=2)
    assert np.allclose(actual[0], expected[0], equal_nan=True, atol=1e-10)
    assert actual[1].tolist() == expected[1].tolist()
    assert actual[2].tolist() == expected[2].tolist()

