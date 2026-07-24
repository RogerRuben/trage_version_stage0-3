import json

import numpy as np
import pandas as pd

from stage1.canonical.labels import aggregate_order_labels_v2
from stage1.canonical.quantiles import MergeableHistogram, empirical_cdf_interpolated


def test_mergeable_quantile_is_partition_invariant():
    values = np.linspace(0.01, 9.99, 10007)
    edges = np.linspace(0.0, 10.0, 4097)
    results = []
    for partitions in (1, 5, 20):
        merged = MergeableHistogram.empty(edges)
        for part in np.array_split(values, partitions):
            local = MergeableHistogram.empty(edges)
            local.update(part)
            merged.merge(local)
        results.append(merged.quantile(0.5))
    assert results[0] == results[1] == results[2]


def test_cdf_interpolates_empty_support_and_preserves_tails():
    query = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, np.nan])
    result = empirical_cdf_interpolated(query, np.array([0.0, 2.0]), np.array([10, 10]))
    assert result[0] == 0.0
    assert 0.0 < result[1] < result[2] < result[3] < result[4]
    assert result[4] == 1.0
    assert np.isnan(result[-1])


def test_pmis_is_reported_but_not_double_counted_in_core_composite():
    frame = pd.DataFrame({
        "order_id": ["o1"],
        "travel_time_sec": [10.0],
        "observed_distance_m": [100.0],
        "lcs_pct_link": [0.2],
        "iis_pct_link": [0.4],
        "gns_pct_link": [0.6],
        "rts_pct_link": [0.8],
        "pmis_pct_link": [1.0],
    })
    row = aggregate_order_labels_v2(frame).iloc[0]
    assert np.isclose(row.core_composite_mean, (0.2 + 0.6 + 0.8) / 3)
    assert row.pmis_mean == 1.0
    assert row.pmis_role == "interaction_output_excluded_from_core_composite"


def test_missing_dimension_is_masked_not_zero_imputed():
    frame = pd.DataFrame({
        "order_id": ["o1"],
        "travel_time_sec": [10.0],
        "observed_distance_m": [100.0],
        "lcs_pct_link": [0.2],
        "iis_pct_link": [np.nan],
        "gns_pct_link": [0.6],
        "rts_pct_link": [0.8],
        "pmis_pct_link": [np.nan],
    })
    row = aggregate_order_labels_v2(frame).iloc[0]
    mask = json.loads(row.dimension_mask)
    assert mask["iis"] is False
    assert np.isnan(row.iis_mean)
    assert row.valid_dimension_count == 3
    assert row.composition_signature == "lcs+gns+rts"
