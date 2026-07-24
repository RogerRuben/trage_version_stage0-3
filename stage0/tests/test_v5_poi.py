import pandas as pd

from stage0.v5.poi import deduplicate_nearest_assignments


def test_poi_equidistant_edges_do_not_duplicate_exposure():
    joined = pd.DataFrame({
        "poi_id": [1, 1, 2],
        "edge_uid": ["b", "a", "c"],
        "edge_distance_m": [5.0, 5.0, 2.0],
    })
    result = deduplicate_nearest_assignments(joined)
    assert result.poi_id.is_unique
    assert result.loc[result.poi_id.eq(1), "edge_uid"].iloc[0] == "a"
