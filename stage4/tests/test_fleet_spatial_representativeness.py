from __future__ import annotations

import pandas as pd
import pytest

from stage4.analysis.fleet_spatial_representativeness import spatial_metrics


def _shares(values: dict[tuple[int, int], float]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"grid_x": x, "grid_y": y, "share": share} for (x, y), share in values.items()]
    )


def test_spatial_metrics_are_exact_for_identical_distributions() -> None:
    frame = _shares({(0, 0): 0.6, (1, 0): 0.4})
    result = spatial_metrics(frame, frame)
    assert result["total_variation_distance"] == 0.0
    assert result["jensen_shannon_divergence"] == 0.0
    assert result["spearman_occupied_share"] == pytest.approx(1.0)
    assert result["top_10pct_hotspot_jaccard"] == 1.0


def test_spatial_metrics_use_union_cells_and_normalized_shares() -> None:
    left = _shares({(0, 0): 1.0})
    right = _shares({(1, 0): 1.0})
    result = spatial_metrics(left, right)
    assert result["union_cell_count"] == 2
    assert result["total_variation_distance"] == 1.0
    assert result["jensen_shannon_divergence"] == 1.0
    assert result["top_10pct_hotspot_jaccard"] == 0.0
