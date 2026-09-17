from __future__ import annotations

import pandas as pd

from stage2.v4.cdf_adapter import apply_frozen_stage1_cdf
from stage2.v4.config import load_config


def test_frozen_stage1_cdf_is_used_without_refit() -> None:
    catalog = pd.read_parquet(
        "stage1/models/stage1_v3_final/directed_edge_catalog.parquet"
    ).iloc[0]
    frame = pd.DataFrame(
        {
            "observed_directed_edge_uid": [
                catalog["observed_directed_edge_uid"],
                None,
            ],
            "canonical_highway": [catalog["canonical_highway"], None],
            "estimated_time_bin": [16, 16],
            "estimated_weekday_type": ["weekday", "weekday"],
            "estimated_entry_time": [1475971200.0, 1475971200.0],
            "pred_lcs_raw": [0.3, 0.3],
            "pred_rts_raw": [0.2, 0.2],
        }
    )
    config = load_config("stage2/config/stage2_v4.json")
    result = apply_frozen_stage1_cdf(frame, config)
    assert result["cdf_model_id"].iloc[0] == config.section("stage1_release")["model_id"]
    assert result["pred_lcs_pct"].between(0.0, 1.0).all()
    assert result["pred_rts_pct"].between(0.0, 1.0).all()
    assert result.loc[1, "pred_lcs_cdf_level_used"] in {"highway", "global"}
