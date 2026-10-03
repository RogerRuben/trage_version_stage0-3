import numpy as np
import pandas as pd
from stage3.scripts.traffic_state_support import decompose, split_bit

CFG = {"reference_min_order_days":100,"reference_min_days":5,
       "pace_ratio_congested":1.5,"pace_ratio_severe":2.,
       "low_speed_share_congested":.5,"low_speed_share_severe":.8}


def test_estimability_is_separate_from_support():
    x = pd.DataFrame({"pred_pace_p50":[1.,1.,np.nan,1.],"pred_crawl":[.1]*4,"pred_stop":[.1]*4,
        "reference_pace_q20":[1.,np.nan,1.,1.],"reference_order_days":[1,np.nan,100,100],"reference_days":[1,np.nan,5,5]})
    r = decompose(x, CFG)
    assert r.unknown_reason.tolist() == ["LOW_COUNT_AND_DAYS","NO_REFERENCE","INVALID_PREDICTION","NOT_UNKNOWN"]
    assert r.research_state.tolist() == ["NON_CONGESTED_PROXY","UNKNOWN","UNKNOWN","NON_CONGESTED_PROXY"]
    assert r.reference_support.iloc[0] == "LOW_REFERENCE_SUPPORT"


def test_mixed_is_not_missing_or_forced_congestion():
    x = pd.DataFrame({"pred_pace_p50":[2.,1.],"pred_crawl":[.1,.8],"pred_stop":[0.,0.],
        "reference_pace_q20":[1.,1.],"reference_order_days":[100,100],"reference_days":[5,5]})
    r = decompose(x, CFG)
    assert r.research_state.eq("MIXED_EVIDENCE").all()
    assert r.mixed_subtype.tolist() == ["RELATIVE_SLOWDOWN_WITHOUT_HIGH_LOW_SPEED_SHARE","HIGH_LOW_SPEED_SHARE_WITHOUT_RELATIVE_SLOWDOWN"]
    assert r.unknown_reason.eq("NOT_UNKNOWN").all()


def test_order_day_split_is_repeatable():
    assert split_bit("20161009", "a") == split_bit("20161009", "a")
    assert {split_bit("20161009", str(i)) for i in range(100)} == {0,1}
