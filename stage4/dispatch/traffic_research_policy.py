"""Opt-in traffic replacement. Frozen requests and profiles are never mutated."""
from math import isfinite
import pandas as pd
from .exposure import exposure_excess

MODE = "TRAFFIC_REPLACEMENT_V1"
VARIABILITY_MODE = "VARIABILITY_ONLY_V1"
KEYS = ["date", "order_id", "profile_id", "selected_route_reference"]


def outside_share(profile, congested, severe, relative_mixed):
    if profile not in ("C", "M", "A"):
        raise ValueError("unknown profile")
    return congested + severe + relative_mixed if profile == "C" else severe if profile == "M" else 0.


class TrafficResearchPolicy:
    def __init__(self, frame, budget=.05, *, mode=MODE):
        if mode not in (MODE, VARIABILITY_MODE):
            raise ValueError("unrecognized research mode")
        self.mode = mode
        if not isfinite(budget) or not 0 <= budget <= 1:
            raise ValueError("invalid traffic budget")
        if frame.duplicated(KEYS).any():
            raise ValueError("duplicate traffic route identity")
        cols = ["outside_share", "unknown_share", "low_support_share", "rho_variability"]
        if frame[cols].isna().any().any():
            raise ValueError("incomplete research evidence")
        for col in cols:
            if not frame[col].map(isfinite).all() or not frame[col].ge(0).all():
                raise ValueError("invalid research value")
        if not frame[cols[:3]].le(1+1e-6).all().all():
            raise ValueError("invalid share")
        self.rows = {tuple(str(row[k]) for k in KEYS): row for row in frame.to_dict("records")}
        self.budget = float(budget)

    @classmethod
    def from_config(cls, config):
        mode = config.get("traffic_research_policy", "FROZEN")
        if mode == "FROZEN":
            return None
        if mode not in (MODE, VARIABILITY_MODE):
            raise ValueError("unrecognized traffic policy")
        return cls(pd.read_parquet(config["traffic_research_table"]), float(config["traffic_research_budget"]), mode=mode)

    def evaluate(self, request):
        if not request.av_smoke_eligible:
            return None  # Baseline structural/evidence exclusions stay unchanged.
        if request.selected_route_type != "ORIGINAL":
            raise ValueError("research traffic not defined for selected fallback route")
        key = (request.request_time.strftime("%Y%m%d"), str(request.order_id), str(request.profile_id), "ORIGINAL:"+str(request.order_id))
        if key not in self.rows:
            raise ValueError(f"missing selected-route traffic evidence: {key}")
        row = self.rows[key]
        exposure = exposure_excess(request.rho_static, row["rho_variability"], request.rho_speed)
        if exposure is None:
            raise ValueError("missing retained-family evidence")
        return {"exposure": exposure, "traffic_allowed": self.mode == VARIABILITY_MODE or row["outside_share"] <= self.budget+1e-6,
                "traffic_outside_share": row["outside_share"], "traffic_unknown_share": row["unknown_share"],
                "traffic_low_support_share": row["low_support_share"], "traffic_policy": self.mode}
