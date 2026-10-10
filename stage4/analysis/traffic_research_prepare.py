"""Prepare prediction-only research policy rows and offline reconciliation."""
import json
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from stage3.scripts.traffic_state_batch1 import sha, write_json, write_parquet, EDGE
from stage3.scripts.traffic_state_batch2 import METRICS, dynamic_ratios
from stage3.scripts.traffic_state_support import decompose
from stage4.dispatch.traffic_research_policy import outside_share

OUT = Path("stage4/output/traffic_research")
DOC = Path("stage4/docs/traffic_research")
VAR = [m for m in METRICS if m.startswith(("speed_cv_", "acceleration_rms_"))]


def aggregate(p):
    w = p.travel_time_p50_s.astype(float)
    assert np.isfinite(w).all() and w.gt(0).all()
    total = w.groupby(p.order_id).sum()
    result = total.to_frame("predicted_time_s")
    masks = {"congested": p.research_state.eq("CONGESTED_PROXY"),
             "severe": p.research_state.eq("SEVERE_CONGESTION_PROXY"),
             "free": p.research_state.eq("NON_CONGESTED_PROXY"),
             "relative_mixed": p.mixed_subtype.eq("RELATIVE_SLOWDOWN_WITHOUT_HIGH_LOW_SPEED_SHARE"),
             "background_mixed": p.mixed_subtype.eq("HIGH_LOW_SPEED_SHARE_WITHOUT_RELATIVE_SLOWDOWN"),
             "unknown": p.research_state.eq("UNKNOWN"),
             "low_support": p.reference_support.eq("LOW_REFERENCE_SUPPORT")}
    for name, mask in masks.items():
        result[name+"_share"] = w.where(mask, 0).groupby(p.order_id).sum()/total
    assert np.allclose(result[[c+"_share" for c in list(masks)[:6]]].sum(axis=1), 1, atol=1e-6)
    return result.reset_index()


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    profiles_path = Path("stage3/config/stage3_av_capability_profiles.json")
    profiles = json.loads(profiles_path.read_text())
    cfgpath = Path("stage3/config/traffic_state_batch1.json")
    cfg = json.loads(cfgpath.read_text())
    reference = Path("stage3/output/traffic_state_batch1/train_reference.parquet")
    protected = {str(path): sha(path) for path in (profiles_path, cfgpath, reference)}
    sources, stats = [], []
    for date in ("20161025", "20161026", "20161027", "20161031"):
        if date != "20161031":
            path = Path(f"stage3/output/traffic_state_support/tokens_date={date}.parquet")
            p = pd.read_parquet(path, columns=["order_id", "travel_time_p50_s", "research_state", "mixed_subtype", "reference_support"])
            dp = Path(f"stage3/output/traffic_state_batch2/route_overlap_date={date}.parquet")
            d = pd.read_parquet(dp, columns=["order_id", *METRICS])
            paths = [path, dp]
        else:
            path = Path("stage3/output/odd_tod/s4/test31_m3_predictions.parquet")
            meta = json.loads(path.with_suffix(".json").read_text())
            assert meta["decision_time_only"] and meta["predicted_progression_only"]
            assert sha(path) == meta["prediction_sha256"]
            rp = Path("stage2/output_v4/route_conditioned_dataset/revealed_route_proxy/day=20161031.parquet")
            assert sha(rp) == meta["route_sha256"]
            dp = Path("stage3/output/odd_tod/s4/test31_original_route_descriptors.parquet")
            d = pd.read_parquet(dp, columns=["order_id", "dynamic_complete", *METRICS])
            d = d.loc[d.dynamic_complete].drop(columns="dynamic_complete")
            p = pd.read_parquet(path, columns=["order_id", "traversal_id", "pred_pace_p50", "pred_crawl", "pred_stop", "travel_time_p50_s"])
            p = p.loc[p.order_id.isin(d.order_id)]
            route = pd.read_parquet(rp, columns=["order_id", "traversal_id", EDGE])
            p = p.merge(route, on=["order_id", "traversal_id"], validate="one_to_one").merge(pd.read_parquet(reference), on=EDGE, how="left", validate="many_to_one")
            p = decompose(p, cfg)
            paths = [path, rp, dp]
            del route
        assert np.isfinite(d[METRICS]).all().all()
        a = aggregate(p).merge(d, on="order_id", validate="one_to_one")
        assert len(a) == len(d)
        ratios = dynamic_ratios(a, profiles)
        tables = []
        outside = []
        for profile in profiles["profiles"]:
            name = profile["profile_id"]
            r = a[[c for c in a if c not in METRICS]].copy()
            r["date"], r["profile_id"] = date, name
            r["order_id"] = r.order_id.astype(str)
            r["selected_route_reference"] = "ORIGINAL:"+r.order_id
            r["outside_share"] = outside_share(name, r.congested_share, r.severe_share, r.relative_mixed_share)
            r["rho_variability"] = pd.DataFrame({m: a[m]/profile["dynamic_caps"][m.rsplit("_",1)[0]][m.rsplit("_",1)[1]] for m in VAR}).max(axis=1)
            r["rho_dynamic_frozen"] = ratios[f"rho_dynamic_{name}"]
            assert (r.rho_variability <= r.rho_dynamic_frozen+1e-10).all()
            prev = None
            for budget in (0., .05, .1):
                allowed = r.outside_share.le(budget+1e-6)
                if prev is not None:
                    assert not (prev & ~allowed).any()
                prev = allowed
                stats.append({"date": date, "profile": name, "budget": budget, "routes": len(r),
                    "traffic_budget_allowed": int(allowed.sum()), "traffic_budget_exceeded": int((~allowed).sum()),
                    "allowed_with_unknown": int((allowed & r.unknown_share.gt(0)).sum()),
                    "mean_unknown_share": float(r.unknown_share.mean()),
                    "mean_low_support_share": float(r.low_support_share.mean())})
            outside.append((name, r.outside_share.to_numpy()))
            tables.append(r)
        u = dict(outside)
        assert (u["C"]+1e-8 >= u["M"]).all() and (u["M"]+1e-8 >= u["A"]).all()
        result = pd.concat(tables, ignore_index=True)
        dest = OUT/f"policy_date={date}.parquet"
        write_parquet(dest, result)
        sources.extend({"path": str(x), "sha256": sha(x)} for x in paths)
        print(f"{date}: {len(a)} routes; replacement/no-stack/nesting/0-5-10 monotonicity PASS", flush=True)
        del p, a, d, tables, result, ratios; gc.collect()
    assert protected == {p: sha(Path(p)) for p in protected}
    write_json(DOC/"offline_summary.json", {"status": "PASS", "rows": stats, "sources": sources,
        "protected_sha256": protected, "profile_modified": False, "new_inference": False,
        "interpretation": "Traffic-budget counts only; Gamma and other eligibility conditions not evaluated"})


if __name__ == "__main__":
    run()
