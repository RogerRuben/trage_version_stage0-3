"""Separate estimability from reference support; no new operational gate."""
from __future__ import annotations
import gc
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd
import psutil

from stage3.scripts.traffic_state_batch1 import EDGE, ORDER, sha, write_json, write_parquet
from stage3.scripts.traffic_state_batch2 import prediction_proxy, exposure_by_route

SEED = "traffic-support-split-v1"


def decompose(frame, cfg):
    x = frame.copy()
    x["strict_state"] = prediction_proxy(x, cfg)
    relaxed = {**cfg, "reference_min_order_days": 1, "reference_min_days": 1}
    x["research_state"] = prediction_proxy(x, relaxed)
    valid_prediction = (np.isfinite(x.pred_pace_p50) & x.pred_pace_p50.gt(0)
        & x.pred_crawl.between(0, 1) & x.pred_stop.between(0, 1)
        & (x.pred_crawl + x.pred_stop).le(1+1e-6))
    no_ref = x.reference_order_days.isna()
    bad_ref = ~np.isfinite(x.reference_pace_q20) | x.reference_pace_q20.le(0)
    n = x.reference_order_days.lt(cfg["reference_min_order_days"])
    d = x.reference_days.lt(cfg["reference_min_days"])
    x["unknown_reason"] = np.select(
        [~valid_prediction, no_ref, bad_ref, n & d, n, d],
        ["INVALID_PREDICTION", "NO_REFERENCE", "INVALID_REFERENCE", "LOW_COUNT_AND_DAYS", "LOW_COUNT_ONLY", "LOW_DAYS_ONLY"],
        default="NOT_UNKNOWN")
    x["reference_support"] = np.select([no_ref, bad_ref, n | d],
        ["NO_REFERENCE", "INVALID_REFERENCE", "LOW_REFERENCE_SUPPORT"], default="MEETS_ORIGINAL_SUPPORT")
    x["pace_ratio"] = x.pred_pace_p50/x.reference_pace_q20
    x["low_speed_share"] = x.pred_crawl+x.pred_stop
    mixed = x.research_state.eq("MIXED_EVIDENCE")
    x["mixed_subtype"] = np.select(
        [mixed & x.pace_ratio.ge(cfg["pace_ratio_congested"]), mixed],
        ["RELATIVE_SLOWDOWN_WITHOUT_HIGH_LOW_SPEED_SHARE", "HIGH_LOW_SPEED_SHARE_WITHOUT_RELATIVE_SLOWDOWN"],
        default="NOT_MIXED")
    assert x.strict_state.eq("UNKNOWN").equals(x.unknown_reason.ne("NOT_UNKNOWN"))
    return x


def size_band(n):
    return pd.cut(n, [0, 9, 29, 99, np.inf], labels=["1-9", "10-29", "30-99", "100+"]).astype(str)


def split_bit(date, order):
    return hashlib.sha256(f"{date}|{order}|{SEED}".encode()).digest()[0] % 2


def run():
    start = time.monotonic()
    out, docs = Path("stage3/output/traffic_state_support"), Path("stage3/docs/traffic_state_support")
    out.mkdir(parents=True, exist_ok=False)
    cfgpath = Path("stage3/config/traffic_state_batch1.json")
    cfg = json.loads(cfgpath.read_text())
    refpath = Path("stage3/output/traffic_state_batch1/train_reference.parquet")
    protected = [cfgpath, refpath, Path("stage3/config/stage3_av_capability_profiles.json"),
                 Path("stage2/output_v5_2/development/M3/epoch_004.pt")]
    before = {str(p): sha(p) for p in protected}
    ref = pd.read_parquet(refpath)
    resource = {"peak_rss_mb": psutil.Process().memory_info().rss/2**20}
    done = threading.Event()
    def monitor():
        while not done.wait(.2):
            resource["peak_rss_mb"] = max(resource["peak_rss_mb"], psutil.Process().memory_info().rss/2**20)
            if time.monotonic()-start > 1800:
                print("HARD_TIMEOUT", flush=True)
                os._exit(124)
    threading.Thread(target=monitor, daemon=True).start()
    sources, pool = [], []
    history = Path(cfg["history_root"])
    history_meta = json.loads((history/"history_store_manifest.json").read_text())
    codes = pd.Series(np.arange(len(ref), dtype=np.int32), index=ref[EDGE])
    for date in cfg["train_dates"][:12]:
        path = history/"events"/f"day={date}.parquet"
        digest = sha(path)
        assert digest == history_meta["event_files"][date]["file_sha256"]
        x = pd.read_parquet(path, columns=[EDGE, ORDER, "availability_timestamp", "observed_sec_per_m", "crawl_time_share", "stop_time_share"])
        end_date = pd.to_datetime(x.availability_timestamp, unit="s", utc=True).dt.tz_convert("Asia/Shanghai").dt.strftime("%Y%m%d")
        valid = (end_date.eq(date) & np.isfinite(x.observed_sec_per_m) & x.observed_sec_per_m.gt(0)
            & x.crawl_time_share.between(0, 1) & x.stop_time_share.between(0, 1)
            & (x.crawl_time_share+x.stop_time_share).le(1+1e-9))
        a = x.loc[valid].groupby([EDGE, ORDER]).observed_sec_per_m.mean().reset_index()
        mapping = {o: split_bit(date, o) for o in a[ORDER].unique()}
        assert a[EDGE].map(codes).notna().all()
        pool.append(pd.DataFrame({"edge_code": a[EDGE].map(codes).to_numpy(np.int32),
            "date": np.full(len(a), int(date), dtype=np.int32), "pace": a.observed_sec_per_m.to_numpy(float),
            "half": a[ORDER].map(mapping).to_numpy(np.int8)}))
        sources.append({"path": str(path), "sha256": digest})
        print(f"Train reference {date}: {len(a)} order-edge-days", flush=True)
        del x, a
        gc.collect()
    numeric = pd.concat(pool, ignore_index=True); pool.clear()
    g = numeric.groupby("edge_code")
    split = g.agg(order_days=("pace", "size"), dates=("date", "nunique"))
    for half in (0, 1):
        sub = numeric.loc[numeric.half.eq(half)].groupby("edge_code").pace
        split[f"q20_{half}"] = sub.quantile(.2)
        split[f"n_{half}"] = sub.size()
    split["size_band"] = size_band(split.order_days)
    split["date_band"] = np.where(split.dates.ge(5), "5+", "1-4")
    split["relative_difference"] = abs(split.q20_0-split.q20_1)/((split.q20_0+split.q20_1)/2)
    split[EDGE] = ref.set_axis(np.arange(len(ref)))[EDGE].reindex(split.index).to_numpy()
    split = split.reset_index(drop=True)
    write_parquet(out/"train_split_reference.parquet", split)
    edge_stats = []
    for key, group in split.groupby(["size_band", "date_band"]):
        v = group.relative_difference.dropna()
        edge_stats.append({"size_band": key[0], "date_band": key[1], "edges": len(group), "comparable_edges": len(v),
            "relative_difference_p50": float(v.median()) if len(v) else None,
            "relative_difference_p90": float(v.quantile(.9)) if len(v) else None})
    del numeric, g, sub; gc.collect()
    heldout_rows = []
    relaxed = {**cfg, "reference_min_order_days": 1, "reference_min_days": 1}
    for date in cfg["train_dates"][12:]:
        path = Path(f"stage3/output/traffic_state_batch1/cells/date={date}.parquet")
        cells = pd.read_parquet(path, columns=[EDGE, "joint_pace", "joint_low_speed_share", "joint_order_count"])
        cells = cells.merge(split, on=EDGE, how="left", validate="many_to_one")
        valid = np.isfinite(cells.joint_pace) & cells.joint_pace.gt(0) & cells.joint_low_speed_share.between(0, 1)
        both = valid & cells.q20_0.notna() & cells.q20_1.notna()
        x = cells.loc[both].copy()
        states = []
        for half in (0, 1):
            probe = pd.DataFrame({"pred_pace_p50": x.joint_pace, "pred_crawl": x.joint_low_speed_share,
                "pred_stop": 0., "reference_pace_q20": x[f"q20_{half}"],
                "reference_order_days": x[f"n_{half}"], "reference_days": 1})
            states.append(prediction_proxy(probe, relaxed))
        x["state_changed"] = states[0].ne(states[1])
        for key, group in x.groupby(["size_band", "date_band"]):
            heldout_rows.append({"date": date, "size_band": key[0], "date_band": key[1],
                "comparable_cells": len(group), "changed_cells": int(group.state_changed.sum())})
        heldout_rows.append({"date": date, "size_band": "ALL", "date_band": "ALL",
            "total_cells": len(cells), "joint_valid_cells": int(valid.sum()),
            "comparable_cells": int(both.sum()), "changed_cells": int(x.state_changed.sum())})
        sources.append({"path": str(path), "sha256": sha(path)})
    write_json(docs/"train_stability.json", {"reference_edges": edge_stats, "heldout_cells": heldout_rows})
    rows, summaries = [], []
    oldmeta = json.loads(Path("stage3/docs/traffic_explanation_v1/summary.json").read_text())
    oldmeta = {r["date"]: r for r in oldmeta["dates"]}
    original_counts = json.loads(Path("stage3/docs/traffic_state_batch2/predicted_state_distribution.json").read_text())["rows"]
    for date in cfg["evaluation_dates"]:
        pp = Path(f"stage3/output/odd_tod/s3/cache/m3_validation/date={date}.parquet")
        rp = Path(f"stage2/output_v4/route_conditioned_dataset/revealed_route_proxy/day={date}.parquet")
        rm = Path(f"stage2/output_v4/route_conditioned_dataset/manifests/revealed_route_proxy/day={date}.json")
        assert sha(pp) == json.loads(pp.with_suffix(".json").read_text())["prediction_sha256"]
        assert sha(rp) == json.loads(rm.read_text())["file_sha256"]
        overlap = Path(f"stage3/output/traffic_state_batch2/route_overlap_date={date}.parquet")
        assert sha(overlap) == oldmeta[date]["source_sha256"]
        ids = pd.read_parquet(overlap, columns=["order_id"]).order_id
        p = pd.read_parquet(pp, columns=["order_id", "traversal_id", "pred_pace_p50", "pred_crawl", "pred_stop", "travel_time_p50_s"])
        route = pd.read_parquet(rp, columns=["order_id", "traversal_id", EDGE, "route_sequence"])
        p = p.loc[p.order_id.isin(ids)].merge(route, on=["order_id", "traversal_id"], validate="one_to_one")
        p = decompose(p.merge(ref, on=EDGE, how="left", validate="many_to_one"), cfg)
        assert p.order_id.nunique() == len(ids)
        assert np.isfinite(p.travel_time_p50_s).all() and p.travel_time_p50_s.gt(0).all()
        count = p.groupby("strict_state").size().to_dict()
        assert count == {r["state"]: r["tokens"] for r in original_counts if r["date"] == date}
        total = float(p.travel_time_p50_s.astype(float).sum())
        for kind, keys in [("unknown_reason", ["unknown_reason"]), ("state_support", ["research_state", "reference_support"]),
                           ("mixed", ["strict_state", "mixed_subtype", "reference_support"])]:
            table = p.assign(weight=p.travel_time_p50_s.astype(float)).groupby(keys, dropna=False).agg(tokens=("weight", "size"), time_s=("weight", "sum")).reset_index()
            for row in table.to_dict("records"):
                rows.append({"date": date, "kind": kind, **row, "total_time_s": total, "time_share": row["time_s"]/total})
        p["state"] = p.research_state
        routes = exposure_by_route(p, "travel_time_p50_s", "research_")
        routes["date"] = date
        for label in ("LOW_REFERENCE_SUPPORT", "NO_REFERENCE", "INVALID_REFERENCE"):
            amounts = p.travel_time_p50_s.where(p.reference_support.eq(label), 0).groupby(p.order_id).sum()
            routes["support_"+label.lower()+"_share"] = routes.order_id.map(amounts)/routes.research_total_time_s
        write_parquet(out/f"routes_date={date}.parquet", routes)
        write_parquet(out/f"tokens_date={date}.parquet", p.drop(columns="state"))
        summaries.append({"date": date, "routes": len(routes), "tokens": len(p), "predicted_time_s": total,
            "strict_unknown_time_s": float(p.loc[p.strict_state.eq("UNKNOWN"), "travel_time_p50_s"].astype(float).sum()),
            "research_unknown_time_s": float(p.loc[p.research_state.eq("UNKNOWN"), "travel_time_p50_s"].astype(float).sum())})
        sources.extend({"path": str(path), "sha256": sha(path)} for path in (pp, rp, overlap))
        print(f"Validation {date}: {len(routes)} routes; strict state reproduced; research states written", flush=True)
        del p, route, routes; gc.collect()
    write_json(docs/"decomposition.json", {"rows": rows})
    assert before == {str(p): sha(p) for p in protected}
    summary = {"status": "COMPLETED_RESEARCH_SUPPORT_SEPARATION", "protocol_commit": "f1af90e", "seed": SEED,
        "days": summaries, "sources": sources, "protected_sha256": before,
        "runtime_s": time.monotonic()-start, **resource, "gpu_used": False, "training": False,
        "dispatch": False, "test31_read": False, "production_threshold_selected": False}
    done.set()
    write_json(docs/"summary.json", summary)
    write_json(out/"summary.json", summary)
    print(json.dumps({k:summary[k] for k in ("status", "runtime_s", "peak_rss_mb")}), flush=True)


if __name__ == "__main__":
    run()
