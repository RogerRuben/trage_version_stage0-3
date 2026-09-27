"""Descriptive replacement bounds, not an operational policy or evaluator."""
from __future__ import annotations

import gc
import argparse
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd
import psutil

from stage3.odd_tod.traffic_explanation import build_explanations, SOURCE_COLUMNS
from stage3.scripts.traffic_state_batch1 import sha, write_json, write_parquet
from stage3.scripts.traffic_state_batch2 import METRICS, dynamic_ratios

PROFILES = ("C", "M", "A")
TOL = 1e-6  # float32 upstream roundoff only
STATES = ("RETAINED_VARIABILITY_EXCEEDS", "KNOWN_EXPOSURE_EXCEEDS",
          "WITHIN_BOUND", "UNRESOLVED_BOUND")


def bounds(explanation: pd.DataFrame, profile: str):
    if profile not in PROFILES:
        raise ValueError("unknown profile")
    z = (explanation.traffic_unknown_share.astype(float)
         + explanation.traffic_mixed_evidence_share.astype(float)).to_numpy()
    severe = explanation.traffic_severe_congestion_proxy_share.to_numpy(float)
    u = (severe + explanation.traffic_congested_proxy_share.to_numpy(float)
         if profile == "C" else severe if profile == "M" else np.zeros(len(z)))
    return u, z, np.minimum(1., u + z)


def classify(u, upper, variability_exceeds, budget):
    if not 0 <= budget <= 1:
        raise ValueError("budget outside unit interval")
    if not (np.isfinite(u).all() and np.isfinite(upper).all()
            and (u >= 0).all() and (upper + TOL >= u).all() and (upper <= 1).all()):
        raise ValueError("invalid exposure bounds")
    result = np.where(upper <= budget + TOL, "WITHIN_BOUND", "UNRESOLVED_BOUND").astype(object)
    result[u > budget + TOL] = "KNOWN_EXPOSURE_EXCEEDS"
    result[np.asarray(variability_exceeds, dtype=bool)] = "RETAINED_VARIABILITY_EXCEEDS"
    return result


def run():
    start = time.monotonic()
    config_path = Path("stage3/config/traffic_state_offline_comparison.json")
    cfg = json.loads(config_path.read_text())
    assert cfg["dates"] == ["20161025", "20161026", "20161027"]
    grid = cfg["budget_grid"]
    budgets = np.round(np.arange(grid["start"], grid["stop"] + grid["step"]/2, grid["step"]), 8)
    assert len(budgets) == 101 and budgets[0] == 0 and budgets[-1] == 1
    docs = Path("stage3/docs/traffic_state_offline_comparison")
    out = Path("stage3/output/traffic_state_offline_comparison")
    out.mkdir(parents=True, exist_ok=False)
    profile_path = Path("stage3/config/stage3_av_capability_profiles.json")
    profile = json.loads(profile_path.read_text())
    protected = [profile_path, config_path, Path("stage3/config/traffic_state_batch1.json"),
                 Path("stage2/output_v5_2/development/M3/epoch_004.pt")]
    before = {str(p): sha(p) for p in protected}
    done = threading.Event()
    resource = {"peak_rss_mb": psutil.Process().memory_info().rss/2**20}
    def monitor():
        while not done.wait(.2):
            resource["peak_rss_mb"] = max(resource["peak_rss_mb"], psutil.Process().memory_info().rss/2**20)
            if time.monotonic() - start > cfg["hard_timeout_s"]:
                print("HARD_TIMEOUT", flush=True)
                os._exit(124)
    threading.Thread(target=monitor, daemon=True).start()
    curves, distributions, examples, sources, counts = [], [], [], [], []
    side_manifest = json.loads(Path("stage3/docs/traffic_explanation_v1/summary.json").read_text())
    expected = {r["date"]: r for r in side_manifest["dates"]}
    family_cols = [f"{family}_{p}" for p in PROFILES
                   for family in ("rho_dynamic", "crawl_stop_exceeds", "variability_exceeds")]
    for date in cfg["dates"]:
        source = Path(f"stage3/output/traffic_state_batch2/route_overlap_date={date}.parquet")
        side_path = Path(f"stage3/output/traffic_explanation_v1/date={date}.parquet")
        source_sha, side_sha = sha(source), sha(side_path)
        assert source_sha == expected[date]["source_sha256"]
        assert side_sha == expected[date]["output_sha256"]
        day = pd.read_parquet(source, columns=["date", "order_id", *SOURCE_COLUMNS, *METRICS, *family_cols])
        assert len(day) == expected[date]["rows"] and day.date.astype(str).eq(date).all()
        assert np.isfinite(day[METRICS].to_numpy(float)).all()
        recomputed = dynamic_ratios(day, profile)
        pd.testing.assert_frame_equal(day[family_cols], recomputed[family_cols])
        side = pd.read_parquet(side_path)
        pd.testing.assert_frame_equal(build_explanations(day), side)
        records = pd.DataFrame({"date": date, "order_id": day.order_id.astype(str),
                                "selected_route_reference": side.selected_route_reference})
        for c in ["traffic_unknown_share", "traffic_mixed_evidence_share", "traffic_total_time_s",
                  "traffic_longest_congested_run_s"]:
            records[c] = side[c].to_numpy()
        profile_masks = {}
        for p in PROFILES:
            u, z, upper = bounds(side, p)
            var = day[f"variability_exceeds_{p}"].to_numpy(bool)
            cs = day[f"crawl_stop_exceeds_{p}"].to_numpy(bool)
            baseline = ~(var | cs)
            assert np.array_equal(baseline, day[f"rho_dynamic_{p}"].le(1).to_numpy())
            zero = classify(u, upper, var, 0.)
            records[f"known_outside_{p}"] = u
            records[f"outside_upper_{p}"] = upper
            records[f"baseline_dynamic_pass_{p}"] = baseline
            records[f"retained_variability_exceeds_{p}"] = var
            records[f"zero_budget_dynamic_bound_{p}"] = zero
            prev_certain, prev_possible = np.zeros(len(day), bool), np.zeros(len(day), bool)
            masks = []
            for b in budgets:
                state = classify(u, upper, var, float(b))
                certain = state == "WITHIN_BOUND"
                possible = np.isin(state, ["WITHIN_BOUND", "UNRESOLVED_BOUND"])
                assert not (prev_certain & ~certain).any() and not (prev_possible & ~possible).any()
                prev_certain, prev_possible = certain, possible
                masks.append((certain, possible))
                row = {"date": date, "profile": p, "budget": float(b), "routes": len(day),
                       "baseline_dynamic_pass": int(baseline.sum()),
                       "traffic_only_known_exceeds": int((u > b + TOL).sum()),
                       "traffic_only_within_bound": int((upper <= b + TOL).sum())}
                for old in (False, True):
                    for name in STATES:
                        row[f"baseline_{'pass' if old else 'fail'}__{name}"] = int(((baseline == old) & (state == name)).sum())
                row["dynamic_within_lower_count"] = int(certain.sum())
                row["dynamic_within_upper_count"] = int(possible.sum())
                assert sum(row[f"baseline_{old}__{name}"] for old in ("pass", "fail") for name in STATES) == len(day)
                curves.append(row)
            profile_masks[p] = masks
            for old_cs in (False, True):
                mask = cs == old_cs
                row = {"date": date, "profile": p, "baseline_crawl_stop_exceeds": old_cs, "routes": int(mask.sum())}
                for name, values in {"known_outside": u, "upper": upper, "unresolved": z,
                    "unknown": side.traffic_unknown_share.to_numpy(float),
                    "mixed": side.traffic_mixed_evidence_share.to_numpy(float),
                    "longest_congested_run_s": side.traffic_longest_congested_run_s.to_numpy(float)}.items():
                    for q in (.5, .9, .99):
                        row[f"{name}_p{int(q*100)}"] = float(np.quantile(values[mask], q)) if mask.any() else None
                distributions.append(row)
            # Stable examples of disagreements/uncertainty, not representative evidence.
            ids = day.order_id.astype(str)
            hashes = ids.map(lambda oid: hashlib.sha256(f"{date}|{oid}|traffic-comparison-v1".encode()).hexdigest())
            for old in (False, True):
                for state in STATES:
                    if (old and state == "WITHIN_BOUND") or (not old and state in STATES[:2]):
                        continue
                    eligible = np.flatnonzero((baseline == old) & (zero == state))
                    selected = sorted(eligible, key=lambda i: hashes.iloc[i])[:2]
                    for i in selected:
                        examples.append({"date": date, "order_id": ids.iloc[i], "profile": p,
                            "baseline_dynamic_pass": old, "zero_budget_bound": state,
                            "known_outside": float(u[i]), "upper": float(upper[i]),
                            "unknown": float(side.traffic_unknown_share.iloc[i]),
                            "mixed": float(side.traffic_mixed_evidence_share.iloc[i]),
                            "longest_congested_run_s": float(side.traffic_longest_congested_run_s.iloc[i]),
                            "semantic_cause": "NOT_IDENTIFIED_FROM_AGGREGATE_PREDICTIONS"})
        for a, b in (("C", "M"), ("M", "A")):
            for ma, mb in zip(profile_masks[a], profile_masks[b]):
                assert not (ma[0] & ~mb[0]).any() and not (ma[1] & ~mb[1]).any()
        output = out/f"date={date}.parquet"
        write_parquet(output, records)
        assert sha(source) == source_sha and sha(side_path) == side_sha
        sources.append({"date": date, "source_sha256": source_sha, "sidecar_sha256": side_sha,
                        "output_sha256": sha(output)})
        counts.append({"date": date, "routes": len(day)})
        print(f"{date}: {len(day)} routes; 101 budgets; baseline/nesting/monotonicity PASS", flush=True)
        del day, recomputed, side, records, profile_masks, masks
        gc.collect()
    curve = pd.DataFrame(curves)
    all_days = curve.drop(columns="date").groupby(["profile", "budget"], as_index=False).sum()
    all_days["date"] = "ALL_VALIDATION"
    write_json(docs/"budget_curves.json", {"rows": pd.concat([curve, all_days], ignore_index=True).to_dict("records")})
    write_json(docs/"distributions.json", {"rows": distributions})
    write_json(docs/"case_locator.json", {"sampling": "2 lowest stable hashes per daily profile zero-budget change cell", "rows": examples})
    assert before == {str(p): sha(p) for p in protected}
    summary = {"status": "COMPLETED_OFFLINE_DESCRIPTIVE_ONLY", "prereg_commit": "e13af72",
        "dates": counts, "routes": sum(r["routes"] for r in counts), "sources": sources,
        "protected_sha256": before, "budget_count": len(budgets),
        "zero_budget": all_days.loc[all_days.budget.eq(0)].to_dict("records"),
        "baseline_recomputed_exact": True, "nestedness_valid": True, "budget_monotonicity_valid": True,
        "production_budget_selected": False, "unknown_policy_selected": False,
        "dispatch": False, "training": False, "test31_read": False, "gpu_used": False,
        "runtime_s": time.monotonic()-start, **resource}
    done.set()
    write_json(docs/"summary.json", summary)
    write_json(out/"summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("status", "routes", "runtime_s", "peak_rss_mb")}), flush=True)


def plot_existing():
    """Render already computed counts without rerunning the comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    docs = Path("stage3/docs/traffic_state_offline_comparison")
    rows = pd.DataFrame(json.loads((docs/"budget_curves.json").read_text())["rows"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, p in zip(axes, PROFILES):
        data = rows.loc[rows.date.eq("ALL_VALIDATION") & rows.profile.eq(p)].sort_values("budget")
        x = data.budget.to_numpy(float)
        lower = data.dynamic_within_lower_count.to_numpy(float)/data.routes.to_numpy(float)
        upper = data.dynamic_within_upper_count.to_numpy(float)/data.routes.to_numpy(float)
        ax.fill_between(x, lower, upper, color="#c7dce9", label="Unresolved-state envelope")
        ax.plot(x, lower, color="#176484", label="Lower: all unresolved outside")
        ax.plot(x, upper, color="#176484", linestyle="--", label="Upper: all unresolved inside")
        ax.axhline(float(data.baseline_dynamic_pass.iloc[0]/data.routes.iloc[0]), color="#b34e29",
                   linestyle=":", label="Frozen all-12 dynamic pass")
        ax.set(title=f"Profile {p}", xlabel="Diagnostic outside-exposure budget", xlim=(0, 1), ylim=(0, 1))
    axes[0].set_ylabel("Fraction of 29,851 complete Validation routes")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Replacement bounds, not dispatch outcomes or confidence intervals", fontsize=11)
    fig.tight_layout()
    fig.savefig(docs/"replacement_bounds.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if not args.plot_only:
        run()
    plot_existing()
