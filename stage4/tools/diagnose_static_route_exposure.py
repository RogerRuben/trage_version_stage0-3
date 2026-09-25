"""Read-only Test31 max/mean/tail diagnostic; never changes dispatch inputs.

Run from the repository root with the stage0-valhalla Python environment.
Only aggregate JSON is exported; original order/complex identifiers stay local.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = Path("stage3/output/odd_tod")
SOURCES = {
    "profile": Path("stage3/config/stage3_av_capability_profiles.json"),
    "encounters": BASE / "s4/test31_route_complex_encounters.parquet",
    "descriptors": BASE / "s4/test31_original_route_descriptors.parquet",
    "complexes": BASE / "s2b/final/stage3_intersection_complexes.parquet",
    "boundary": BASE / "s2b/final/stage3_edge_complex_boundary_index.parquet",
    "edges": BASE / "s2a/stage3_full_network_edges.parquet",
}
DIMENSIONS = {
    "A": "external_physical_connection_count",
    "M": "topological_movement_count",
    "D": "road_class_diversity",
    "L": "internal_length_m",
}


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def distribution(values: pd.Series) -> dict:
    values = values.dropna()
    if not len(values):
        return {"n": 0}
    return {"n": int(len(values)), "mean": float(values.mean()), **{
        name: float(values.quantile(q))
        for name, q in [("min", 0), ("p25", .25), ("p50", .5),
                        ("p75", .75), ("p90", .9), ("max", 1)]
    }}


def tail_summary(count: pd.Series, total: pd.Series, mean: pd.Series,
                 maximum: pd.Series, cap: float) -> dict:
    tail = count / total
    exceeding = count.gt(0)
    assert np.array_equal(exceeding.to_numpy(), maximum.gt(cap).to_numpy())
    assert bool((mean <= maximum + 1e-10).all())
    assert bool(tail.between(0, 1).all())
    return {
        "route_count": int(len(total)), "cap": float(cap),
        "max_above_cap": int(exceeding.sum()),
        "mean_above_cap": int(mean.gt(cap).sum()),
        "max_above_but_mean_at_or_below_cap": int((exceeding & mean.le(cap)).sum()),
        "exactly_one_exceeding_encounter": int(count.eq(1).sum()),
        "multiple_exceeding_encounters": int(count.gt(1).sum()),
        "all_encounters_above_cap": int(count.eq(total).sum()),
        "tail_share_among_max_exceeding": distribution(tail[exceeding]),
        "max_over_cap": distribution(maximum / cap),
        "mean_over_cap": distribution(mean / cap),
        "tail_bins_all_routes": {
            "zero": int(tail.eq(0).sum()),
            "gt0_le10pct": int((tail.gt(0) & tail.le(.1)).sum()),
            "gt10_le25pct": int((tail.gt(.1) & tail.le(.25)).sum()),
            "gt25_le50pct": int((tail.gt(.25) & tail.le(.5)).sum()),
            "gt50_le100pct": int(tail.gt(.5).sum()),
        },
    }


def main() -> None:
    started = time.perf_counter()
    before = {k: sha(ROOT / v) for k, v in SOURCES.items()}
    profile = json.loads((ROOT / SOURCES["profile"]).read_text(encoding="utf-8"))
    read = lambda key, columns: pd.read_parquet(ROOT / SOURCES[key], columns=columns)
    encounters = read("encounters", ["date", "order_id", "movement_occurrence_index",
                                     "intersection_complex_uid", "movement_lookup_status"])
    assert set(encounters.date.astype(str)) == {"20161031"}
    assert not encounters.duplicated(["order_id", "movement_occurrence_index"]).any()
    descriptor = read("descriptors", ["order_id", "unresolved_token_count",
        "reverse_overlay_token_count", "unresolved_movement_lookup_count",
        "resolved_complex_encounter_count", *[f"route_max_{d}_c" for d in DIMENSIONS]])
    assert len(descriptor) == 30000 and descriptor.order_id.is_unique
    static = read("complexes", ["intersection_complex_uid", DIMENSIONS["A"],
                              DIMENSIONS["M"], DIMENSIONS["L"]])
    assert len(static) == 43685 and static.intersection_complex_uid.is_unique
    boundary = read("boundary", ["intersection_complex_uid", "stage3_edge_uid", "boundary_role"])
    edges = read("edges", ["stage3_edge_uid", "valhalla_road_class"])
    assert edges.stage3_edge_uid.is_unique
    boundary = boundary[boundary.boundary_role.isin(["INCOMING", "OUTGOING"])].drop_duplicates(
        ["intersection_complex_uid", "stage3_edge_uid"])
    boundary = boundary.merge(edges, on="stage3_edge_uid", how="left", validate="many_to_one")
    assert not boundary.valhalla_road_class.isna().any()
    diversity = boundary.groupby("intersection_complex_uid").valhalla_road_class.nunique()
    static[DIMENSIONS["D"]] = static.intersection_complex_uid.map(diversity)
    static = static.rename(columns={value: key for key, value in DIMENSIONS.items()})
    detail = encounters.merge(static, on="intersection_complex_uid", how="left", validate="many_to_one")
    assert not detail[list(DIMENSIONS)].isna().any().any()
    assert bool((detail[list(DIMENSIONS)] >= 0).all().all())
    assert set(detail.order_id).issubset(set(descriptor.order_id))
    desc = descriptor.set_index("order_id")
    grouped = detail.groupby("order_id", sort=True)
    total = grouped.size()
    maximum = grouped[list(DIMENSIONS)].max()
    means = grouped[list(DIMENSIONS)].mean()
    p90 = grouped[list(DIMENSIONS)].quantile(.9, interpolation="higher")
    assert np.array_equal(total.to_numpy(), desc.loc[total.index, "resolved_complex_encounter_count"].to_numpy())
    assert (desc.loc[~desc.index.isin(total.index), "resolved_complex_encounter_count"] == 0).all()
    reconciliation = {}
    for dim in DIMENSIONS:
        original = desc[f"route_max_{dim}_c"]
        actual = maximum[dim].reindex(desc.index)
        same = np.isclose(original, actual, rtol=0, atol=1e-10, equal_nan=True)
        reconciliation[dim] = {"orders_checked": len(desc), "mismatches": int((~same).sum())}
        assert same.all(), f"Frozen route max mismatch: {dim}"
    clean = desc.unresolved_token_count.eq(0) & desc.reverse_overlay_token_count.eq(0) & desc.unresolved_movement_lookup_count.eq(0)
    scopes = {"all_observed_encounter_routes": total.index,
              "full_network_and_movement_resolved": total.index.intersection(desc.index[clean])}
    rows = []
    joint = []
    for pr in profile["profiles"]:
        caps = {d: float(pr["static_caps"][name]) for d, name in DIMENSIONS.items()}
        assert all(v > 0 for v in caps.values())
        flags = detail[list(DIMENSIONS)].gt(pd.Series(caps))
        flags["order_id"] = detail.order_id
        counts = flags.groupby("order_id")[list(DIMENSIONS)].sum()
        any_flag = flags[list(DIMENSIONS)].any(axis=1)
        any_count = any_flag.groupby(detail.order_id).sum()
        for scope, idx in scopes.items():
            for dim in DIMENSIONS:
                row = tail_summary(counts.loc[idx, dim], total.loc[idx], means.loc[idx, dim],
                                   maximum.loc[idx, dim], caps[dim])
                row.update(profile=pr["profile_id"], dimension=dim, scope=scope,
                           p90_above_cap=int(p90.loc[idx, dim].gt(caps[dim]).sum()))
                rows.append(row)
            values = any_count.loc[idx]
            joint.append({"profile": pr["profile_id"], "scope": scope,
                          "route_count": len(idx), "any_dimension_exceeds": int(values.gt(0).sum()),
                          "exactly_one_exceeding_encounter_any_dimension": int(values.eq(1).sum()),
                          "tail_share_among_exceeding": distribution((values / total.loc[idx])[values.gt(0)])})
    after = {k: sha(ROOT / v) for k, v in SOURCES.items()}
    assert before == after
    # Bounded synthetic check: an isolated extreme and repeated extremes differ in tail,
    # even when their maximum is identical. This does not authorize a mean-based gate.
    fixture = tail_summary(pd.Series([1, 3]), pd.Series([4, 4]),
                           pd.Series([3., 7.]), pd.Series([9., 9.]), 5.)
    assert fixture["max_above_but_mean_at_or_below_cap"] == 1
    assert fixture["exactly_one_exceeding_encounter"] == 1
    result = {
        "status": "READ_ONLY_STATIC_EXPOSURE_DIAGNOSTIC_COMPLETE",
        "code_sha_before_diagnostic": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "sources": {k: {"path": v.as_posix(), "sha256": before[k]} for k, v in SOURCES.items()},
        "scope": "Test31 original historical routes; not fallback routes or matched dispatch arcs",
        "weighting": "One encounter one observation; repeated visits retained; not duration or distance weighted",
        "tail_definition": "Fraction of observed encounters strictly above the existing frozen profile cap",
        "p90_definition": "Within-route empirical quantile with interpolation=higher; descriptive only",
        "route_count": len(desc), "encounter_rows": len(detail),
        "unique_observed_complexes": int(detail.intersection_complex_uid.nunique()),
        "routes_without_encounters": int((~desc.index.isin(total.index)).sum()),
        "full_network_and_movement_resolved_routes_with_encounters": len(scopes["full_network_and_movement_resolved"]),
        "routes_with_unresolved_tokens": int(desc.unresolved_token_count.gt(0).sum()),
        "routes_with_reverse_overlay_tokens": int(desc.reverse_overlay_token_count.gt(0).sum()),
        "routes_with_unresolved_movement_lookup": int(desc.unresolved_movement_lookup_count.gt(0).sum()),
        "encounters_per_route": distribution(total),
        "unique_complexes_per_route": distribution(grouped.intersection_complex_uid.nunique()),
        "repeated_complex_visit_rows": int(len(detail) - len(detail.drop_duplicates(["order_id", "intersection_complex_uid"]))),
        "max_reconciliation": reconciliation, "input_hashes_unchanged": before == after,
        "synthetic_check": "PASS", "dimension_profile_comparison": rows, "joint_static_summary": joint,
        "runtime_s": time.perf_counter() - started,
    }
    try:
        import psutil
        mem = psutil.Process().memory_info()
        result["peak_working_set_mb"] = getattr(mem, "peak_wset", mem.rss) / 2**20
    except ImportError:
        result["peak_working_set_mb"] = None
    output = ROOT / "stage4/docs/static_exposure_diagnostic/summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["status", "route_count", "encounter_rows", "routes_without_encounters", "full_network_and_movement_resolved_routes_with_encounters", "runtime_s", "peak_working_set_mb"]}, indent=2))


if __name__ == "__main__":
    main()
