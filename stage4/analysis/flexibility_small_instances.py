"""Hand-checkable dispatch contrasts, including an adverse forecast example."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import psutil

from stage3.odd_tod.research_compatibility import ResearchMovement, compatible_profiles
from stage4.dispatch.flexibility_model import (
    CurrentPickup, FuturePickup, Problem, Request, Scenario, Vehicle,
    score_fixed_action, solve_dispatch,
)


POLICIES = ("MYOPIC", "AV_FIRST", "LOOKAHEAD")
NC = {"NC": 1.0, "ML": 0.0, "MR": 0.0, "CG": 0.0, "SC": 0.0, "U": 0.0}


def small_case(name):
    """Fixtures are constructed before execution, not fitted to replay results."""
    vehicles = (Vehicle(1, "HV", "HV_START", 0, 1200),
                Vehicle(2, "C", "AV_START", 0, 1200))
    common = compatible_profiles((ResearchMovement("LEFT", True, "SYNTHETIC"),), NC)
    hv_only = compatible_profiles((ResearchMovement("LEFT", False, "SYNTHETIC"),), NC)
    # An unsignalized left turn is M/A-compatible, but the only AV here is C.
    current = Request(10, 0, 300, 300, "CURRENT_DROPOFF", common)
    pickups = (CurrentPickup(1, 10, 0), CurrentPickup(2, 10, 30))
    reserve_request = Request(20, 60, 90, 120, "FUTURE_DROPOFF", hv_only)
    location_request = Request(20, 60, 90, 120, "FUTURE_DROPOFF", common)

    def scenario(label, request, hv_eta, av_eta, asof=0):
        estimates = []
        for vehicle, eta in zip(vehicles, (hv_eta, av_eta)):
            estimates.extend((
                FuturePickup(vehicle.vehicle_id, None, request.request_id, eta, vehicle.ready_position),
                FuturePickup(vehicle.vehicle_id, 10, request.request_id, eta, "CURRENT_DROPOFF"),
            ))
        return Scenario(label, 1.0, asof, (request,), tuple(estimates))

    reserve = scenario("FORECAST_HV_DEPENDENT", reserve_request, 10, 0)
    location = scenario("FORECAST_AV_NEARBY", location_request, 500, 0)
    if name == "flexibility_reservation":
        forecast, realized = reserve, reserve
    elif name == "location_counterexample":
        forecast, realized = location, location
    elif name == "forecast_error_counterexample":
        forecast, realized = reserve, scenario("REALIZED_AV_NEARBY", location_request, 500, 0, 60)
    else:
        raise ValueError("unrecognized fixture")
    return Problem(0, vehicles, (current,), pickups, (forecast,)), realized


def signal_input_inventory(release):
    """Read only a sealed positive union; never run HTTP or impute negatives."""
    release = Path(release)
    seal = json.loads((release / "SEAL.json").read_text(encoding="utf-8"))
    if seal.get("status") != "FINALIZED" or not seal.get("http_closed"):
        raise ValueError("signal input is not the user-specified sealed release")
    manifest = release / "manifest.json"
    sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if sha != seal["manifest_sha256"]:
        raise ValueError("sealed signal manifest digest mismatch")
    tables = {}
    for filename in ("xian_signalized_intersections_final.csv", "map_or_amap_positive_final.csv"):
        path = release / "results" / filename
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))  # Only 699 positions, no route matrix.
        ids = {row["intersection_id"] for row in rows}
        if len(ids) != len(rows) or any(row["signalized"] != "1" for row in rows):
            raise ValueError("signal union identity/presence mismatch")
        tables[filename] = {
            "path": str(path.resolve()), "row_count": len(rows),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    full = tables["xian_signalized_intersections_final.csv"]["row_count"]
    supported = tables["map_or_amap_positive_final.csv"]["row_count"]
    if full != seal["final_signal_positions"] or supported != seal["map_or_amap_supported_positions"]:
        raise ValueError("signal release counts disagree with seal")
    return {
        "release": str(release.resolve()), "manifest_sha256": sha,
        "full_positive_union_count": full, "map_or_amap_supported_count": supported,
        "legacy_only_count": full - supported, "http_closed": True,
        "unlisted_locations_are_not_proven_unsignalized": True,
        "evidence_period_is_not_2016_field_truth": True,
        "canonical_complex_mapping_completed": False,
        "tables": tables,
    }


def run_contrasts():
    rows = []
    for name in ("flexibility_reservation", "location_counterexample", "forecast_error_counterexample"):
        problem, realized = small_case(name)
        for policy in POLICIES:
            decision = solve_dispatch(problem, policy)
            evaluation = score_fixed_action(problem, decision.selected_pairs, realized)
            rows.append({
                "case": name, "policy": policy, "decision": asdict(decision),
                "realized_next_service_score": evaluation.expected_total_service_count,
                "realized_continuation": asdict(evaluation),
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("stage4/docs/flexibility_dispatch/small_instances_result.json"))
    parser.add_argument("--signal-release", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    inventory = signal_input_inventory(args.signal_release) if args.signal_release else None
    rows = run_contrasts()
    peak = getattr(psutil.Process().memory_info(), "peak_wset", psutil.Process().memory_info().rss)
    output = {
        "status": "COMPLETE", "scope": "SYNTHETIC_OR_CORRECTNESS_AND_COUNTEREXAMPLES_ONLY",
        "native_fleetpy_dynamic_window_run": False,
        "signal_input": inventory, "rows": rows,
        "runtime_s": time.perf_counter() - started, "peak_rss_mb": peak / 1024**2,
        "gpu_used": False, "dense_request_vehicle_matrix_used": False,
        "continuation_rule": "SAME_OPTIMAL_ONE_NEXT_SERVICE_CONTINUATION_FOR_ALL_POLICIES",
        "scientific_claim": "EXISTENCE_OF_BENEFIT_AND_FAILURE_CASES_NOT_EMPIRICAL_SUPERIORITY",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: output[k] for k in ("status", "runtime_s", "peak_rss_mb", "scientific_claim")}))


if __name__ == "__main__":
    main()
