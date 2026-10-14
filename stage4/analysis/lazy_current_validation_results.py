"""Read-only arithmetic reconciliation of the single saved four-graph run.

No routing, re-solving, simulation, forecasts, or new sample selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .competitive_chain_benchmark import atomic_json

OUT = Path("stage4/output/capability_chain_planning/lazy_current_validation_v1/run_001")
DOC = Path("stage4/docs/capability_chain_planning/lazy_current_validation_v1")
CASES = ("H0730", "H0830", "H1700", "H1800")


def analyse(root):
    summary = json.loads((root/OUT/"summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "FOUR_FIXED_GRAPHS_COMPLETE" or len(summary["rows"]) != 8:
        raise ValueError("the fixed run is not complete")
    pairs, witnesses, source_equal = [], 0, 0
    modes = ("EAGER", "LAZY_CERTIFIED")
    for case_id in CASES:
        graph = json.loads((root/OUT/case_id/"graph.json").read_text(encoding="utf-8"))
        case = graph["case"]
        known = {a["action_id"]:a for a in graph["known"]}
        unknown = {a["action_id"]:a for a in graph["unknown"]}
        all_vehicles = {a["vehicle_id"] for a in graph["known"] if a["kind"] == "WAIT"}
        eager = case["modes"]["EAGER"]
        lazy = case["modes"]["LAZY_CERTIFIED"]
        if set(eager["outcomes"]) != set(unknown):
            raise ValueError("all-check comparator has omitted unknown candidates")
        for aid, answer in lazy["outcomes"].items():
            if eager["outcomes"].get(aid) != answer:
                raise ValueError("oracle order/subset disagreement")
            source_equal += 1
        for mode in modes:
            saved = case["modes"][mode]
            selected, vehicles, requests = [], [], []
            for aid in saved["selected_ids"]:
                if aid in known:
                    action = known[aid]
                else:
                    bound, truth = unknown[aid], saved["outcomes"].get(aid)
                    if not truth or not truth["admitted"]:
                        raise ValueError("selected candidate was not validated")
                    action = dict(bound,distance_m=truth["distance_m"],units=truth["units"])
                selected.append(action)
                vehicles.append(action["vehicle_id"])
                if action["kind"] == "SERVE":
                    requests.append(action["request_id"])
            if (len(vehicles) != len(set(vehicles)) or set(vehicles) != all_vehicles
                    or len(requests) != len(set(requests))):
                raise ValueError("resource/request conservation failed")
            score = [sum(a["critical"] for a in selected),len(requests),
                sum(a["carry_over"] for a in selected),
                sum(3*(a["kind"]=="SERVE")+a["units"] for a in selected),
                -sum(int(round(a["distance_m"]*1000)) for a in selected)]
            if score != saved["result"]["objective"]:
                raise ValueError("saved objective arithmetic differs")
            witnesses += 1
        e,l = eager["result"],lazy["result"]
        pairs.append(dict(case_id=case_id,unknown_C_arcs=len(unknown),
            objective_equal=e["objective"]==l["objective"],
            both_certified=e["full_objective_certified"] and l["full_objective_certified"],
            eager_queries=e["oracle_calls"],lazy_queries=l["oracle_calls"],
            eager_scalar=e["scalar_route_calls"],lazy_scalar=l["scalar_route_calls"],
            eager_edge_walk=e["edge_walk_calls"],lazy_edge_walk=l["edge_walk_calls"],
            eager_s=e["mode_time_s"],lazy_s=l["mode_time_s"],
            accepted_unknown_C=sum(bool(x["admitted"]) for x in eager["outcomes"].values()),
            common_setup_s=case["common_setup_time_s"],
            common_matrix_backend_queries=case["common_matrix_backend_queries"]))
    totals = {key:sum(row[key] for row in pairs) for key in (
        "eager_queries","lazy_queries","eager_scalar","lazy_scalar","eager_edge_walk", "lazy_edge_walk",
        "eager_s","lazy_s","accepted_unknown_C","common_setup_s","common_matrix_backend_queries")}
    # A descriptive sum of this fixed run, not a cross-day speedup estimate.
    result = dict(status="SAVED_ARITHMETIC_RECONCILED",verification_status="ANALYZED_NOT_ROUTING_RERUN",
        code_sha=summary["code_sha"],objective_pairs_equal=sum(p["objective_equal"] for p in pairs),
        both_certified_pairs=sum(p["both_certified"] for p in pairs),selected_witnesses_checked=witnesses,
        lazy_oracle_answers_equal=source_equal,totals=totals,pairs=pairs,
        cases_with_unknown_C=sum(p["unknown_C_arcs"]>0 for p in pairs),
        whole_day_speedup_demonstrated=False,actual_admitted_C_arc_present=totals["accepted_unknown_C"]>0,
        routing_rerun=False,solver_rerun=False)
    atomic_json(root/DOC/"analysis.json",result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("."))
    print(json.dumps(analyse(parser.parse_args().root.resolve()),ensure_ascii=False,indent=2))
