"""Bounded sparse synthetic solver timing, NOT a native/scientific experiment."""
import argparse
import json
from pathlib import Path
import random
from statistics import mean
from time import perf_counter

import psutil

from stage4.analysis.capability_chain_instances import atomic_json
from stage4.dispatch.scheme_a_city_master import CurrentAction
from stage4.dispatch.scheme_a_current_flow import solve_current_flow


def check(root):
    started = perf_counter()
    runs = []
    for seed in range(5):
        rng, actions, units = random.Random(20261011 + seed), [], {}
        for vid in range(200):
            wait = CurrentAction(f"V{vid}:WAIT", vid, "WAIT", None)
            actions.append(wait)
            units[wait.action_id] = 0
            for rid in rng.sample(range(150), 3):
                action = CurrentAction(f"V{vid}:SERVE:{rid}", vid, "SERVE", rid,
                    rid % 11 == 0, rid % 3 == 0, float(rng.randrange(10, 4000)))
                actions.append(action)
                units[action.action_id] = 0
        old = solve_current_flow(actions, units)
        new = solve_current_flow(actions, units, empty_distance_mode="PRIMARY_FACE_MM")
        if (old["score_units"] != new["score_units"] or not new["secondary_optimality_proved"]
                or new["current_empty_distance_m"] > old["current_empty_distance_m"] + 1e-6):
            raise AssertionError("synthetic global-distance/primary-preservation check failed")
        runs.append(dict(seed=20261011 + seed, actions=len(actions),
            old_runtime_s=old["runtime_s"], new_runtime_s=new["runtime_s"],
            secondary_time_s=new["secondary_time_s"], secondary_backend=new["secondary_backend"],
            primary_score=list(new["score_units"]), old_empty_m=old["current_empty_distance_m"],
            new_empty_m=new["current_empty_distance_m"], nodes=new["nodes"],
            residual_edges=new["residual_edges"], wall_deadline_exceeded=new["wall_deadline_exceeded"]))
    memory = psutil.Process().memory_info()
    result = dict(status="SYNTHETIC_KERNEL_CHECK_COMPLETE", samples=5, runs=runs,
        scope="SYNTHETIC_800_ACTION_CURRENT_GRAPHS_NOT_RECORDED_CITY_ARCS",
        old_warm_mean_s=mean(r["old_runtime_s"] for r in runs[1:]),
        new_warm_mean_s=mean(r["new_runtime_s"] for r in runs[1:]),
        maximum_new_runtime_s=max(r["new_runtime_s"] for r in runs),
        peak_rss_mib=max(memory.rss, getattr(memory, "peak_wset", memory.rss))/2**20,
        total_process_body_s=perf_counter() - started, routing_calls=0, inference_calls=0,
        native_replay=False, dense_matrix=False, GPU_used=False,
        operational_benefit_claimed=False, whole_day_runtime_estimated=False)
    atomic_json(Path(root) / "stage4/docs/capability_chain_planning/operations_v5/kernel_check.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    print(json.dumps(check(parser.parse_args().root), ensure_ascii=False))


if __name__ == "__main__":
    main()
