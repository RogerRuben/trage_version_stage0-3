"""Read-only witness/accounting checks; never solve, route or rerun experiments."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .competitive_chain_benchmark import CONFIG, OUTPUT, PUBLIC, load_configuration, load_inputs
from stage4.dispatch.competitive_chain import FiniteCatalogue, Root
from stage4.dispatch.flexibility_native import TrainDemandForecast
from stage4.dispatch.scheme_a_city_graph import ChainTask
from stage4.dispatch.scheme_a_match_deadline_value import MatchDeadlineSceneDAG


def verify(root):
    root = Path(root).resolve()
    summary = json.loads((root/PUBLIC/"summary.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(root/PUBLIC/"results.csv")
    if (summary["status"] != "TWELVE_INSTANCES_COMPLETED"
            or len(summary["cases"]) != 12 or len(frame) != 48
            or frame.duplicated(["case_id", "method"]).any()):
        raise ValueError("fixed instance/method reconciliation failed")
    cfg, current_bindings = load_configuration(root, CONFIG)
    templates, _, _ = load_inputs(root, cfg, current_bindings)
    if current_bindings != summary["inputs"]:
        raise ValueError("a historical input/configuration changed after the run")
    pace = TrainDemandForecast(templates, dict(cfg, fast_forecast=False), 86434)
    checked = 0
    certified_optima = 0
    max_gap = Fraction(0)
    for case in summary["cases"]:
        private = root/OUTPUT/case["case_id"]
        source = json.loads((private/"input.json").read_text(encoding="utf-8"))
        digest = hashlib.sha256(json.dumps(source, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if digest != case["input_sha256"]:
            raise ValueError("private declared input differs from its original hash")
        results = json.loads((private/"case_results.json").read_text(encoding="utf-8"))
        models = {name: MatchDeadlineSceneDAG([ChainTask(**task) for task in tasks],
            source["sites"], source["now"], source["now"]+1800, pace, cfg, lambda: None)
            for name, tasks in source["tasks"].items()}
        fixed = {tuple(item["key"]): item["count"] for item in source["fixed_supply"]}
        fresh = {tuple(item["group"]): item["count"] for item in source["fresh_supply"]}
        weights = {name: Fraction(1, len(models)) for name in models}
        catalogue = FiniteCatalogue(models, weights, fixed, fresh,
                                    [site["site_id"] for site in source["sites"]])
        index = {(root_key, path): column for column, (root_key, path) in enumerate(
            catalogue.columns, len(catalogue.placement))}
        common_upper = Fraction(*results["methods"]["EXACT_DECLARED_DOMAIN"]["bound"]["upper_bound_exact"])
        for method, result in results["methods"].items():
            observed = result["witness"]
            deployment = {tuple(map(int, group.split(":"))): np.asarray(values, np.int64)
                          for group, values in observed["deployment"].items()}
            vector = catalogue.idle(deployment)
            for item in observed["paths"]:
                root_key = Root(item["scene"], item["kind"], tuple(item["root"]))
                vector[index[root_key, tuple(item["tasks"])]] += 1
            _, scaled, exact, scenes = catalogue.validate(vector)
            if (exact != Fraction(*result["evaluation"]["service_exact"])
                    or exact != Fraction(*observed["service_exact"])
                    or scenes != result["evaluation"]["scenario_services"]):
                raise ValueError("integer service witness disagrees with saved evaluation")
            method_upper = Fraction(*result["bound"]["upper_bound_exact"])
            if exact > min(common_upper, method_upper):
                raise ValueError("saved exact bound is below a feasible integer witness")
            gap = common_upper-exact
            max_gap = max(max_gap, gap)
            # The primary expected count has a known rational objective lattice.
            if common_upper * catalogue.scale < scaled + 1:
                certified_optima += 1
            public = frame.loc[(frame.case_id == case["case_id"]) & (frame.method == method)].iloc[0]
            if abs(public.service_value-float(exact)) > 1e-10:
                raise ValueError("public result differs from private witness")
            checked += 1
    values = frame.pivot(index="case_id", columns="method", values="service_value")
    details = {
        "scope": "READ_ONLY_RECONSTRUCTION_NO_SOLVERS_NO_RERUN",
        "instances": 12, "witnesses_checked": checked, "public_rows": len(frame),
        "shared_customer_supply_and_integer_time_domain_checks": "ALL_RECONSTRUCTED",
        "saved_bound_inequalities_checked": True,
        "price_vectors_not_independently_replayed": True,
        "recorded_bound_objective_lattice_optimal_witnesses": certified_optima,
        "max_recorded_certificate_gap": float(max_gap),
        "existing_better_than_independent": int((values.EXISTING_RESTRICTED-values.INDEPENDENT_VALUE>1e-9).sum()),
        "recovery_better_than_existing": int((values.BOUND_RECOVERY-values.EXISTING_RESTRICTED>1e-9).sum()),
        "recovery_exchange_attempts": int(frame.exchange_attempts.sum()),
        "accepted_exchanges": int(frame.accepted_exchanges.sum()),
        "max_variables": int(frame.variables.max()), "max_nonzeros": int(frame.nonzeros.max()),
        "all_existing_pricing_closed": bool(frame.loc[frame.method.eq("EXISTING_RESTRICTED"), "pricing_closed"].eq(True).all()),
        "max_existing_pricing_rounds": int(frame.pricing_rounds.max()),
        "peak_rss_mib": summary["peak_rss_mib"], "processing_runtime_s": summary["processing_runtime_s"],
        "actual_case_runtime_sum_s": sum(case["runtime_s"] for case in summary["cases"]),
    }
    return details


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2, ensure_ascii=False))
