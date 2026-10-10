"""Finite Scheme-A joint layout / executable-action / service-chain planner.

This is an independent three-resource prototype. It is NOT the city next-job
adapter, an independent macro flow optimizer, or a native/full-day experiment.
Future paths use supplied directed profile-compatible connections, not chord
distance promises. Only the shared first action is executable.
"""
from __future__ import annotations

from copy import deepcopy

from stage4.analysis.capability_chain_rolling import (
    epoch_problem, first_actions,
)
from stage4.dispatch.scheme_a_joint_solver import solve_joint_epoch


LAYOUT_MODES = ("HOTSPOT_FIXED", "CHAIN_JOINT")


class CoarseHistoricalScenarios:
    """Refresh earlier-history input at 300 s; preserve exact task timestamps.

The coarse layer is a scenario/connection reference, not a separate claimed
city-scale macro optimization. Fine actions are reoptimized every 30 s. The
30-min prediction horizon is clipped to the declared finite episode end.
"""
    def __init__(self, cohorts, cfg):
        self.cohorts = deepcopy(cohorts)
        self.cfg = cfg
        self.last_refresh = None
        self.snapshot = ()
        self.refresh_count = 0
        for scene in self.cohorts:
            if any(t.get("source_kind") != "FORECAST" for t in scene["tasks"]):
                raise ValueError("coarse scenarios may contain historical FORECAST tasks only")

    def view(self, now):
        refresh = now // self.cfg["coarse_reference_update_s"] * self.cfg["coarse_reference_update_s"]
        if self.last_refresh != refresh:
            end = min(refresh + self.cfg["planning_horizon_s"], self.cfg["horizon_s"])
            self.snapshot = [dict(scenario_id=s["scenario_id"], weight=s["weight"],
                tasks=[deepcopy(t) for t in s["tasks"] if refresh < t["release_s"] < end])
                for s in self.cohorts]
            self.last_refresh = refresh
            self.refresh_count += 1
        end = min(now + self.cfg["planning_horizon_s"], self.cfg["horizon_s"])
        return [dict(scenario_id=s["scenario_id"], weight=s["weight"],
            tasks=[deepcopy(t) for t in s["tasks"] if now < t["release_s"] < end])
            for s in self.snapshot]


class SchemeAJointPlanner:
    """Share a finite typed-chain model between layout and rolling actions."""
    def __init__(self, provider, cohorts, cfg, *, policy="CHAIN_DEFER"):
        self.cfg = dict(cfg)
        self.provider = provider
        self.policy = policy
        if policy not in ("CHAIN_DEFER", "SERVICE_PRESERVING"):
            raise ValueError("unknown joint-chain control")
        fixed = dict(planning_horizon_s=1800, step_s=30, patience_s=300,
            coarse_reference_update_s=300, solver_decision_limit_s=10.0)
        if any(self.cfg.get(k) != v for k, v in fixed.items()):
            raise ValueError("outside the declared 30-min finite Scheme-A protocol")
        if self.cfg.get("skip_planned_post_service_geometry_queries", False):
            raise ValueError("joint-chain planning requires supplied typed post-service connections")
        self.reference = CoarseHistoricalScenarios(cohorts, self.cfg)

    def _solve(self, problem, *, include_bound):
        solution = solve_joint_epoch(problem, self.policy, include_bound=include_bound)
        solution["scheme_a_scope"] = dict(
            kind="FINITE_JOINT_LAYOUT_AND_MULTI_SERVICE_CHAIN_CURRENT_ACTION_PROTOTYPE",
            maximum_planning_horizon_s=self.cfg["planning_horizon_s"],
            execution_step_s=self.cfg["step_s"],
            coarse_history_refresh_s=self.cfg["coarse_reference_update_s"],
            refresh_is_input_reference_not_separate_macro_optimizer=True,
            scenario_paths_are_value_approximations_not_future_execution=True,
            only_shared_first_action_executable=True,
            customer_and_connector_profiles_both_enforced=True,
            future_customer_to_customer_connections="SUPPLIED_DIRECTED_SUPPORTED_SNAPSHOT",
            current_relocation_is_in_same_master_as_SERVE_WAIT_BUSY=True,
            future_independent_relocations_between_customers_in_recourse=False,
            native_execution=False, city_scale=False, formal_q10=False)
        return solution

    def layout_problem(self, resource_templates, sites, hotspot_ranking, *, mode):
        if mode not in LAYOUT_MODES:
            raise ValueError("unknown initial-layout mode")
        site_ids = {s["site_id"] for s in sites}
        if not site_ids or set(hotspot_ranking) != site_ids:
            raise ValueError("hotspot ranking must cover the identical common candidate set")
        resources = [{k: r[k] for k in ("resource_id", "profile_id", "admission_end_s")}
            for r in resource_templates]
        actions = []
        for template, resource in zip(resource_templates, resources):
            if resource["profile_id"] == "HV":
                # AV layout choices must not also rearrange the HV baseline.
                choices = [template["location_id"]]
            elif resource["profile_id"] == "C":
                choices = [hotspot_ranking[0]] if mode == "HOTSPOT_FIXED" else sorted(site_ids)
            else:
                raise ValueError("this finite batch uses the declared HV/HV/C resources")
            if not set(choices) <= site_ids:
                raise ValueError("initial resource position outside the common candidate pool")
            for site in choices:
                actions.append(dict(resource_id=resource["resource_id"],
                    action_id=resource["resource_id"]+":LAYOUT:"+site, kind="LAYOUT",
                    ready_s=0.0, location_id=site, job_id=None,
                    empty_distance_m=0.0, critical=False, carry_over=False))
        # Actual target-day requests are intentionally not accepted here.
        return epoch_problem(resources, [], self.reference.view(0), actions,
            self.provider, 0, self.cfg)

    def plan_initial_layout(self, resource_templates, sites, hotspot_ranking, *, mode):
        problem = self.layout_problem(resource_templates, sites, hotspot_ranking, mode=mode)
        solution = self._solve(problem, include_bound=True)
        selected = {a["resource_id"]: a["location_id"] for a in solution["selected_actions"]}
        initial = [dict(r, location_id=selected[r["resource_id"]], ready_s=0.0,
            last_relocation_s=-self.cfg["reposition_interval_s"], relocation_count=0)
            for r in resource_templates]
        solution["layout_mode"] = mode
        return initial, solution, problem

    def plan_epoch(self, resources, pending, now, *, include_bound=False):
        if now < 0 or now >= self.cfg["horizon_s"] or now % self.cfg["step_s"]:
            raise ValueError("current epoch outside declared finite execution window")
        for task in pending:
            if (task.get("source_kind") != "ACTUAL_PENDING"
                or task["release_s"] > now or task["deadline_s"] < now
                or task["deadline_s"] != task["release_s"] + self.cfg["patience_s"]):
                raise ValueError("pending tasks must be revealed and retain original 300s deadlines")
        actions = first_actions(resources, pending, self.provider, now, self.cfg)
        problem = epoch_problem(resources, pending, self.reference.view(now), actions,
            self.provider, now, self.cfg)
        solution = self._solve(problem, include_bound=include_bound)
        return solution, problem
