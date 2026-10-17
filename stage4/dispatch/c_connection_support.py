"""Opt-in shared C pickup support with bounded prefetch, never road I/O in search.

Unexamined links contribute no certified future service; they are not labelled
physically impossible. Legacy mode leaves HV proxies unchanged; economic mode
shares the small prefetch budget with cached HV road support. Neither changes
passenger bodies or capability rules. The old proxy planner remains default.
"""
from collections import Counter
from math import ceil, isfinite
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

from .scheme_a_city_graph import RouteState, _xy


MODE = "BOUNDED_CERTIFIED_C_V1"


def context_key(context):
    if context is None:
        return ("INITIAL",)
    kind = context.get("kind")
    if kind == "CUSTOMER":
        task = context["task"]
        return kind, task.source_date, task.source_order_id, task.pickup, task.dropoff
    if kind == "NATIVE_VEHICLE":
        return kind, context["native_vehicle_id"], context["timestamp_s"]
    if kind == "EMPTY_ARRIVAL":
        uids = context.get("edge_uids")
        if uids is None:
            uids = tuple(edge["stage3_edge_uid"] for edge in context.get("edges", ()))
        return kind, tuple(context["point"]), tuple(uids), context_key(context.get("previous_context"))
    return (str(kind),)


class ConnectionBudget:
    """Extra backend REQUEST budget shared by current recovery and future prefetch.

    The already-existing default current pickup query is not an extra query.
    One atomic Actor call cannot be preempted; no subsequent optional call is
    started after its budget/deadline. No scientific threshold is relaxed.
    """
    def __init__(self, *, maximum_per_epoch=2, maximum_per_snapshot=32, wall_s=.15):
        if (type(maximum_per_epoch) is not int or not 0 <= maximum_per_epoch <= 2
                or type(maximum_per_snapshot) is not int or not 0 <= maximum_per_snapshot <= 32
                or not isfinite(wall_s) or not 0 <= wall_s <= .15):
            raise ValueError("C support cannot exceed the declared small optional budget")
        self.maximum_per_epoch, self.maximum_per_snapshot, self.wall_s = (
            maximum_per_epoch, maximum_per_snapshot, float(wall_s))
        self.epoch = self.snapshot = None
        self.epoch_calls = self.snapshot_calls = 0
        self.spent_epoch_s = 0.
        self.deadline = -1.
        self.counts = Counter()

    def begin(self, now, deadline):
        if not isfinite(float(now)) or not isfinite(float(deadline)):
            raise ValueError("C support needs finite epoch/deadline")
        if self.epoch is not None and now < self.epoch:
            raise ValueError("C connection budget cannot move backwards")
        if now != self.epoch:
            self.epoch, self.epoch_calls, self.spent_epoch_s = float(now), 0, 0.
            self.deadline = float(deadline)
        else:
            self.deadline = min(self.deadline, float(deadline))
        snapshot = int(now // 300)
        if snapshot != self.snapshot:
            self.snapshot, self.snapshot_calls = snapshot, 0

    def claim(self, purpose):
        if (self.epoch_calls >= self.maximum_per_epoch
                or self.snapshot_calls >= self.maximum_per_snapshot
                or self.spent_epoch_s >= self.wall_s or perf_counter() >= self.deadline):
            self.counts["optional_requests_deferred"] += 1
            return False
        self.epoch_calls += 1
        self.snapshot_calls += 1
        self.counts["optional_query_slots_consumed"] += 1
        self.counts[str(purpose) + "_budget_slots"] += 1
        return True

    def record(self, elapsed):
        if not isfinite(elapsed) or elapsed < 0:
            raise ValueError("optional support time must be finite/nonnegative")
        self.spent_epoch_s += float(elapsed)
        self.counts["optional_work_time_s"] += float(elapsed)


class SharedCConnectionSupport:
    def __init__(self, connectors, task_factory, cfg):
        if cfg.get("joint_c_connection_mode") != MODE:
            raise ValueError("shared C support is explicitly opt-in")
        self.connectors, self.task_factory = connectors, task_factory
        self.all_profiles=cfg.get("economic_future_connections")=="BOTH_CACHE_CERTIFIED_V1"
        self.connectors.bounded_all_profile_future=self.all_profiles
        self.budget = ConnectionBudget(
            maximum_per_epoch=cfg.get("c_extra_backend_requests_per_epoch", 2),
            maximum_per_snapshot=cfg.get("c_extra_backend_requests_per_snapshot", 32),
            wall_s=cfg.get("c_optional_connection_wall_s", .15))
        self.connectors.c_connection_budget = self.budget
        self.counts = Counter()
        self.sparse_index=None; self.visibility={}
        if cfg.get("economic_sparse_connections") is not None:
            from .sparse_road_support import MODE as SPARSE_MODE,SparseRoadIndex
            if cfg["economic_sparse_connections"]!=SPARSE_MODE or not self.all_profiles:
                raise ValueError("shared sparse connections require the economic two-type road mode")
            self.sparse_index=SparseRoadIndex(context_key,cfg["sparse_connection_namespace"],
                path=cfg.get("sparse_connection_store"))

    def begin(self, now, deadline):
        if self.budget.epoch!=now: self.visibility.clear()
        self.budget.begin(now, deadline)
        if self.sparse_index is not None: self.sparse_index.begin(now)

    def current(self, control, runtime, request, now):
        if perf_counter() >= self.budget.deadline:
            self.counts["current_connections_deferred_at_deadline"] += 1
            return dict(supported=False,reason_codes=["C_CONNECTION_NOT_EVALUATED_DECISION_DEADLINE"])
        origin = tuple(control.routing_engine.return_position_coordinates(runtime.native_vehicle.pos))
        state = RouteState(origin, float(now), float(control.fixture_windows_s[runtime.fixture.native_id][1]),
            "C", dict(kind="NATIVE_VEHICLE", native_vehicle_id=int(runtime.fixture.native_id), timestamp_s=float(now)))
        task = self.task_factory(control, int(request.native_id))
        result = self.connectors(state, task, now, "C", require_geometry=True,
                                 enforce_pickup_deadline=False, query_scope="CURRENT")
        self.counts["current_connections_examined"] += 1
        self.counts["current_connections_supported"] += bool(result["supported"])
        if self.sparse_index is not None: self.sparse_index.remember(state,task,result)
        return result

    def _visible(self,state,task):
        validate=getattr(self.connectors,"_task_metadata",None)
        if validate is None: return None  # Small injected test connectors have no source store.
        check=[task]
        if (state.context or {}).get("kind")=="CUSTOMER": check.append(state.context["task"])
        for t in check:
            key=t.source_date,t.source_order_id,t.source_kind
            if key not in self.visibility: self.visibility[key]=validate(t)[1]
            if self.visibility[key]: return self.visibility[key]
        return None

    def _beta(self,departure):
        control=self.connectors.control
        return control.eta_adapter.beta_for(control._timestamp(departure))[1]

    def prepare_edge(self,state,task,departure):
        """One shared candidate, at most the existing optional query allowance."""
        reason=self._visible(state,task)
        if reason: return dict(supported=False,reason_codes=[reason])
        index=self.sparse_index
        if index.assessed(state,task):
            result=index.lookup(state,task,self._beta(departure))
        else:
            result=self.connectors(state,task,departure,state.profile_id,query_scope="FUTURE",
                enforce_pickup_deadline=False)
            index.remember(state,task,result)
        self.counts["prefetch_connections_examined"]+=1
        self.counts[state.profile_id+"_prefetch_connections_examined"]+=1
        return result

    def lookup(self, state, task, departure):
        """Read-only certificate lookup. It must NEVER generate a road query."""
        profile=state.profile_id if self.all_profiles else "C"
        if self.sparse_index is not None:
            reason=self._visible(state,task)
            result=(dict(supported=False,reason_codes=[reason]) if reason else
                self.sparse_index.lookup(state,task,self._beta(departure)))
        else:
            result = self.connectors(state, task, departure, profile, cache_only=True,
                                     enforce_pickup_deadline=False, query_scope="FUTURE")
        self.counts["planning_cache_lookups"] += 1
        self.counts["planning_cache_supported"] += bool(result["supported"])
        if not result["supported"] and any("NOT_EVALUATED" in reason for reason in result["reason_codes"]):
            self.counts["planning_not_evaluated_not_incapable"] += 1
        return result

    def prefill(self, actions, roots, scenes, pending, selected, memory, now, check):
        """Small warm/nearby candidate list, at most 64 lookups and two new calls.

        Only historical scenarios and already-revealed pending tasks are supplied.
        Source and target identities retain their incoming/outgoing direction.
        The search starts AFTER this prefetch: cache misses stay constant during
        that search instead of turning millions of labels into router calls.
        """
        if self.budget.epoch != float(now):
            raise ValueError("C prefetch must share the current decision budget")
        if self.all_profiles:
            return self._prefill_both(actions,roots,scenes,pending,selected,memory,now,check)
        selected = set(selected)
        vehicles = [(a.vehicle_id, a.action_id) for a in actions
                    if a.action_id in selected and roots[a.action_id].profile_id == "C"]
        vehicles.sort()
        if not vehicles:
            return ()
        offset = int(now // 30) % len(vehicles)
        vehicles = vehicles[offset:] + vehicles[:offset]
        catalogs = {}
        for sid, forecast in sorted(scenes.items()):
            tasks = {t.request_id: t for t in (*pending, *forecast)
                     if "C" in t.compatible_profiles and t.passenger_accepts_av}
            ordered = tuple(tasks.values())
            catalogs[sid] = (tasks, ordered, cKDTree(np.asarray([_xy(t.pickup) for t in ordered])) if ordered else None)
        offers = []
        for vid, aid in vehicles:
            state = roots[aid]
            for sid, (tasks, ordered, tree) in catalogs.items():
                old = getattr(memory, "paths", {}).get((sid, vid), ()) if memory is not None else ()
                previous = state
                for rid in old[:3]:
                    task = tasks.get(rid)
                    if task is None:
                        break
                    offers.append((sid, previous, task))
                    previous = RouteState(task.dropoff, task.release_s + task.service_time_s,
                        state.admission_end_s, "C", dict(kind="CUSTOMER", task=task))
                if tree is not None:
                    nearby = tree.query_ball_point(_xy(state.position), 2000.)
                    ranked = sorted((float(np.linalg.norm(tree.data[i] - _xy(state.position))),
                                     ordered[i].release_s, ordered[i].request_id, ordered[i]) for i in nearby)
                    offers.extend((sid, state, item[-1]) for item in ranked[:3])
        seen, newly_supported = set(), set()
        for sid, state, task in offers:
            if len(seen) >= 64:
                break
            key = state.position, context_key(state.context), task.source_date, task.source_order_id
            if key in seen:
                continue
            seen.add(key)
            check()
            dispatch = (ceil(max(state.ready_s, now + 30, task.release_s) / 30) * 30
                        if task.source_kind == "ACTUAL_PENDING" else ceil(task.release_s / 300) * 300)
            if not max(state.ready_s, task.release_s) <= dispatch < min(task.deadline_s, state.admission_end_s, now + 1800):
                continue
            before = self.connectors.counts["certificate_cache_misses"]
            result = self.connectors(state, task, dispatch, "C", query_scope="FUTURE", enforce_pickup_deadline=False)
            if (result["supported"] and 0 <= float(result["travel_time_s"]) <= 600.
                    and self.connectors.counts["certificate_cache_misses"] > before):
                newly_supported.add((sid, task.request_id))
            self.counts["prefetch_connections_examined"] += 1
        return tuple(sorted(newly_supported))

    def _prefill_both(self,actions,roots,scenes,pending,selected,memory,now,check):
        """Same cache-only search evidence for BOTH types, one shared I/O budget.

        Alternate the type examined first across epochs. This is bounded
        exploration, not a completeness claim or a reward for using AVs.
        """
        selected=set(selected); by_type={p:[] for p in ("C","HV")}
        action_map={a.action_id:a for a in actions}
        for a in actions:
            if a.action_id in selected: by_type[roots[a.action_id].profile_id].append(a.action_id)
        for profile,ids in by_type.items():
            ids.sort(); offset=int(now//30)%max(1,len(ids))
            by_type[profile]=ids[offset:]+ids[:offset]
        catalog={}
        for sid,forecast in sorted(scenes.items()):
            tasks={t.request_id:t for t in (*pending,*forecast)}
            rows=tuple(tasks.values())
            tree=cKDTree(np.asarray([_xy(t.pickup) for t in rows])) if rows else None
            catalog[sid]=tasks,rows,tree
        order=("C","HV") if int(now//30)%2==0 else ("HV","C")
        offers=[]
        for depth in range(4):
            for profile in order:
                if depth>=len(by_type[profile]): continue
                aid=by_type[profile][depth]; root=roots[aid]; vid=action_map[aid].vehicle_id
                for sid,(tasks,rows,tree) in catalog.items():
                    previous=root
                    old=getattr(memory,"paths",{}).get((sid,vid),()) if memory is not None else ()
                    for rid in old[:3]:
                        task=tasks.get(rid)
                        if task is None: break
                        offers.append((sid,previous,task))
                        previous=RouteState(task.dropoff,task.release_s+task.service_time_s,
                            root.admission_end_s,profile,dict(kind="CUSTOMER",task=task))
                    point=np.asarray(_xy(root.position))
                    indices=tree.query_ball_point(point,2000.) if tree is not None else ()
                    choices=[rows[i] for i in indices if profile in rows[i].compatible_profiles
                        and rows[i].request_id!=action_map[aid].request_id
                        and (profile=="HV" or rows[i].passenger_accepts_av)]
                    choices.sort(key=lambda t:(float(np.linalg.norm(point-_xy(t.pickup))),
                        t.release_s,t.request_id))
                    offers.extend((sid,root,t) for t in choices[:3])
        seen=set(); discovered=set()
        for sid,state,task in offers:
            if len(seen)>=64: break
            key=state.profile_id,state.position,context_key(state.context),task.source_date,task.source_order_id,task.request_id
            if key in seen: continue
            seen.add(key); check()
            if state.profile_id not in task.compatible_profiles or (state.profile_id!="HV" and not task.passenger_accepts_av): continue
            dispatch=(ceil(max(state.ready_s,now+30,task.release_s)/30)*30 if task.source_kind=="ACTUAL_PENDING"
                else ceil(task.release_s/300)*300)
            if not max(state.ready_s,task.release_s)<=dispatch<min(task.deadline_s,state.admission_end_s,now+1800): continue
            before=self.connectors.counts["certificate_cache_misses"]
            link=self.connectors(state,task,dispatch,state.profile_id,query_scope="FUTURE",enforce_pickup_deadline=False)
            if link["supported"] and float(link["travel_time_s"])<=600.:
                if self.connectors.counts["certificate_cache_misses"]>before: discovered.add((sid,task.request_id))
            self.counts["prefetch_connections_examined"]+=1
            self.counts[state.profile_id+"_prefetch_connections_examined"]+=1
        return tuple(sorted(discovered))

    def diagnostics(self):
        return dict(mode=MODE, counts=dict(self.counts), budget=dict(self.budget.counts),
            all_profile_cache_evidence=self.all_profiles,
            extra_backend_requests_per_epoch=self.budget.maximum_per_epoch,
            extra_backend_requests_per_300s=self.budget.maximum_per_snapshot,
            optional_wall_budget_s=self.budget.wall_s,
            search_road_queries=0, unexamined_is_not_incapable=True,
            complete_C_reachability_claimed=False, objective_changed=False,
            sparse_road_index=self.sparse_index.diagnostics() if self.sparse_index is not None else None)

    def close(self):
        if self.sparse_index is not None: self.sparse_index.close()
