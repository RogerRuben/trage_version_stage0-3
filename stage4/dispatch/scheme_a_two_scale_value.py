"""300-second competitive continuation values over a sparse forecast DAG.

No Valhalla, Test31 future orders, realized times, or physical execution enters
this module. The future empty-time proxy is the EXISTING Train-M3 chord pace.
Customer capability masks are frozen inputs; future empty links are spatial /
temporal planning approximations, NOT directed road certificates.

Each forecast node has one fixed pickup slot on the 300-s grid (inside its
original 300-s patience), and a frozen M3 service duration. Top3 successor
arcs form a time-forward DAG. Shared customer capacities in a sparse LP price
competition; bounded pricing adds useful paths of at most three services.
Only the resulting state value is handed to the 30-s CURRENT action flow.
"""
from __future__ import annotations

from collections import Counter
from math import ceil, floor, isfinite
from time import perf_counter
import warnings

import numpy as np
from scipy.optimize import linprog, OptimizeWarning
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

from .flexibility_native import TrainDemandForecast
from .scheme_a_city_adapter import sites_for
from .scheme_a_city_graph import ChainTask, RouteState, _xy


class _ValueBudgetExpired(TimeoutError):
    pass


class _SceneDAG:
    """O(N*TopK) storage; never an N-by-N distance or reachability table."""
    def __init__(self, tasks, sites, now, horizon, pace, cfg, check):
        self.tasks = tuple(tasks)
        for task in self.tasks:
            if (not isinstance(task, ChainTask) or task.source_kind != "FORECAST"
                    or task.request_id >= 0 or not task.source_date or task.source_date > "20161024"):
                raise ValueError("coarse values accept earlier-history prediction tasks only")
        if len({t.request_id for t in self.tasks}) != len(self.tasks):
            raise ValueError("duplicate synthetic task in one coarse scenario")
        self.now, self.horizon, self.pace, self.cfg = now, horizon, pace, cfg
        self.site_points = np.asarray([_xy(site["position"]) for site in sites],float).reshape(-1,2)
        self.pickup = np.asarray([_xy(t.pickup) for t in self.tasks],float).reshape(-1,2)
        self.dropoff = np.asarray([_xy(t.dropoff) for t in self.tasks],float).reshape(-1,2)
        self.release = np.asarray([t.release_s for t in self.tasks],float)
        self.ids = np.asarray([t.request_id for t in self.tasks],np.int64)
        self.pickup_slot = np.ceil(self.release/300)*300
        self.finish = self.pickup_slot + np.asarray([t.service_time_s for t in self.tasks],float)
        self.deadline = np.asarray([t.deadline_s for t in self.tasks],float)
        if np.any(self.pickup_slot > self.deadline + 1e-9):
            raise ValueError("coarse pickup grid exceeded original forecast patience")
        n = len(self.tasks)
        self.allowed = {
            "HV": np.asarray(["HV" in t.compatible_profiles for t in self.tasks],bool),
            "C": np.asarray(["C" in t.compatible_profiles and t.passenger_accepts_av for t in self.tasks],bool),
        }
        self.tree = cKDTree(self.pickup) if n else None
        self.site_neighbors = []
        for point in self.site_points:
            indices = np.asarray(self.tree.query_ball_point(point,2000.) if self.tree else (),np.int32)
            distance = np.linalg.norm(self.pickup[indices]-point,axis=1)
            self.site_neighbors.append((indices,distance))
        self.successors, self.departures = {}, {}
        self.edge_count = 0
        for profile in ("HV","C"):
            successors = np.full((n,3),-1,np.int32)
            departures = np.full((n,3),np.inf)
            for i in np.flatnonzero(self.allowed[profile]):
                if i % 32 == 0:
                    check()
                indices = np.asarray(self.tree.query_ball_point(self.dropoff[i],2000.),np.int32)
                distance = np.linalg.norm(self.pickup[indices]-self.dropoff[i],axis=1)
                depart = self._first_feasible_departure(
                    np.maximum(self.finish[i],self.release[indices]),distance,
                    self.pickup_slot[indices],horizon)
                keep = (self.allowed[profile][indices] & (indices != i)
                        & np.isfinite(depart))
                indices, distance, depart = indices[keep],distance[keep],depart[keep]
                if len(indices):
                    order = np.lexsort((self.ids[indices],self.release[indices],
                                       self.pickup_slot[indices],distance))[:3]
                    successors[i,:len(order)] = indices[order]
                    departures[i,:len(order)] = depart[order]
                    self.edge_count += len(order)
            self.successors[profile], self.departures[profile] = successors,departures
        self.alpha = np.zeros(n)
        self.beta = {}
        self.dp_cache, self.entry_cache = {}, {}
        self.info = {}

    def _paces(self, departure):
        result = np.full(len(departure),self.pace.global_pace,float)
        slots = (departure//1800).astype(int)
        for slot in np.unique(slots):
            result[slots == slot] = self.pace.pace_by_slot.get(int(slot),self.pace.global_pace)
        return result

    def _first_feasible_departure(self, earliest, distance, pickup_slot, end):
        """Allow waiting across pace slots; earlier readiness cannot lose a path.

        The existing pace proxy is constant in each 1800-s slot. Trying the
        earliest 30-s departure in each intersecting slot is sufficient to
        determine whether any departure in that slot reaches a fixed pickup.
        This uses vectors for one spatial neighbor list, never all pairs.
        """
        result = np.full(len(distance),np.inf)
        for slot in range(int(self.now//1800),int(ceil(self.horizon/1800))):
            depart = np.ceil(np.maximum(earliest,slot*1800)/30)*30
            pace = self.pace.pace_by_slot.get(slot,self.pace.global_pace)
            feasible = ((depart < min((slot+1)*1800,self.horizon,end))
                        & (depart+distance*pace <= pickup_slot+1e-9))
            result = np.minimum(result,np.where(feasible,depart,np.inf))
        return result

    def entries(self, key):
        cached = self.entry_cache.get(key)
        if cached is not None:
            return cached
        profile, zone, ready_bin, end_bin = key
        ready,end = self.now+ready_bin*300,self.now+end_bin*300
        indices,distance = self.site_neighbors[zone]
        depart = self._first_feasible_departure(np.maximum(ready,self.release[indices]),
            distance,self.pickup_slot[indices],end)
        keep = self.allowed[profile][indices] & np.isfinite(depart)
        indices,distance = indices[keep],distance[keep]
        # Keep every feasible head in the bounded 2-km neighborhood. A Top3
        # truncation AFTER readiness filtering can make a later-ready state
        # spuriously MORE valuable, invalidating current-MOVE value bounds.
        # This is a single sparse neighbor list, not a state-by-order matrix.
        order = np.lexsort((self.ids[indices],self.release[indices],
                           self.pickup_slot[indices],distance))
        result = indices[order]
        self.entry_cache[key] = result
        return result

    def _dp(self, profile, end_bin):
        key = profile,end_bin
        cached = self.dp_cache.get(key)
        if cached is not None:
            return cached
        n = len(self.tasks)
        values = np.full((4,n),-np.inf)
        values[0] = 0.
        following = np.full((4,n),-1,np.int32)
        reward = 1.-self.alpha
        eligible = self.allowed[profile]
        successors = self.successors[profile]
        valid = ((successors >= 0)
                 & (self.departures[profile] < self.now+end_bin*300))
        safe = np.maximum(successors,0)
        for depth in range(1,4):
            if n:
                candidate = np.where(valid,values[depth-1,safe],-np.inf)
                best_column = np.argmax(candidate,axis=1)
                best = candidate[np.arange(n),best_column]
                use = best > 0
                values[depth,eligible] = reward[eligible] + np.maximum(0,best[eligible])
                following[depth,use] = successors[np.arange(n)[use],best_column[use]]
        result = values,following
        self.dp_cache[key] = result
        return result

    def price(self, key):
        profile,_,_,end_bin = key
        entries = self.entries(key)
        if not len(entries):
            return 0.,()
        values,following = self._dp(profile,end_bin)
        # Optional idle/termination is allowed at every depth. The fixed-time
        # DAG prevents a path visiting the same synthetic customer twice.
        offered = values[1:4,entries]
        score = float(np.max(offered))
        ties = np.argwhere(offered == score)
        depth,column = min(ties,key=lambda item:(int(item[0]),int(self.ids[entries[int(item[1])]])))
        depth,node = int(depth)+1,int(entries[int(column)])
        if score <= 1e-9:
            return 0.,()
        path = []
        while node >= 0 and depth > 0:
            path.append(node)
            node = int(following[depth,node])
            depth -= 1
        return min(3.,max(0.,score)),tuple(path)

    def set_prices(self, alpha, beta):
        self.alpha, self.beta = alpha,beta
        self.dp_cache.clear()

    def solve_competition(self, supply, check, remaining, cfg):
        keys = sorted(supply)
        self.info = dict(status="EMPTY_FORECAST", available=True, columns=0,nonzeros=0,
                         pricing_rounds=0,pricing_closed=True,lp_time_s=0.,pricing_time_s=0.)
        if not len(self.tasks) or not keys:
            return
        columns, seen = [],set()

        def add(state_index,path):
            signature = state_index,path
            if path and signature not in seen:
                seen.add(signature)
                columns.append(signature)

        for i,key in enumerate(keys):
            if i % 32 == 0:
                check()
            _,path = self.price(key)
            add(i,path)
        if not columns:
            self.info["status"] = "NO_CONNECTED_FORECAST_PATH"
            return
        last_alpha,last_beta = None,None
        closed=False
        for iteration in range(int(cfg.get("coarse_value_max_pricing_rounds",6))):
            check()
            if len(columns) > cfg.get("coarse_value_max_columns",20_000):
                self.info["status"] = "APPROXIMATE_COLUMN_LIMIT"
                break
            row,col,data = [],[],[]
            for j,(state_index,path) in enumerate(columns):
                for r in (state_index,*(len(keys)+node for node in path)):
                    row.append(r); col.append(j); data.append(1.)
            if len(data) > cfg.get("coarse_value_max_nonzeros",150_000):
                self.info["status"] = "APPROXIMATE_NONZERO_LIMIT"
                break
            matrix = coo_matrix((data,(row,col)),shape=(len(keys)+len(self.tasks),len(columns))).tocsr()
            bound = np.r_[[supply[key] for key in keys],np.ones(len(self.tasks))]
            costs = -np.asarray([len(path) for _,path in columns],float)
            lp_started=perf_counter()
            # A single CPU worker is passed to the installed HiGHS backend;
            # SciPy reports pass-through options, which are explicitly checked.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always",OptimizeWarning)
                solved=linprog(costs,A_ub=matrix,b_ub=bound,bounds=(0,None),method="highs-ds",
                    options={"time_limit":max(.001,remaining()),"presolve":True,"threads":1})
            for warning in caught:
                if "Unrecognized options detected" not in str(warning.message):
                    raise RuntimeError(f"coarse LP warning: {warning.message}")
            self.info["lp_time_s"] += perf_counter()-lp_started
            self.info.update(pricing_rounds=iteration+1,columns=len(columns),nonzeros=len(data))
            if solved.status == 1:
                self.info["status"] = "APPROXIMATE_COARSE_LP_TIME_LIMIT"
                break
            if solved.status != 0 or not solved.success:
                raise RuntimeError(f"coarse WAIT-feasible LP failed: {solved.message}")
            margins=np.asarray(solved.ineqlin.marginals,float)
            if not np.all(np.isfinite(margins)):
                raise RuntimeError("coarse LP returned nonfinite scarcity prices")
            last_beta={key:max(0.,float(-margins[i])) for i,key in enumerate(keys)}
            last_alpha=np.maximum(0.,-margins[len(keys):])
            self.set_prices(last_alpha,last_beta)
            self.info.update(status="APPROXIMATE_RESTRICTED_COARSE_PRICES",available=True,
                             lp_service_value=float(-solved.fun),alpha_nonzero_count=int((last_alpha>1e-7).sum()))
            pricing_started=perf_counter()
            added=0
            for i,key in enumerate(keys):
                if i % 32 == 0:
                    check()
                value,path=self.price(key)
                if value > last_beta[key]+1e-7 and (i,path) not in seen:
                    add(i,path); added+=1
            self.info["pricing_time_s"] += perf_counter()-pricing_started
            if not added:
                closed=True
                break
        self.info["available"] = last_alpha is not None
        self.info["pricing_closed"] = closed
        if last_alpha is not None:
            self.set_prices(last_alpha,last_beta)


class CoarseServiceValue:
    """One shared sparse future model per 300-s bucket, not per live action."""
    def __init__(self, library, reference, templates, cfg):
        self.library,self.reference,self.cfg=library,reference,dict(cfg)
        if cfg.get("coarse_reference_update_s",300) != 300 or cfg.get("planning_horizon_s",1800) != 1800:
            raise ValueError("two-scale value uses the selected 300s/1800s layers")
        if cfg.get("coarse_empty_eta_model","EXISTING_TRAIN_M3_CHORD_PACE") != "EXISTING_TRAIN_M3_CHORD_PACE":
            raise ValueError("no new future empty-time calibration is authorized")
        pace_cfg=dict(cfg,forecast_horizon_s=1800,fast_forecast=True)
        self.pace=TrainDemandForecast(templates,pace_cfg,cfg.get("coarse_reference_day_end_s",86434))
        self.coarse_s=None
        self.sites=[]
        self.tree=None
        self.models={}
        self.lookup_cache={}
        self.info=dict(available=False,status="NOT_REFRESHED",refreshed=False)
        self.counts=Counter()
        self.timings=Counter()
        self.refresh_records=[]

    def state_key(self,state):
        if state.profile_id not in ("HV","C"):
            raise ValueError("outside fixed C/HV coarse comparison")
        ready=max(float(state.ready_s),float(self.coarse_s))
        end=min(float(state.admission_end_s),self.coarse_s+1800)
        ready_bin=int(ceil((ready-self.coarse_s)/300))
        end_bin=int(floor((end-self.coarse_s)/300+1e-10))
        if ready_bin >= end_bin or ready_bin >= 6:
            return None
        distance,zone=self.tree.query(np.asarray(_xy(state.position)))
        self.counts["projection_queries"]+=1
        self.counts["projection_over_2km"]+=int(distance>2000.)
        return state.profile_id,int(zone),ready_bin,end_bin

    def refresh(self,now,supply_states):
        coarse=int(now//300)*300
        if self.coarse_s == coarse:
            return dict(self.info,refreshed=False)
        started=perf_counter()
        budget=float(self.cfg.get("coarse_value_wall_limit_s",5.))
        if not 0 < budget <= 5:
            raise ValueError("coarse value construction retains the bounded 5-second budget")
        deadline=started+budget
        def check():
            if perf_counter() >= deadline:
                raise _ValueBudgetExpired("coarse value wall budget exhausted")
        self.coarse_s=coarse
        self.sites=sites_for(self.reference,coarse,self.cfg)
        self.tree=cKDTree(np.asarray([_xy(s["position"]) for s in self.sites]))
        self.models={}
        self.lookup_cache.clear()
        supply=Counter()
        for state in supply_states:
            if not isinstance(state,RouteState):
                raise ValueError("coarse supply must be prediction-only RouteState objects")
            key=self.state_key(state)
            if key is not None:
                supply[key]+=1
        self.info=dict(available=False,status="COARSE_PRICES_UNAVAILABLE_SERVICE_FIRST",refreshed=True,
            snapshot_s=coarse,source_resource_count=sum(supply.values()),source_supply_states=len(supply),
            scenario_count=len(self.library.weights),forecast_route_queries=0,
            future_directed_road_certificate=False,projection_is_physical_position=False)
        scenario_info=[]
        try:
            for name,tasks in self.library.view(coarse).items():
                check()
                dag_started=perf_counter()
                model=_SceneDAG(tasks,self.sites,coarse,coarse+1800,self.pace,self.cfg,check)
                self.timings["dag_build_time_s"]+=perf_counter()-dag_started
                self.counts["forecast_nodes_built"]+=len(model.tasks)
                self.counts["forecast_dag_arcs_built"]+=model.edge_count
                model.solve_competition(supply,check,lambda:deadline-perf_counter(),self.cfg)
                self.models[name]=model
                scenario_info.append(dict(model.info,forecast_nodes=len(model.tasks),dag_arcs=model.edge_count))
                self.timings["lp_time_s"]+=model.info.get("lp_time_s",0.)
                self.timings["pricing_time_s"]+=model.info.get("pricing_time_s",0.)
            available=(len(self.models)==len(self.library.weights)
                       and all(m.info["available"] for m in self.models.values()))
            self.info.update(available=available,
                status="COARSE_COMPETITIVE_PRICES_READY" if available else "COARSE_PRICES_UNAVAILABLE_SERVICE_FIRST")
        except _ValueBudgetExpired:
            self.info.update(available=False,status="COARSE_TIME_LIMIT_SERVICE_FIRST")
            self.counts["coarse_wall_budget_exhausted"]+=1
        elapsed=perf_counter()-started
        self.info.update(wall_time_s=elapsed,scenario_info=scenario_info,
                         all_pricing_closed=(len(scenario_info)==len(self.library.weights)
                             and bool(scenario_info) and all(i["pricing_closed"] for i in scenario_info)))
        self.counts["refreshes"]+=1
        self.counts["unavailable_refreshes"]+=int(not self.info["available"])
        self.timings["refresh_wall_time_s"]+=elapsed
        self.refresh_records.append(dict(self.info))
        return dict(self.info)

    def value(self,state):
        started=perf_counter()
        try:
            if not self.info["available"]:
                return 0.
            key=self.state_key(state)
            if key is None:
                return 0.
            prior=self.lookup_cache.get(key)
            if prior is not None:
                self.counts["value_cache_hits"]+=1
                return prior
            # Zero-supply states are priced by the SAME DAG/alpha oracle, not
            # silently filled with a solver's absent/degenerate beta value.
            value=sum(self.library.weights[name]*model.price(key)[0] for name,model in self.models.items())
            value=min(3.,max(0.,float(value)))
            self.lookup_cache[key]=value
            self.counts["priced_state_queries"]+=1
            return value
        finally:
            self.timings["value_lookup_wall_time_s"]+=perf_counter()-started

    def diagnostics(self):
        arrays=sum(array.nbytes for model in self.models.values()
                   for array in (model.pickup,model.dropoff,model.release,model.ids,model.pickup_slot,
                                 model.finish,model.deadline,model.alpha,
                                 *model.allowed.values(),*model.successors.values(),*model.departures.values()))
        arrays+=sum(array.nbytes for model in self.models.values()
                    for pair in model.dp_cache.values() for array in pair)
        return dict(counts=dict(self.counts),timings_s=dict(self.timings),latest=dict(self.info),
            resident_array_mib=arrays/2**20,lookup_cache_entries=len(self.lookup_cache),
            forecast_empty_eta_model="EXISTING_TRAIN_M3_CHORD_PACE",
            global_pace_s_per_m=float(self.pace.global_pace),forecast_route_queries=0,
            future_directed_road_certificate=False,actual_route_validation_unchanged=True,
            realized_targets_used=False,full_city_upper_bound_claimed=False,
            shared_future_customer_capacity=True,scenario_refresh_s=300,forecast_horizon_s=1800,
            fixed_future_pickup_grid_s=300,maximum_future_services=3,future_move_variables=0,
            raw_source_orders_or_coordinates_exported=False)
