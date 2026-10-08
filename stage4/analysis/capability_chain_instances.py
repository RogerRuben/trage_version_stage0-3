"""Small Train-data service-chain instances; never a native/full-day replay.

Customer tasks use frozen predictions. Empty connectors are model-generated
directed auto routes, not observed deadheading. Constant window-start connector
times make the finite chain model inspectable; they are not retimed M3 forecasts.
Only compact anonymous summaries belong in Git. Instance coordinates and source
order identities remain under the ignored stage4/output directory.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import gc
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from stage3.odd_tod.intersection_complex import _bearing
from stage3.odd_tod.research_compatibility import evaluate_research_compatibility
from stage4.analysis.flexibility_prepare import MovementClassifier, S2A, S2B, S3
from stage4.analysis.symmetric_research_prepare import HistoricalResearchParser
from stage4.dispatch.controlled_routes import (
    CONTROL_POLICY, MOVEMENT_COLUMNS, TRAFFIC_OUTSIDE_BUDGET, UNKNOWN_TRAFFIC,
    ControlledEmptyRouter, _gap_m, _read_columns, _select,
)
from stage4.fleetpy_adapter.valhalla_time_adapter import ValhallaPickupTimeAdapter


CONFIG = Path("stage4/config/capability_chain_instances_v1.json")
OUT = Path("stage4/output/capability_chain_planning/instances_v1")
DOC = Path("stage4/docs/capability_chain_planning/instances_v1")
TEMPLATES = Path("stage4/output/symmetric_flexibility_v1/input/train_request_templates.parquet")
REFERENCE = Path("stage4/output/paper_enhancement/repositioning_robustness/train_tod_demand_reference.parquet")
IDENTITY_COLUMNS = ["date", "order_id", "route_sequence", "canonical_edge_uid",
    "route_token_type", "resolved_stage3_edge_uid"]
TOPOLOGY_COLUMNS = ["stage3_edge_uid", "geometry", "from_stage3_node_uid", "to_stage3_node_uid"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
        allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def atomic_parquet(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def grid_keys(frame, lon, lat, cfg):
    gx = np.floor((frame[lon].to_numpy(float) - cfg["grid_origin_wgs84"][0])
                  / cfg["grid_size_degrees"]).astype(int)
    gy = np.floor((frame[lat].to_numpy(float) - cfg["grid_origin_wgs84"][1])
                  / cfg["grid_size_degrees"]).astype(int)
    return pd.Series([f"{x}_{y}" for x, y in zip(gx, gy)], index=frame.index)


def common_sites(reference, cfg):
    """All-demand, 16-day pool. Representative is an observed node, not centroid."""
    work = reference.copy()
    work["cell"] = grid_keys(work, "lon_wgs84", "lat_wgs84", cfg)
    ranked = work.groupby("cell").train_pickup_count.sum().reset_index()
    ranked = ranked.sort_values(["train_pickup_count", "cell"], ascending=[False, True])
    sites = []
    for index, cell in enumerate(ranked.cell.iloc[:cfg["candidate_site_count"]]):
        group = work.loc[work.cell.eq(cell)]
        nodes = group.groupby(["node_id", "lon_wgs84", "lat_wgs84"], as_index=False).train_pickup_count.sum()
        weights = nodes.train_pickup_count.to_numpy(float)
        x = nodes.lon_wgs84.to_numpy(float) * math.cos(math.radians(34.25))
        y = nodes.lat_wgs84.to_numpy(float)
        nodes["center_gap"] = (x - np.average(x, weights=weights))**2 + (y - np.average(y, weights=weights))**2
        anchor = nodes.sort_values(["center_gap", "node_id"], kind="stable").iloc[0]
        sites.append(dict(site_id=f"S{index + 1:02d}", cell=str(cell),
            lon_wgs84=float(anchor.lon_wgs84), lat_wgs84=float(anchor.lat_wgs84),
            historical_pickups=int(weights.sum()), node_id=str(anchor.node_id)))
    return sites, work


def choose_tasks(templates, sites, cfg, cut):
    """Prespecified C/non-C diagnostic strata, not a population service sample."""
    work = templates.loc[templates.date.astype(str).eq(cfg["historical_date"])
        & templates.common_eligible & templates.release_second.ge(cut)
        & templates.release_second.lt(cut + cfg["horizon_s"])].copy()
    work["cell"] = grid_keys(work, "start_lon_wgs84", "start_lat_wgs84", cfg)
    work = work.loc[work.cell.isin([s["cell"] for s in sites])].copy()
    work["priority"] = work.order_id.astype(str).map(lambda order: hashlib.sha256(
        f"{cfg['historical_date']}|{order}|{cfg['selection_seed']}".encode()).hexdigest())
    work = work.sort_values(["priority", "order_id"], kind="stable")
    wanted = cfg["maximum_tasks"]
    c = work.loc[work.compatible_C].head(cfg["diagnostic_C_task_target"])
    other = work.loc[~work.compatible_C].head(wanted - len(c))
    picked = pd.concat([c, other])
    if len(picked) < wanted:
        picked = pd.concat([picked, work.loc[~work.order_id.isin(picked.order_id)].head(wanted - len(picked))])
    picked = picked.sort_values(["release_second", "order_id"], kind="stable").reset_index(drop=True)
    picked["job_id"] = [f"J{i + 1:02d}" for i in range(len(picked))]
    return picked, dict(cell_window_orders=len(work), cell_window_C_orders=int(work.compatible_C.sum()),
        selected_orders=len(picked), selected_C_orders=int(picked.compatible_C.sum()),
        sampling="STABLE_HASH_WITH_UP_TO_4_C_TASKS_REMAINDER_NON_C_DIAGNOSTIC_NOT_POPULATION_ESTIMATE")


def interface_supported(left, right, left_point, right_point, tolerance):
    """No arbitrary road jump or mid-block reversal at a customer/empty join.

    Same directed edge uses a bounded location snap. Different edges require
    exact directed endpoint connectivity and both stops near that endpoint.
    A distinct historical-reverse identity has genuinely swapped endpoints.
    """
    if left["uid"] == right["uid"]:
        if (_gap_m(left_point, right_point) > tolerance
                or max(_point_geometry_gap(left_point, left["geometry"]),
                       _point_geometry_gap(right_point, right["geometry"])) > tolerance):
            return False, "SAME_DIRECTED_EDGE_SNAP_GAP"
        return True, "SAME_DIRECTED_EDGE_WITHIN_SNAP_TOLERANCE"
    if not left["to_node"] or left["to_node"] != right["from_node"]:
        return False, "CUSTOMER_EMPTY_DIRECTED_ENDPOINT_DISCONNECT"
    if max(_gap_m(left_point, left["geometry"][-1]),
           _gap_m(right_point, right["geometry"][0])) > tolerance:
        return False, "CUSTOMER_EMPTY_JOIN_TOO_FAR_FROM_SHARED_ENDPOINT"
    return True, "EXACT_DIRECTED_ENDPOINT_WITHIN_SNAP_TOLERANCE"


def _point_geometry_gap(point, geometry):
    """Small, local equirectangular projection, used only at the two joins."""
    scale = math.cos(math.radians(point[1]))
    origin = np.array([point[0] * scale, point[1]], float)
    best = float("inf")
    for first, last in zip(geometry, geometry[1:]):
        a = np.array([first[0] * scale, first[1]], float)
        b = np.array([last[0] * scale, last[1]], float)
        delta = b - a
        norm = float(delta @ delta)
        fraction = float(np.clip((origin - a) @ delta / norm, 0, 1)) if norm else 0.0
        best = min(best, float(np.linalg.norm(origin - a - fraction * delta)) * 111195.08)
    return best


class InstanceRouter(ControlledEmptyRouter):
    """One columnar geometry copy avoids rereading the entire graph per connector."""
    def __init__(self, root, eta_adapter):
        super().__init__(root, eta_adapter)
        self.topology = _read_columns(self.root / S2A / "stage3_full_network_edges.parquet", TOPOLOGY_COLUMNS)
        self.geometry_cache = OrderedDict()

    def small_geometry(self, uids):
        needed = set(uids) - self.geometry_cache.keys()
        if needed:
            for row in _select(self.topology, "stage3_edge_uid", needed).to_pylist():
                self.geometry_cache[row["stage3_edge_uid"]] = dict(uid=row["stage3_edge_uid"],
                    from_node=row["from_stage3_node_uid"], to_node=row["to_stage3_node_uid"],
                    geometry=json.loads(row["geometry"]))
        found = {uid: self.geometry_cache[uid] for uid in uids if uid in self.geometry_cache}
        while len(self.geometry_cache) > 4096:
            self.geometry_cache.popitem(last=False)
        return found

    def _bearings_for(self, edge_uids):
        result = {}
        for uid, row in self.small_geometry(edge_uids).items():
            geometry = row["geometry"]
            result[uid] = ((_bearing(geometry[0], geometry[1]), _bearing(geometry[-2], geometry[-1]))
                           if len(geometry) >= 2 else (None, None))
        return result


class HistoricalSnapshotETA(ValhallaPickupTimeAdapter):
    """Reuse Actor access, NOT S0's Test31-calibrated multiplier or its table.

    A unit scenario factor is a transparent uncalibrated routing-time proxy.
    It is not a fitted traffic estimate and cannot support empirical wait gains.
    """
    def __init__(self, root, multiplier):
        self.root = Path(root).resolve()
        self._actor = None
        self.multiplier = float(multiplier)
        if self.multiplier != 1.0:
            raise ValueError("this first historical snapshot uses only the declared unit factor")

    def beta_for(self, timestamp):
        local = pd.Timestamp(timestamp)
        return int((local.hour * 60 + local.minute) // 15), self.multiplier


class JoinEvidence:
    """Reuse the historical reverse parser; check only EXTRA cross-part actions."""
    def __init__(self, root, router, selected):
        self.router = router
        date = str(selected.date.iloc[0])
        path = root / S3 / f"cache/train/date={date}/identity.parquet"
        table = _read_columns(path, IDENTITY_COLUMNS)
        table = _select(table, "order_id", selected.order_id.astype(str))
        self.tokens = {str(order): group.sort_values("route_sequence").copy()
            for order, group in table.to_pandas().groupby("order_id", sort=False)}
        if set(self.tokens) != set(selected.order_id.astype(str)):
            raise ValueError("selected historical route identities missing")
        overlay = _read_columns(root / S2A / "stage3_historical_direction_overlay.parquet",
            ["canonical_edge_uid", "physical_forward_stage3_edge_uid"])
        needed = set(table.to_pandas().loc[lambda d: d.route_token_type.eq("HISTORICAL_REVERSE_OVERLAY"), "canonical_edge_uid"])
        self.overlay = _select(overlay, "canonical_edge_uid", needed).to_pandas()
        self.overlay_forward = dict(zip(self.overlay.canonical_edge_uid, self.overlay.physical_forward_stage3_edge_uid))
        self.selected = selected.set_index("order_id")

    @staticmethod
    def _signature(row):
        return (str(row.intersection_complex_uid), str(row.incoming_stage3_edge_uid),
            str(row.outgoing_stage3_edge_uid), str(row.internal_stage3_edge_uids))

    def evaluate(self, source_order, target_order, routed, tolerance):
        previous = self.tokens[source_order].copy() if source_order else pd.DataFrame(columns=IDENTITY_COLUMNS)
        following = self.tokens[target_order].copy()
        empty = pd.DataFrame([dict(date=str(following.date.iloc[0]), order_id="EMPTY",
            route_sequence=i, canonical_edge_uid=None, route_token_type="FULL_NETWORK_EDGE",
            resolved_stage3_edge_uid=e["stage3_edge_uid"]) for i, e in enumerate(routed["edges"])], columns=IDENTITY_COLUMNS)
        pieces = [("PREVIOUS", previous), ("EMPTY", empty), ("FOLLOWING", following)]
        joined = pd.concat([p for _, p in pieces], ignore_index=True)
        uids = set(joined.loc[joined.route_token_type.eq("FULL_NETWORK_EDGE"), "resolved_stage3_edge_uid"].dropna())
        reverse_keys = set(joined.loc[joined.route_token_type.eq("HISTORICAL_REVERSE_OVERLAY"), "canonical_edge_uid"])
        uids |= {self.overlay_forward[k] for k in reverse_keys if k in self.overlay_forward}
        geometry = self.router.small_geometry(uids)
        boundary = _select(self.router._boundary, "stage3_edge_uid", uids).to_pandas()
        cids = set(boundary.intersection_complex_uid.astype(str))
        movements = _select(self.router._movements, "intersection_complex_uid", cids)
        movements = _select(_select(movements, "incoming_stage3_edge_uid", uids), "outgoing_stage3_edge_uid", uids).to_pandas()
        controls = _select(self.router._controls, "intersection_complex_uid", cids).to_pandas()
        classifier = MovementClassifier.__new__(MovementClassifier)
        classifier.complexes = controls.set_index("intersection_complex_uid").to_dict("index")
        classifier.lookup = {}
        for row in movements.itertuples(index=False):
            record = row._asdict()
            if record["movement_legality_state"] == "CERTIFIED_PROHIBITED":
                record["movement_legality_state"] = "PROHIBITED"
            classifier.lookup[tuple(str(record[c]) for c in MOVEMENT_COLUMNS[:3])] = SimpleNamespace(**record)
        classifier.bearings = self.router._bearings_for(uids)
        parser = HistoricalResearchParser(boundary, self.overlay.loc[self.overlay.canonical_edge_uid.isin(reverse_keys)], classifier)
        for canonical, virtual in parser.virtual.items():
            forward = self.overlay_forward[canonical]
            if forward in geometry:
                base = geometry[forward]
                geometry[virtual] = dict(uid=virtual, from_node=base["to_node"], to_node=base["from_node"],
                    geometry=list(reversed(base["geometry"])))

        def endpoint(frame, first):
            row = frame.iloc[0 if first else -1]
            uid = (parser.virtual.get(row.canonical_edge_uid) if row.route_token_type == "HISTORICAL_REVERSE_OVERLAY"
                   else row.resolved_stage3_edge_uid)
            return geometry.get(uid)

        interfaces = []
        if len(empty):
            if len(previous):
                src = self.selected.loc[source_order]
                interfaces.append((endpoint(previous, False), endpoint(empty, True),
                    (src.end_lon_wgs84, src.end_lat_wgs84), routed["snapped_origin"]))
            dst = self.selected.loc[target_order]
            interfaces.append((endpoint(empty, False), endpoint(following, True), routed["snapped_target"],
                (dst.start_lon_wgs84, dst.start_lat_wgs84)))
        elif len(previous):
            src, dst = self.selected.loc[source_order], self.selected.loc[target_order]
            interfaces.append((endpoint(previous, False), endpoint(following, True),
                (src.end_lon_wgs84, src.end_lat_wgs84), (dst.start_lon_wgs84, dst.start_lat_wgs84)))
        for left, right, left_point, right_point in interfaces:
            if left is None or right is None:
                return dict(supported=False, reason_codes=["JOIN_DIRECTION_GEOMETRY_MISSING"])
            valid, reason = interface_supported(left, right, left_point, right_point, tolerance)
            if not valid:
                return dict(supported=False, reason_codes=[reason])

        groups = []
        for label, frame in [("JOINT", joined), *pieces]:
            if len(frame):
                work = frame.copy()
                work["order_id"] = label
                work["route_sequence"] = np.arange(len(work))
                groups.append(work)
        identity, _, _, encounters = parser.parse(pd.concat(groups, ignore_index=True))
        if not identity.direction_supported.all():
            return dict(supported=False, reason_codes=["JOIN_DIRECTION_UNSUPPORTED"])
        individual = Counter(self._signature(r) for r in encounters.loc[~encounters.order_id.eq("JOINT")].itertuples(index=False)) if len(encounters) else Counter()
        added = []
        if len(encounters):
            for row in encounters.loc[encounters.order_id.eq("JOINT")].itertuples(index=False):
                key = self._signature(row)
                if individual[key]:
                    individual[key] -= 1
                else:
                    added.append(row._asdict())
        if any(individual.values()):
            return dict(supported=False, reason_codes=["JOIN_CHANGED_STANDALONE_MOVEMENT_PARSE"])
        extra = pd.DataFrame(added)
        if len(extra):
            extra["order_id"] = "JOIN"
        classified, summary = classifier.summarize(extra)
        valid, actions = classified.get("JOIN", (True, ()))
        if not valid:
            return dict(supported=False, reason_codes=["JOIN_MANEUVER_UNRESOLVED"])
        evaluations = {k: evaluate_research_compatibility(k, actions, dict(UNKNOWN_TRAFFIC),
            conservative_control_policy=CONTROL_POLICY, outside_budget=TRAFFIC_OUTSIDE_BUDGET) for k in ("HV", "C")}
        prohibited = any(m.certified_prohibited for m in actions)
        return dict(supported=not prohibited, reason_codes=["CERTIFIED_COMMON_PROHIBITION"] if prohibited else [],
            compatible_profiles=[k for k, result in evaluations.items() if result.compatible],
            join_encounter_count=len(actions), join_maneuvers=[m.maneuver for m in actions],
            C_reason_codes=list(evaluations["C"].reason_codes),
            join_bearing_fallback_count=int(summary.bearing_fallback_count.sum()) if len(summary) else 0)


def result_metrics(solution, horizon_s):
    jobs = [job for chain in solution["selected"] for job in chain["jobs"]]
    return dict(served=solution["objective"]["served"], empty_distance_m=solution["objective"]["empty_distance_m"],
        served_by_profile=dict(Counter(chain["profile_id"] for chain in solution["selected"] for _ in chain["jobs"])),
        mean_pickup_wait_s=float(np.mean([j["pickup_s"] - j["release_s"] for j in jobs])) if jobs else None,
        delayed_commit_count=sum(j["commit_s"] > j["release_s"] + 30 for j in jobs),
        cross_window_completion_count=sum(j["finish_s"] > horizon_s for j in jobs),
        selected=[dict(resource_id=c["resource_id"], profile_id=c["profile_id"], site_id=c["site_id"],
            job_ids=c["job_ids"], jobs=c["jobs"]) for c in solution["selected"]],
        lp=solution["lp"], enumeration=solution["enumeration"], runtime_s=solution["runtime_s"])


def comparison_metrics(fixed, joint):
    before = {j["job_id"]: j for c in fixed["selected"] for j in c["jobs"]}
    after = {j["job_id"]: j for c in joint["selected"] for j in c["jobs"]}
    shared = sorted(set(before) & set(after))
    distance_before = fixed["objective"]["empty_distance_m"]
    return dict(gained_jobs=sorted(set(after) - set(before)), lost_jobs=sorted(set(before) - set(after)),
        jointly_served_jobs=shared,
        paired_mean_pickup_wait_change_s=float(np.mean([after[j]["pickup_s"] - before[j]["pickup_s"] for j in shared])) if shared else None,
        empty_distance_reduction_pct=100 * (distance_before - joint["objective"]["empty_distance_m"]) / distance_before if distance_before else None,
        average_wait_population_is_identical=set(before) == set(after))


def augment_existing(root, summary):
    """Aggregation only: reuse the measured connections, never route a second time."""
    cfg = summary["config"]
    templates = pd.read_parquet(root / TEMPLATES)
    templates["origin_cell"] = grid_keys(templates, "start_lon_wgs84", "start_lat_wgs84", cfg)
    templates["destination_cell"] = grid_keys(templates, "end_lon_wgs84", "end_lat_wgs84", cfg)
    templates["hour"] = (templates.release_second // 3600).astype(int)
    by_hour = templates.groupby("hour").agg(orders=("order_id", "size"), C_orders=("compatible_C", "sum")).reset_index()
    by_date = templates.assign(date=templates.date.astype(str)).groupby("date").agg(
        orders=("order_id", "size"), C_orders=("compatible_C", "sum")).reset_index()
    flows = templates.groupby(["hour", "origin_cell", "destination_cell"], as_index=False).agg(
        orders=("order_id", "size"), C_orders=("compatible_C", "sum"),
        mean_predicted_service_time_s=("predicted_route_time_p50_s", "mean"))
    atomic_parquet(root / OUT / "historical_C_HV_od_flows.parquet", flows)
    summary["historical_inventory"] = dict(template_count=len(templates), C_task_count=int(templates.compatible_C.sum()),
        date_counts=by_date.to_dict("records"), hourly_counts=by_hour.to_dict("records"),
        OD_flow_row_count=len(flows), OD_flow_private_path=str(OUT / "historical_C_HV_od_flows.parquet"),
        demand_definition="THREE_TRAIN_MONDAYS_COMMON_ELIGIBLE_CUSTOMER_ROUTE_TEMPLATES_NOT_ALL_16_DAYS",
        task_compatibility_is_not_empty_connection_compatibility=True)
    for case in summary["cases"]:
        destination = root / case["private_instance_path"]
        instance = json.loads((destination / "instance.json").read_text(encoding="utf-8"))
        fixed = json.loads((destination / "fixed_layout_solution.json").read_text(encoding="utf-8"))
        joint = json.loads((destination / "joint_layout_solution.json").read_text(encoding="utf-8"))
        case["layout_comparison"] = comparison_metrics(fixed, joint)
        c_tasks = {t["job_id"] for t in instance["tasks"] if "C" in t["compatible_profiles"]}
        pairs = [c for c in instance["connections"] if c["origin_id"] in c_tasks and c["target_job_id"] in c_tasks]
        case["C_task_pair_state_counts"] = dict(Counter(c["evidence_state"] for c in pairs))
        case["C_task_pair_supported_but_C_incompatible_count"] = sum(c["supported"] and "C" not in c["compatible_profiles"] for c in pairs)
        release = {t["job_id"]: t["release_s"] for t in instance["tasks"]}
        forward = [c for c in pairs if release[c["origin_id"]] <= release[c["target_job_id"]]]
        case["release_ordered_C_task_pair_state_counts"] = dict(Counter(c["evidence_state"] for c in forward))
        case["release_ordered_C_task_pair_count"] = len(forward)
    atomic_json(root / OUT / "run_summary.json", summary)
    atomic_json(root / DOC / "summary.json", summary)
    return summary


def run(root, cfg_path=CONFIG):
    from stage4.analysis.capability_chain_instance_solver import solve_instance, validate_solution

    started = perf_counter()
    process = psutil.Process()
    cfg = json.loads((root / cfg_path).read_text(encoding="utf-8"))
    if not ("20161009" <= cfg["historical_date"] <= "20161024") or cfg["native_or_full_day_authorized"]:
        raise ValueError("only historical small instances are authorized")
    if cfg["maximum_tasks"] > 10 or cfg["candidate_site_count"] > 4 or cfg["resource_profiles"] != ["HV", "HV", "C"]:
        raise ValueError("small-instance resource bound exceeded")
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    templates = pd.read_parquet(root / TEMPLATES)
    templates["order_id"] = templates.order_id.astype(str)
    reference = pd.read_parquet(root / REFERENCE)
    sites, reference = common_sites(reference, cfg)
    router = InstanceRouter(root, HistoricalSnapshotETA(root, cfg["scenario_eta_multiplier"]))
    summary = dict(status="RUNNING", kind="TRAIN_DATA_SMALL_OFFLINE_INSTANCE_NOT_CITY_REPLAY",
        config=dict(cfg), sources=dict(templates=str(TEMPLATES), templates_sha256=sha(root / TEMPLATES),
            template_dates=sorted(templates.date.astype(str).unique()), reference=str(REFERENCE),
            reference_sha256=sha(root / REFERENCE), site_reference_train_dates="20161009--20161024",
            no_Test31_demand_or_calibration=True, no_new_M3_inference=True,
            customer_service_time="FROZEN_TRAIN_M3_PREDICTED_P50",
            connector_time="UNCALIBRATED_FROZEN_ROUTING_ENGINE_AUTO_TIME_AT_WINDOW_START",
            Test31_fitted_S0_eta_multiplier_reused=False), cases=[])
    atomic_json(root / OUT / "run_summary.json", summary)

    def check_budget():
        if router.route_request_count > cfg["maximum_route_queries"]:
            raise RuntimeError("small-instance route query budget exceeded; no truncation permitted")
        if process.memory_info().rss / 2**20 > cfg["maximum_rss_mib"]:
            raise RuntimeError("small-instance RSS budget exceeded")
        if perf_counter() - started > cfg["maximum_runtime_s"]:
            raise RuntimeError("small-instance wall-time budget exceeded")

    for cut in cfg["window_starts_s"]:
        case_started = perf_counter()
        case_id = f"TRAIN_{cfg['historical_date']}_{cut // 3600:02d}{cut % 3600 // 60:02d}"
        selected, sample = choose_tasks(templates, sites, cfg, cut)
        if not len(selected):
            raise ValueError("prespecified historical window has no task in the candidate cells")
        evidence = JoinEvidence(root, router, selected)
        clock = pd.Timestamp(cfg["historical_date"], tz="Asia/Shanghai") + pd.Timedelta(seconds=cut)
        pool = sorted(sites, key=lambda site: (-int(reference.loc[reference.cell.eq(site["cell"])
            & reference.time_bin_index.ge(cut // 900) & reference.time_bin_index.lt((cut + cfg["horizon_s"]) // 900), "train_pickup_count"].sum()), site["site_id"]))
        task_rows, local_rows = [], {}
        for row in selected.itertuples(index=False):
            task_rows.append(dict(job_id=row.job_id, release_s=float(row.release_second - cut),
                deadline_s=float(row.release_second - cut + cfg["patience_s"]),
                service_time_s=float(row.predicted_route_time_p50_s),
                compatible_profiles=[k for k in ("HV", "C") if bool(getattr(row, f"compatible_{k}"))]))
            local_rows[row.job_id] = row
        tasks = {task["job_id"]: task for task in task_rows}
        origins = [(s["site_id"], (s["lon_wgs84"], s["lat_wgs84"]), None) for s in sites]
        origins += [(r.job_id, (r.end_lon_wgs84, r.end_lat_wgs84), r.order_id) for r in selected.itertuples(index=False)]
        connections = []
        state_counts = Counter()
        c_reasons = Counter()
        route_started = perf_counter()
        for origin_id, origin_point, source_order in origins:
            for target, row in local_rows.items():
                if target == origin_id:
                    continue
                record = dict(origin_id=origin_id, target_job_id=target, supported=False,
                    compatible_profiles=[], travel_time_s=None, empty_distance_m=None,
                    evidence_state=None, reason_codes=[])
                lower_ready = (tasks[origin_id]["release_s"] + tasks[origin_id]["service_time_s"]) if origin_id in tasks else 0.0
                if lower_ready > tasks[target]["deadline_s"]:
                    record.update(evidence_state="TIME_IMPOSSIBLE_WITH_ZERO_CONNECTOR_LOWER_BOUND",
                        reason_codes=["FASTEST_PREVIOUS_SERVICE_ALREADY_AFTER_TARGET_DEADLINE"])
                else:
                    check_budget()
                    routed = router.route(*origin_point, row.start_lon_wgs84, row.start_lat_wgs84, clock)
                    record.update(travel_time_s=routed["duration_s"], empty_distance_m=routed["distance_m"],
                        route_edge_count=len(routed["edges"]), route_encounter_count=routed["encounter_count"],
                        snap_gap_origin_m=routed["snap_gap_origin_m"], snap_gap_target_m=routed["snap_gap_target_m"])
                    if not routed["common_supported"]:
                        record.update(evidence_state="COMMON_ROUTE_INPUT_UNSUPPORTED", reason_codes=list(routed["reason_codes"]))
                    elif routed["duration_s"] + max(lower_ready, tasks[target]["release_s"]) > tasks[target]["deadline_s"]:
                        record.update(evidence_state="TIME_IMPOSSIBLE_UNDER_SNAPSHOT", reason_codes=["CONNECTOR_EXCEEDS_ORIGINAL_PICKUP_DEADLINE"])
                    elif max(routed["snap_gap_origin_m"], routed["snap_gap_target_m"]) > cfg["interface_snap_tolerance_m"]:
                        record.update(evidence_state="COMMON_INTERFACE_INPUT_UNSUPPORTED", reason_codes=["EMPTY_ROUTE_SNAP_GAP_EXCEEDS_INSTANCE_TOLERANCE"])
                    else:
                        join = evidence.evaluate(source_order, str(row.order_id), routed, cfg["interface_snap_tolerance_m"])
                        if not join["supported"]:
                            record.update(evidence_state="COMMON_INTERFACE_INPUT_UNSUPPORTED", reason_codes=join["reason_codes"])
                        else:
                            allowed = set(routed["compatible_profiles"]) & set(join["compatible_profiles"])
                            allowed = sorted(allowed & {"HV", "C"})
                            reasons = sorted(set(routed["profile_reason_codes"].get("C", ())) | set(join["C_reason_codes"]))
                            record.update(supported=True, compatible_profiles=allowed,
                                evidence_state="SUPPORTED_MODELED_DIRECTED_CONNECTOR_AND_JOIN", C_reason_codes=reasons,
                                join_encounter_count=join["join_encounter_count"], join_maneuvers=join["join_maneuvers"],
                                join_bearing_fallback_count=join["join_bearing_fallback_count"])
                            c_reasons.update(reasons)
                state_counts[record["evidence_state"]] += 1
                connections.append(record)
            print(json.dumps(dict(case=case_id, completed_origin=origin_id,
                route_queries=router.route_request_count, elapsed_s=round(perf_counter() - started, 2),
                rss_mib=round(process.memory_info().rss / 2**20, 2))), flush=True)
        route_runtime = perf_counter() - route_started
        resources = [dict(resource_id="H1", profile_id="HV", start_s=0, end_s=cfg["horizon_s"],
                         site_ids=[pool[0]["site_id"]], fixed_site_id=pool[0]["site_id"]),
            dict(resource_id="H2", profile_id="HV", start_s=0, end_s=cfg["horizon_s"],
                 site_ids=[pool[1]["site_id"]], fixed_site_id=pool[1]["site_id"]),
            dict(resource_id="C1", profile_id="C", start_s=0, end_s=cfg["horizon_s"],
                 site_ids=[s["site_id"] for s in sites], fixed_site_id=pool[0]["site_id"])]
        instance = dict(schema_version=cfg["schema_version"], execution_step_s=cfg["execution_step_s"],
            case_id=case_id, tasks=task_rows, sites=sites, resources=resources, connections=connections,
            assumptions=dict(cfg), information="KNOWN_HISTORICAL_SCENARIO_ONLY_OFFLINE_BENCHMARK")
        fixed = solve_instance(instance, optimize_layout=False)
        joint = solve_instance(instance, optimize_layout=True)
        validate_solution(instance, fixed)
        validate_solution(instance, joint)
        if joint["objective"]["served"] < fixed["objective"]["served"]:
            raise RuntimeError("joint domain must include fixed layout")
        destination = root / OUT / case_id
        atomic_json(destination / "instance.json", instance)
        atomic_json(destination / "fixed_layout_solution.json", fixed)
        atomic_json(destination / "joint_layout_solution.json", joint)
        atomic_parquet(destination / "source_order_mapping.parquet", selected)
        atomic_parquet(destination / "connections.parquet", pd.DataFrame(connections))
        site_evidence = []
        for site in sites:
            outgoing = [c for c in connections if c["origin_id"] == site["site_id"] and c["supported"]
                and math.ceil(tasks[c["target_job_id"]]["release_s"] / cfg["execution_step_s"])
                    * cfg["execution_step_s"] + c["travel_time_s"] <= tasks[c["target_job_id"]]["deadline_s"]]
            direct_c = sum("C" in c["compatible_profiles"] and "C" in tasks[c["target_job_id"]]["compatible_profiles"] for c in outgoing)
            site_evidence.append(dict(site_id=site["site_id"], historical_pickups_all_train=site["historical_pickups"],
                first_C_task_count=direct_c, supported_first_HV_task_count=len(outgoing),
                is_fixed_C_hotspot=site["site_id"] == pool[0]["site_id"]))
        case = dict(case_id=case_id, sample=sample, sites=site_evidence,
            connection_state_counts=dict(state_counts), C_connector_failure_reason_counts=dict(c_reasons),
            common_input_failure_reason_counts=dict(Counter(reason for c in connections
                if c["evidence_state"] in ("COMMON_ROUTE_INPUT_UNSUPPORTED", "COMMON_INTERFACE_INPUT_UNSUPPORTED")
                for reason in c["reason_codes"])),
            supported_job_to_job_connectors=sum(c["supported"] and c["origin_id"] in tasks for c in connections),
            supported_C_job_to_job_connectors=sum(c["supported"] and c["origin_id"] in tasks
                and "C" in c["compatible_profiles"] and "C" in tasks[c["origin_id"]]["compatible_profiles"]
                and "C" in tasks[c["target_job_id"]]["compatible_profiles"] for c in connections),
            fixed_layout=result_metrics(fixed, cfg["horizon_s"]), joint_layout=result_metrics(joint, cfg["horizon_s"]),
            served_gain=joint["objective"]["served"] - fixed["objective"]["served"],
            routing_and_join_runtime_s=route_runtime, total_case_runtime_s=perf_counter() - case_started,
            private_instance_path=str(destination.relative_to(root)),
            interpretation="EXACT_ONLY_IN_THIS_SNAPSHOT_TASK_SITE_SUPPORTED_CONNECTOR_DOMAIN_NOT_CITY_OR_ONLINE_OPTIMUM")
        summary["cases"].append(case)
        atomic_json(root / OUT / "run_summary.json", summary)
        del evidence, selected, instance, connections, fixed, joint
        gc.collect()
        check_budget()
    summary.update(status="W1_W2_SMALL_INSTANCES_COMPLETE", runtime_s=perf_counter() - started,
        peak_rss_mib=process.memory_info().peak_wset / 2**20,
        routing=router.diagnostics(), extra_columnar_topology_mib=router.topology.nbytes / 2**20,
        gpu_used=False, full_day_runs=0, new_training_runs=0)
    augment_existing(root, summary)
    print(json.dumps(dict(status=summary["status"], runtime_s=summary["runtime_s"],
        peak_rss_mib=summary["peak_rss_mib"], cases=len(summary["cases"]))), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--summarize-existing", action="store_true", help="aggregate saved results without routing or solving")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.summarize_existing:
        summary = json.loads((root / OUT / "run_summary.json").read_text(encoding="utf-8"))
        augment_existing(root, summary)
        print(json.dumps(dict(status="EXISTING_RESULTS_AGGREGATED_NO_RERUN", cases=len(summary["cases"]))), flush=True)
    else:
        run(root, args.config)


if __name__ == "__main__":
    main()
