"""Opt-in historical research direction overlay; never edits frozen AV topology.

Stream complete orders, preserve reversed geometry/roles under distinct virtual
identities, and apply the same input/legality population to HV and C/M/A.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import psutil

from stage3.odd_tod.capability_envelope import parse_route_complex_encounters
from stage3.odd_tod.research_compatibility import evaluate_research_compatibility
from stage3.scripts.traffic_state_batch1 import EDGE, sha, write_json, write_parquet
from stage3.scripts.traffic_state_support import decompose
from stage4.analysis.flexibility_prepare import MovementClassifier, S2A, S2B, S3, S4
from stage4.analysis.traffic_research_prepare import aggregate
from stage4.replay_foundation import add_coordinate_lineage, _timestamp_series

CONFIG = Path("stage4/config/symmetric_flexibility_full_day_v1.json")
OUT = Path("stage4/output/symmetric_flexibility_v1/input")
DOC = Path("stage4/docs/flexibility_dispatch/symmetric_full_day")
OLD_INPUT = Path("stage4/output/flexibility_dispatch_v1/input")


class HistoricalResearchParser:
    """Reversal means a NEW directed identity, never the forward traversed edge."""
    def __init__(self, boundary, overlay, classifier):
        self.classifier = classifier
        self.virtual = {}
        extra = []
        grouped = {uid: g for uid, g in boundary.groupby("stage3_edge_uid", sort=False)}
        for r in overlay.itertuples(index=False):
            forward = r.physical_forward_stage3_edge_uid
            if pd.isna(forward) or forward not in classifier.bearings:
                continue  # Unsupported geometry is common input exclusion, not AV inability.
            first, last = classifier.bearings[forward]
            if first is None or last is None:
                continue
            uid = "RESEARCH_REVERSE:" + str(r.canonical_edge_uid) + ":R"
            self.virtual[str(r.canonical_edge_uid)] = uid
            classifier.bearings[uid] = ((last + 180) % 360, (first + 180) % 360)
            if forward in grouped:
                b = grouped[forward].copy()
                b["stage3_edge_uid"] = uid
                b["boundary_role"] = b.boundary_role.replace({"INCOMING": "OUTGOING", "OUTGOING": "INCOMING"})
                extra.append(b)
        self.boundary = pd.concat([boundary, *extra], ignore_index=True)
        self.roles = {}
        for r in self.boundary.itertuples(index=False):
            self.roles.setdefault(str(r.stage3_edge_uid), {}).setdefault(str(r.intersection_complex_uid), set()).add(str(r.boundary_role))
        self.movement_pairs = set(classifier.lookup)
        # New reverse movements use actual reversed boundary tangents. A forward
        # movement/restriction is NEVER copied to the reverse direction.
        self.lookup = pd.DataFrame(columns=[
            "intersection_complex_uid", "incoming_stage3_edge_uid", "outgoing_stage3_edge_uid"])

    def parse(self, frame):
        work = frame.copy()
        reverse = work.route_token_type.eq("HISTORICAL_REVERSE_OVERLAY")
        virtual = work.canonical_edge_uid.map(self.virtual)
        supported_reverse = reverse & virtual.notna()
        work.loc[supported_reverse, "resolved_stage3_edge_uid"] = virtual[supported_reverse]
        # Local parser vocabulary only; source route_token_type is retained in
        # the frozen file and never rewritten as a full-network observation.
        work.loc[supported_reverse, "route_token_type"] = "FULL_NETWORK_EDGE"
        supported = work.route_token_type.eq("FULL_NETWORK_EDGE") & work.resolved_stage3_edge_uid.notna()
        identity = pd.DataFrame({"order_id": work.order_id, "supported": supported,
            "reverse": reverse, "unresolved": frame.route_token_type.eq("UNRESOLVED"),
            "unsupported_reverse": reverse & ~supported_reverse}).groupby("order_id", sort=False).agg(
                direction_supported=("supported", "all"), reverse_token_count=("reverse", "sum"),
                unresolved_token_count=("unresolved", "sum"), unsupported_reverse_token_count=("unsupported_reverse", "sum"))
        encounters = parse_route_complex_encounters(work, self.boundary, self.lookup,
            roles_cache=self.roles, movement_pairs_cache=self.movement_pairs)
        classified, summary = self.classifier.summarize(encounters)
        return identity, classified, summary, encounters


def complete_batches(path, columns, ids=None):
    """Hold only the final (possibly split) order across Arrow batch boundaries."""
    carry = None
    for batch in pq.ParquetFile(path).iter_batches(batch_size=65536, columns=columns):
        frame = batch.to_pandas()
        if ids is not None:
            frame = frame.loc[frame.order_id.isin(ids)]
        if carry is not None:
            frame = pd.concat([carry, frame], ignore_index=True)
        if frame.empty:
            carry = None
            continue
        last = frame.order_id.iloc[-1]
        carry = frame.loc[frame.order_id.eq(last)].copy()
        done = frame.loc[~frame.order_id.eq(last)]
        if len(done):
            yield done
    if carry is not None and len(carry):
        yield carry


IDENTITY_COLUMNS = ["date", "order_id", "route_sequence", "canonical_edge_uid", "route_token_type", "resolved_stage3_edge_uid"]


def classify_day(parser, identity_path, traffic, predictions, cfg, ids=None):
    traffic = traffic.set_index("order_id").to_dict("index")
    predictions = predictions.set_index("order_id").predicted_route_time_p50_s.to_dict()
    rows = []
    seen = set()
    reported = 0
    for frame in complete_batches(identity_path, IDENTITY_COLUMNS, ids):
        identity, classified, summary, encounters = parser.parse(frame)
        if seen.intersection(identity.index):
            raise ValueError("identity file is not order-contiguous; cannot stream complete orders")
        seen.update(identity.index)
        summaries = summary.set_index("order_id").to_dict("index") if len(summary) else {}
        for order, record in identity.iterrows():
            valid, movements = classified.get(str(order), (True, ()))
            t = traffic.get(order)
            pred = predictions.get(order, np.nan)
            ready = bool(valid and t is not None and np.isfinite(pred) and pred > 0)
            prohibited = any(m.certified_prohibited for m in movements)
            common = bool(ready and record.direction_supported and not prohibited)
            reasons = []
            if not ready: reasons.append("MISSING_RESEARCH_INPUT")
            if not record.direction_supported: reasons.append("UNSUPPORTED_ROUTE_IDENTITY")
            if prohibited: reasons.append("CERTIFIED_COMMON_PROHIBITION")
            allowed = {}
            if common:
                partition = {s: float(t[c]) for s, c in {
                    "NC": "free_share", "ML": "background_mixed_share", "MR": "relative_mixed_share",
                    "CG": "congested_share", "SC": "severe_share", "U": "unknown_share"}.items()}
                for k in ("C", "M", "A", "HV"):
                    allowed[k] = evaluate_research_compatibility(k, movements, partition,
                        outside_budget=cfg["traffic_outside_budget"],
                        conservative_control_policy=cfg["conservative_control_policy"]).compatible
            s = summaries.get(order, {})
            rows.append(dict(order_id=str(order), research_data_ready=ready, common_eligible=common,
                common_exclusion_reasons=json.dumps(reasons),
                **{f"compatible_{k}": common and allowed.get(k, False) for k in ("C", "M", "A", "HV")},
                **record.to_dict(), encounter_count=int(s.get("encounter_count", 0)),
                control_assumption_count=int(s.get("control_assumption_count", 0)),
                bearing_fallback_count=int(s.get("bearing_fallback_count", 0))))
        del frame, identity, classified, summary, encounters
        if len(seen) - reported >= 3000:
            reported = len(seen)
            print(json.dumps(dict(identity=str(identity_path), processed_orders=reported,
                rss_mib=psutil.Process().memory_info().rss/2**20)), flush=True)
    result = pd.DataFrame(rows)
    if not result.compatible_A.equals(result.compatible_HV):
        raise RuntimeError("human-equivalent A capability must equal HV in the common population")
    if not ((~result.compatible_C | result.compatible_M) & (~result.compatible_M | result.compatible_A)).all():
        raise RuntimeError("C/M/A capability nesting violated")
    return result


def prepare(root):
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    cfg = json.loads((root / CONFIG).read_text())
    destination = root / OUT
    destination.mkdir(parents=True, exist_ok=False)
    complexes = pd.read_parquet(root / OLD_INPUT / "complex_control_overlay.parquet")
    classifier = MovementClassifier(root, complexes)
    boundary = pd.read_parquet(root / S2B / "stage3_edge_complex_boundary_index.parquet")
    overlay = pd.read_parquet(root / S2A / "stage3_historical_direction_overlay.parquet")
    parser = HistoricalResearchParser(boundary, overlay, classifier)
    old = pd.read_parquet(root / OLD_INPUT / "test31_research_routes.parquet")
    traffic = old.drop(columns=["research_data_ready", "compatible_C", "compatible_M", "compatible_A",
        "control_assumption_count", "bearing_fallback_count", "encounter_count", "signalized_encounter_count",
        "unresolved_maneuver_count"], errors="ignore")
    new = classify_day(parser, root / S4 / "test31_route_identity_resolution.parquet", traffic,
                       old, cfg)
    routes = new.merge(traffic, on="order_id", validate="one_to_one")
    write_parquet(destination / "test31_research_routes.parquet", routes)
    train_parts, train_sources = [], []
    traffic_cfg = json.loads((root / "stage3/config/traffic_state_batch1.json").read_text())
    reference = pd.read_parquet(root / "stage3/output/traffic_state_batch1/train_reference.parquet")
    for date in cfg["forecast_train_dates"]:
        day = root / f"stage1/input_v1/split=train/date={date}"
        base = pd.concat([pd.read_parquet(p, columns=["order_id", "departure_time"]) for p in sorted(day.glob("bucket=*/order_base.parquet"))], ignore_index=True)
        base["release_second"] = (_timestamp_series(base.departure_time) - pd.Timestamp(date, tz="Asia/Shanghai")).dt.total_seconds()
        base["order_id"] = base.order_id.astype(str)
        cache = root / S3 / f"cache/train/date={date}"
        dynamic = pd.read_parquet(cache / "dynamic_descriptors.parquet")
        prediction = root / S3 / f"cache/m3/date={date}.parquet"
        tokens = pd.read_parquet(prediction, columns=["order_id", "traversal_id", "pred_pace_p50", "pred_crawl", "pred_stop", "travel_time_p50_s"])
        route = pd.read_parquet(root / f"stage2/output_v4/route_conditioned_dataset/revealed_route_proxy/day={date}.parquet", columns=["order_id", "traversal_id", EDGE])
        tokens = tokens.merge(route, on=["order_id", "traversal_id"], validate="one_to_one")
        t = aggregate(decompose(tokens.merge(reference, on=EDGE, how="left", validate="many_to_one"), traffic_cfg))
        del tokens, route
        common = classify_day(parser, cache / "identity.parquet", t, dynamic, cfg)
        coords = pd.read_parquet(root / f"stage0/work_v6_final/candidate_manifests/date={date}.parquet",
            columns=["order_id", "start_lon", "start_lat", "end_lon", "end_lat"], filters=[("order_id", "in", list(dynamic.order_id))])
        coords = add_coordinate_lineage(coords)
        library = base.merge(coords[["order_id", "start_lon_wgs84", "start_lat_wgs84", "end_lon_wgs84", "end_lat_wgs84"]], on="order_id", validate="one_to_one").merge(
            dynamic[["order_id", "predicted_route_time_p50_s"]], on="order_id", validate="one_to_one").merge(common, on="order_id", validate="one_to_one")
        library = library.loc[library.common_eligible].copy()
        library["date"] = date
        train_parts.append(library)
        train_sources.append(dict(date=date, templates=len(library), frozen_prediction_sha256=sha(prediction)))
        print(json.dumps(dict(prepared_date=date, templates=len(library))), flush=True)
        del base, dynamic, t, common, coords, library
        gc.collect()
    templates = pd.concat(train_parts, ignore_index=True)
    write_parquet(destination / "train_request_templates.parquet", templates)
    counts = dict(orders=len(routes), common_eligible=int(routes.common_eligible.sum()),
        compatible={k:int(routes[f"compatible_{k}"].sum()) for k in ("C", "M", "A", "HV")},
        reverse_orders=int(routes.reverse_token_count.gt(0).sum()),
        reverse_supported_orders=int((routes.reverse_token_count.gt(0) & routes.common_eligible).sum()),
        unsupported_reverse_orders=int(routes.unsupported_reverse_token_count.gt(0).sum()),
        unresolved_orders=int(routes.unresolved_token_count.gt(0).sum()),
        before_compatible={k:int(old[f"compatible_{k}"].sum()) for k in "CMA"},
        new_encounters=int(routes.encounter_count.sum()),
        original_encounters=int(old.encounter_count.fillna(0).sum()),
        common_exclusion_counts=routes.loc[~routes.common_eligible,"common_exclusion_reasons"].value_counts().to_dict())
    summary = dict(status="COMPLETE", policy_version="SYMMETRIC_HISTORICAL_RESEARCH_V1", config_sha256=sha(root / CONFIG),
        test31_routes=counts, train_templates=train_sources,
        virtual_reverse_identity_count=len(parser.virtual), physical_forward_reference_is_not_traversed_edge=True,
        frozen_topology_changed=False, signal_collection_calls=0, new_m3_inference=False,
        missing_input_is_common_exclusion_not_capability_failure=True,
        A_equals_HV_capability=True, full_day_Train_library_not_test31_future_demand=True,
        peak_rss_mib=psutil.Process().memory_info().peak_wset / 2**20)
    write_json(destination / "preparation_summary.json", summary)
    write_json(root / DOC / "input_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare(Path.cwd())
