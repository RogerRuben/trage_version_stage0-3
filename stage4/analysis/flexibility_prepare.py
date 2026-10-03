"""Positive-control overlay and prediction-only historical request templates.

Reads one Train day at a time and only the two prespecified time bands.
No HTTP, no clustering, no new M3 inference, no Test31 future demand model fit.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from stage3.odd_tod.intersection_complex import _bearing, signed_turn, turn_type
from stage3.odd_tod.research_compatibility import ResearchMovement, compatible_profiles
from stage3.scripts.traffic_state_batch1 import EDGE, sha, write_json, write_parquet
from stage3.scripts.traffic_state_support import decompose
from stage4.analysis.flexibility_small_instances import signal_input_inventory
from stage4.analysis.traffic_research_prepare import aggregate
from stage4.replay_foundation import add_coordinate_lineage, _timestamp_series

CONFIG = Path("stage4/config/flexibility_windows_v1.json")
OUT = Path("stage4/output/flexibility_dispatch_v1/input")
DOC = Path("stage4/docs/flexibility_dispatch")
S2A = Path("stage3/output/odd_tod/s2a")
S2B = Path("stage3/output/odd_tod/s2b/final")
S3 = Path("stage3/output/odd_tod/s3")
S4 = Path("stage3/output/odd_tod/s4")


def _xy(lon, lat):
    return np.column_stack((np.asarray(lon, float) * 111320 * np.cos(np.deg2rad(34.25)),
                            np.asarray(lat, float) * 110540))


def signal_overlay(root, cfg):
    inventory = signal_input_inventory(cfg["signal_release"])
    positions = pd.read_csv(Path(cfg["signal_release"]) / "results/xian_signalized_intersections_final.csv")
    nodes = pd.read_parquet(root / S2A / "stage3_full_network_nodes.parquet",
                            columns=["stage3_node_uid", "lon", "lat", "osm_node_id"])
    membership = pd.read_parquet(root / S2B / "stage3_intersection_node_membership.parquet",
                                 columns=["stage3_node_uid", "intersection_complex_uid"])
    complexes = pd.read_parquet(root / S2B / "stage3_intersection_complexes.parquet",
        columns=["intersection_complex_uid", "signal_evidence_present", "roundabout_evidence_present",
                 "grade_separation_evidence_present"])
    points = membership.merge(nodes, on="stage3_node_uid", validate="one_to_one")
    tree = cKDTree(_xy(points.lon, points.lat))
    direct = dict(zip(points.stage3_node_uid, points.intersection_complex_uid))
    osm_direct = dict(zip(points.osm_node_id.dropna().astype(int).astype(str),
                         points.loc[points.osm_node_id.notna(), "intersection_complex_uid"]))
    grade = complexes.set_index("intersection_complex_uid").grade_separation_evidence_present.to_dict()
    links, associations = [], []
    for row in positions.itertuples(index=False):
        anchor_ids = json.loads(row.member_nodes) if isinstance(row.member_nodes, str) else []
        anchors = {direct[str(n)] for n in anchor_ids if str(n) in direct}
        anchors |= {osm_direct[str(n)] for n in anchor_ids if str(n) in osm_direct}
        # Do not absorb a surface signal into a grade-separated complex via proximity.
        anchors = {c for c in anchors if not grade.get(c, False)}
        point = _xy([row.longitude_wgs84], [row.latitude_wgs84])[0]
        nearby = tree.query_ball_point(point, cfg["signal_association_radius_m"])
        distances = {}
        for j in nearby:
            c = str(points.iloc[j].intersection_complex_uid)
            if grade.get(c, False):
                continue
            distance = float(np.linalg.norm(tree.data[j] - point))
            distances[c] = min(distance, distances.get(c, np.inf))
        ranked = sorted(distances.items(), key=lambda x: (x[1], x[0]))
        anchors &= set(distances)  # Exact identity still needs local spatial agreement.
        if anchors:
            selected, basis = sorted(anchors), "EXACT_MEMBER_NODE_ID"
        elif ranked and (len(ranked) == 1 or ranked[1][1] - ranked[0][1] >= cfg["signal_association_unique_margin_m"]):
            selected, basis = [ranked[0][0]], "UNIQUE_NEAREST_SURFACE_COMPLEX"
        else:
            selected, basis = [], "AMBIGUOUS_OR_NO_SURFACE_COMPLEX_ASSOCIATION"
        associations.append(dict(signal_position_id=row.intersection_id, mapping_basis=basis,
            mapped_complex_count=len(selected), candidate_complex_count=len(ranked),
            nearest_distance_m=ranked[0][1] if ranked else None,
            confidence_class=row.confidence_class, field_verified=False))
        for complex_id in selected:
            links.append(dict(signal_position_id=row.intersection_id,
                              intersection_complex_uid=complex_id, mapping_basis=basis))
    link_frame = pd.DataFrame(links, columns=["signal_position_id", "intersection_complex_uid", "mapping_basis"])
    positive = set(link_frame.intersection_complex_uid)
    complexes["signalized_research"] = complexes.signal_evidence_present | complexes.intersection_complex_uid.isin(positive)
    complexes["control_basis"] = np.select(
        [complexes.intersection_complex_uid.isin(positive), complexes.signal_evidence_present],
        ["NEW_699_POSITIVE_OVERLAY", "FROZEN_OSM_POSITIVE"],
        default="RESEARCH_ASSUMED_NON_SIGNAL_NOT_OBSERVED_ABSENCE")
    summary = {**inventory, "canonical_complex_mapping_completed": True,
               "position_mapping_basis_counts": pd.DataFrame(associations).mapping_basis.value_counts().to_dict(),
               "mapped_new_positive_complex_count": len(positive),
               "frozen_positive_complex_count": int(complexes.signal_evidence_present.sum()),
               "union_positive_complex_count": int(complexes.signalized_research.sum()),
               "unlisted_control_policy": cfg["unlisted_control_policy"],
               "grade_separated_proximity_transfer_allowed": False}
    write_parquet(root / OUT / "signal_position_associations.parquet", pd.DataFrame(associations))
    write_parquet(root / OUT / "signal_complex_links.parquet", link_frame)
    write_parquet(root / OUT / "complex_control_overlay.parquet", complexes)
    return complexes, summary


class MovementClassifier:
    def __init__(self, root, complexes):
        self.complexes = complexes.set_index("intersection_complex_uid").to_dict("index")
        columns = ["intersection_complex_uid", "incoming_stage3_edge_uid", "outgoing_stage3_edge_uid",
                   "route_turn_type", "restriction_enforcement_certified", "movement_legality_state"]
        movements = pd.read_parquet(root / S2B / "stage3_route_movement_lookup.parquet", columns=columns)
        self.lookup = {tuple(str(getattr(r, c)) for c in columns[:3]): r for r in movements.itertuples(index=False)}
        edges = pd.read_parquet(root / S2A / "stage3_full_network_edges.parquet", columns=["stage3_edge_uid", "geometry"])
        self.bearings = {}
        for row in edges.itertuples(index=False):
            g = json.loads(row.geometry)
            self.bearings[row.stage3_edge_uid] = (
                _bearing(g[0], g[1]) if len(g) >= 2 else None,
                _bearing(g[-2], g[-1]) if len(g) >= 2 else None,
            )

    def summarize(self, encounters):
        records = []
        for row in encounters.itertuples(index=False):
            cid, incoming, outgoing = str(row.intersection_complex_uid), str(row.incoming_stage3_edge_uid), str(row.outgoing_stage3_edge_uid)
            control = self.complexes[cid]
            known = self.lookup.get((cid, incoming, outgoing))
            maneuver = known.route_turn_type if known else "UNKNOWN"
            basis = "FROZEN_MOVEMENT_LOOKUP"
            if maneuver == "UNKNOWN":
                entry = self.bearings.get(incoming, (None, None))[1]
                exit_ = self.bearings.get(outgoing, (None, None))[0]
                maneuver = turn_type(signed_turn(entry, exit_))
                basis = "BOUNDARY_TANGENT_BEARING_FALLBACK"
            if control["roundabout_evidence_present"]:
                maneuver = "ROUNDABOUT"
            prohibited = bool(known and known.restriction_enforcement_certified
                              and known.movement_legality_state == "PROHIBITED")
            records.append(dict(order_id=str(row.order_id), maneuver=maneuver,
                signalized=bool(control["signalized_research"]), control_basis=control["control_basis"],
                maneuver_basis=basis, certified_prohibited=prohibited))
        frame = pd.DataFrame(records, columns=["order_id", "maneuver", "signalized", "control_basis",
                                             "maneuver_basis", "certified_prohibited"])
        if frame.empty:
            return {}, pd.DataFrame(columns=["order_id"])
        result = {}
        summaries = []
        for order, group in frame.groupby("order_id", sort=False):
            valid = group.maneuver.isin(["LEFT", "RIGHT", "STRAIGHT", "UTURN", "ROUNDABOUT"]).all()
            movements = tuple(ResearchMovement(r.maneuver, r.signalized,
                "RESEARCH_ASSUMPTION" if r.control_basis.startswith("RESEARCH_") else "POSITIVE_EVIDENCE",
                r.certified_prohibited) for r in group.itertuples(index=False)) if valid else ()
            result[order] = (bool(valid), movements)
            summaries.append(dict(order_id=order, encounter_count=len(group),
                signalized_encounter_count=int(group.signalized.sum()),
                control_assumption_count=int(group.control_basis.str.startswith("RESEARCH_").sum()),
                bearing_fallback_count=int(group.maneuver_basis.eq("BOUNDARY_TANGENT_BEARING_FALLBACK").sum()),
                unresolved_maneuver_count=int((~group.maneuver.isin(["LEFT", "RIGHT", "STRAIGHT", "UTURN", "ROUNDABOUT"])).sum())))
        return result, pd.DataFrame(summaries)


def route_compatibility(orders, classified, traffic, direction_ok, cfg):
    rows = []
    traffic = traffic.set_index("order_id").to_dict("index")
    for order in orders:
        order = str(order)
        valid, movements = classified.get(order, (True, ()))
        shares = traffic.get(order)
        ready = valid and shares is not None
        if ready:
            partition = {s: float(shares[c]) for s, c in {
                "NC": "free_share", "ML": "background_mixed_share", "MR": "relative_mixed_share",
                "CG": "congested_share", "SC": "severe_share", "U": "unknown_share"}.items()}
            allowed = compatible_profiles(movements, partition, direction_routable=bool(direction_ok.get(order, False)),
                outside_budget=cfg["traffic_outside_budget"], conservative_control_policy=cfg["conservative_control_policy"])
        else:
            allowed = frozenset()
        rows.append(dict(order_id=order, research_data_ready=ready,
                         **{f"compatible_{k}": k in allowed for k in ("C", "M", "A")}))
    return pd.DataFrame(rows)


def test31_policy(root, classifier, cfg):
    descriptors = pd.read_parquet(root / S4 / "test31_original_route_descriptors.parquet")
    encounters = pd.read_parquet(root / S4 / "test31_route_complex_encounters.parquet")
    classified, movement_summary = classifier.summarize(encounters)
    traffic = pd.read_parquet(root / "stage4/output/traffic_research/policy_date=20161031.parquet")
    traffic = traffic.loc[traffic.profile_id.eq("M")].drop(columns=["profile_id"])
    direction = descriptors.set_index("order_id").eval("reverse_overlay_token_count == 0 and unresolved_token_count == 0").to_dict()
    result = route_compatibility(descriptors.order_id, classified, traffic, direction, cfg)
    result = result.merge(movement_summary, on="order_id", how="left", validate="one_to_one")
    result = result.merge(traffic, on="order_id", how="left", validate="one_to_one")
    result = result.merge(descriptors[["order_id", "predicted_route_time_p50_s", "route_max_A_c", "route_max_M_c",
        "route_max_D_c", "route_max_L_c", "max_route_speed_domain_kmh", "speed_cv_E", "speed_cv_Q", "speed_cv_C",
        "acceleration_rms_E", "acceleration_rms_Q", "acceleration_rms_C"]], on="order_id", validate="one_to_one")
    write_parquet(root / OUT / "test31_research_routes.parquet", result)
    return {"orders": len(result), "data_ready": int(result.research_data_ready.sum()),
            "compatible": {k: int(result[f"compatible_{k}"].sum()) for k in ("C", "M", "A")},
            "bearing_fallback_encounters": int(result.bearing_fallback_count.fillna(0).sum()),
            "unresolved_maneuver_encounters": int(result.unresolved_maneuver_count.fillna(0).sum())}


def train_library(root, classifier, cfg):
    traffic_cfg = json.loads((root / "stage3/config/traffic_state_batch1.json").read_text())
    reference = pd.read_parquet(root / "stage3/output/traffic_state_batch1/train_reference.parquet")
    summaries, parts = [], []
    for date in cfg["forecast_train_dates"]:
        if not "20161009" <= date <= "20161024":
            raise ValueError("forecast source is outside frozen Train")
        day_root = root / f"stage1/input_v1/split=train/date={date}"
        base = pd.concat([pd.read_parquet(p, columns=["order_id", "departure_time"])
                          for p in sorted(day_root.glob("bucket=*/order_base.parquet"))], ignore_index=True)
        departure = _timestamp_series(base.departure_time)
        midnight = pd.Timestamp(date, tz="Asia/Shanghai")
        base["release_second"] = (departure - midnight).dt.total_seconds()
        mask = np.zeros(len(base), dtype=bool)
        for cut in cfg["cuts_s"]:
            mask |= base.release_second.between(cut - 900, cut + cfg["window_duration_s"] + cfg["patience_s"] + cfg["forecast_horizon_s"] + 900)
        base = base.loc[mask, ["order_id", "release_second"]].copy()
        base["order_id"] = base.order_id.astype(str)
        ids = list(base.order_id)
        manifest = root / f"stage0/work_v6_final/candidate_manifests/date={date}.parquet"
        coordinates = pd.read_parquet(manifest, columns=["order_id", "start_lon", "start_lat", "end_lon", "end_lat"],
                                       filters=[("order_id", "in", ids)])
        coordinates = add_coordinate_lineage(coordinates)
        cache = root / S3 / f"cache/train/date={date}"
        dynamic = pd.read_parquet(cache / "dynamic_descriptors.parquet", filters=[("order_id", "in", ids)])
        eligible = list(dynamic.order_id.astype(str))
        prediction = root / S3 / f"cache/m3/date={date}.parquet"
        tokens = pd.read_parquet(prediction, columns=["order_id", "traversal_id", "pred_pace_p50", "pred_crawl", "pred_stop", "travel_time_p50_s"],
                                 filters=[("order_id", "in", eligible)])
        route = pd.read_parquet(root / f"stage2/output_v4/route_conditioned_dataset/revealed_route_proxy/day={date}.parquet",
                                columns=["order_id", "traversal_id", EDGE], filters=[("order_id", "in", eligible)])
        tokens = tokens.merge(route, on=["order_id", "traversal_id"], validate="one_to_one")
        traffic = aggregate(decompose(tokens.merge(reference, on=EDGE, how="left", validate="many_to_one"), traffic_cfg))
        encounters = pd.read_parquet(cache / "encounters.parquet", filters=[("order_id", "in", eligible)])
        classified, _ = classifier.summarize(encounters)
        identity = pd.read_parquet(cache / "identity.parquet", columns=["order_id", "route_token_type"], filters=[("order_id", "in", eligible)])
        direction = identity.route_token_type.eq("FULL_NETWORK_EDGE").groupby(identity.order_id).all().to_dict()
        compatibility = route_compatibility(dynamic.order_id, classified, traffic, direction, cfg)
        library = base.merge(coordinates[["order_id", "start_lon_wgs84", "start_lat_wgs84", "end_lon_wgs84", "end_lat_wgs84"]],
                             on="order_id", validate="one_to_one").merge(
            dynamic[["order_id", "predicted_route_time_p50_s"]], on="order_id", validate="one_to_one").merge(
            compatibility, on="order_id", validate="one_to_one")
        library["date"] = date
        parts.append(library)
        summaries.append(dict(date=date, source_band_orders=len(base), template_orders=len(library),
                              m3_prediction_path=str(prediction), m3_prediction_sha256=sha(prediction),
                              source_midnight_is_before_test=True))
        print(json.dumps(dict(prepared_train=date, templates=len(library))), flush=True)
        del base, coordinates, dynamic, tokens, route, traffic, encounters, identity, compatibility, library
        gc.collect()
    library = pd.concat(parts, ignore_index=True)
    write_parquet(root / OUT / "train_request_templates.parquet", library)
    return summaries


def run(root):
    cfg = json.loads((root / CONFIG).read_text())
    (root / OUT).mkdir(parents=True, exist_ok=False)
    complexes, signals = signal_overlay(root, cfg)
    classifier = MovementClassifier(root, complexes)
    route_summary = test31_policy(root, classifier, cfg)
    sources = train_library(root, classifier, cfg)
    summary = dict(status="COMPLETE", config_sha256=sha(root / CONFIG), signal_overlay=signals,
                   test31_routes=route_summary, train_templates=sources,
                   new_clustering=False, new_m3_inference=False, http_calls=0,
                   test31_used_to_fit_forecast=False)
    write_json(root / OUT / "preparation_summary.json", summary)
    write_json(root / DOC / "input_connection_summary.json", summary)
    print(json.dumps(dict(status="COMPLETE", test31_routes=route_summary)), flush=True)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run(Path.cwd())
