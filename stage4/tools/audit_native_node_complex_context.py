"""55-node read-only topology context, never membership assignment or clustering."""
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import psutil
from pyproj import Transformer
from scipy.spatial import cKDTree
from valhalla.baldr import GraphId

from stage4.tools.read_native_endpoint_candidates import NativeTiles, CONFIG
from stage4.tools.reconcile_native_endpoint_shadow import transition_component, classify_anchors
from stage4.tools.control_freedom_diagnostic import sha, write_json

OUT = Path('stage4/output/paper_enhancement/native_node_complex_context_final')
PUBLIC = Path('stage4/docs/stage3_action_space_attribution/native_node_complex_context_final')


def context_class(distance_m, complexes):
    # Distance is measured to frozen candidate members, NOT a node identity rule.
    if distance_m <= 10: return 'WITHIN_FROZEN_CANDIDATE_BUFFER_REVIEW'
    if distance_m <= 20: return 'WITHIN_CANDIDATE_PAIR_DISTANCE_REVIEW'
    if len(complexes) > 1: return 'OUTSIDE_BUFFERS_MULTICOMPLEX_ADJACENCY'
    if len(complexes) == 1: return 'OUTSIDE_BUFFERS_SINGLE_COMPLEX_ADJACENCY'
    return 'OUTSIDE_BUFFERS_NO_DIRECT_COMPLEX_ANCHOR'


def main():
    started = time.perf_counter()
    if (OUT/'summary.json').exists(): raise RuntimeError('Preserve completed audit')
    OUT.mkdir(parents=True, exist_ok=True); PUBLIC.mkdir(parents=True, exist_ok=True)
    root = Path('stage3/output/odd_tod')
    paths = [root/'s2a/stage3_full_network_nodes.parquet',root/'s2b/final/stage3_intersection_node_membership.parquet',
             root/'s2b/final/stage3_intersection_complexes.parquet',root/'s2b/final/stage3_route_movement_lookup.parquet',
             root/'s2a/stage3_full_network_edges.parquet',root/'s2b/final/stage3_edge_complex_boundary_index.parquet',
             Path('stage4/output/paper_enhancement/native_endpoint_candidates_v2/endpoint_candidates.parquet')]
    before = {str(p):sha(p) for p in paths}
    nodes = pd.read_parquet(paths[0],columns=['valhalla_node_id','stage3_node_uid'])
    members = pd.read_parquet(paths[1]); candidates = pd.read_parquet(paths[-1])
    targets = sorted(set(candidates.candidate_node_id.astype(int)))
    if len(targets) != 55: raise RuntimeError('55-node scope changed')
    node_uid = dict(zip(nodes.valhalla_node_id.astype(int),nodes.stage3_node_uid))
    uid_node = {v:k for k,v in node_uid.items()}
    membership = dict(zip(members.stage3_node_uid,members.intersection_complex_uid))
    native = NativeTiles(json.loads(CONFIG.read_text())['mjolnir']['tile_dir'])
    transform = Transformer.from_crs('EPSG:4326','EPSG:32649',always_xy=True)
    # One sparse spatial index. Native coordinates of EXISTING frozen IDs only;
    # no new candidates, query_pairs, union-find, or membership reconstruction.
    cm = members[members.candidate_node].reset_index(drop=True)
    xy = []
    for uid in cm.stage3_node_uid:
        n = native.node(uid_node[uid]); xy.append(transform.transform(n['lon'],n['lat']))
    tree = cKDTree(np.asarray(xy))
    aliases = {}
    def alias(gid):
        if gid not in aliases:
            component = transition_component(native,gid)
            for x in component: aliases[x] = component
        return aliases[gid]
    rows = []; adjacency = []
    for gid in targets:
        component = alias(gid); n = native.node(gid)
        point = transform.transform(n['lon'],n['lat'])
        distance, index = tree.query(point)
        within10 = tree.query_ball_point(point,10); within20 = tree.query_ball_point(point,20)
        complexes10 = sorted(set(cm.iloc[within10].intersection_complex_uid))
        complexes20 = sorted(set(cm.iloc[within20].intersection_complex_uid))
        outgoing = set(); incoming = set(); adjacent_complexes = set(); neighbors = set(); ordinary_neighbors = set()
        seen_edges = set(); signal = False; roundabout = False; per_level = {}; ordinary_complexes = set()
        for source in component:
            sn = native.node(source); sg = GraphId(source)
            word = int(sn['tile']['nodes'][sg.id()][1]); signal |= bool((word >> 61)&1)
            level_neighbors = set()
            for local in range(sn['first_edge'],sn['first_edge']+sn['count']):
                eid = int(GraphId(sg.tileid(),sg.level(),local).value)
                if eid in seen_edges: continue
                seen_edges.add(eid); e = native.edge(eid); opp = native.edge(e['opposing_edge_id'])
                if not (e['access']&1 or opp['access']&1): continue
                ng = e['target']; group = alias(ng); physical_key = min(group)
                if ng in component: continue
                level_neighbors.add(physical_key); neighbors.add(physical_key)
                if e['access']&1: outgoing.add(physical_key)
                if opp['access']&1: incoming.add(physical_key)
                status, _, cs = classify_anchors(group,node_uid,membership)
                adjacent_complexes.update(cs)
                t,g = native.tile(eid); words = t['edges'][g.id()]
                is_roundabout = bool((int(words[2]) >> 61)&1); roundabout |= is_roundabout
                is_shortcut = bool((int(words[5]) >> 60)&1)
                if not is_shortcut:
                    ordinary_neighbors.add(physical_key); ordinary_complexes.update(cs)
                adjacency.append(dict(native_node_id=gid,source_alias=source,directed_edge_graph_id=eid,
                    native_neighbor_id=ng,neighbor_transition_component=json.dumps(group),
                    length_m=e['length'],outgoing_auto=bool(e['access']&1),incoming_auto=bool(opp['access']&1),
                    neighbor_anchor_status=status,neighbor_complexes=json.dumps(cs),roundabout_native=is_roundabout,
                    is_native_shortcut=is_shortcut))
            per_level[str(sg.level())] = len(level_neighbors)
        rows.append(dict(native_node_id=gid,transition_component=json.dumps(component),
            affected_edge_count=int((candidates.candidate_node_id==gid).sum()),
            nearest_frozen_candidate_distance_m=float(distance),nearest_candidate_complex=str(cm.iloc[int(index)].intersection_complex_uid),
            frozen_complexes_with_candidate_within10m=json.dumps(complexes10),
            frozen_complexes_with_candidate_within20m=json.dumps(complexes20),
            directly_adjacent_complexes=json.dumps(sorted(adjacent_complexes)),
            nonshortcut_adjacent_complexes=json.dumps(sorted(ordinary_complexes)),
            native_neighbor_count=len(neighbors),incoming_neighbor_count=len(incoming),outgoing_neighbor_count=len(outgoing),
            nonshortcut_native_neighbor_count=len(ordinary_neighbors),
            native_neighbor_count_by_level=json.dumps(per_level,sort_keys=True),native_signal=signal,native_roundabout_edge=roundabout,
            branching_or_control_evidence=len(ordinary_neighbors)>=3 or signal or roundabout,
            topology_context=context_class(float(distance),ordinary_complexes),
            membership_assignment='NOT_ASSIGNED',production_eligibility='UNCHANGED'))
    frame = pd.DataFrame(rows); frame.to_csv(PUBLIC/'node_context.csv',index=False)
    pd.DataFrame(adjacency).to_csv(PUBLIC/'native_adjacency.csv',index=False)
    frame.to_parquet(OUT/'node_context.parquet',index=False)
    summary = dict(status='READ_ONLY_TOPOLOGY_CONTEXT_COMPLETE',node_count=len(rows),affected_edges=len(candidates),
        unique_transition_components=len(set(frame.transition_component)),
        unique_component_context_counts=dict(Counter(frame.drop_duplicates('transition_component').topology_context)),
        unique_components_branching_or_control=int(frame.drop_duplicates('transition_component').branching_or_control_evidence.sum()),
        unique_components_signal=int(frame.drop_duplicates('transition_component').native_signal.sum()),
        shortcut_adjacency_records=sum(r['is_native_shortcut'] for r in adjacency),
        nonshortcut_neighbor_degree_counts=dict(Counter(frame.nonshortcut_native_neighbor_count)),
        nodes_near_multiple_complexes_within10m=sum(len(json.loads(x))>1 for x in frame.frozen_complexes_with_candidate_within10m),
        nodes_near_multiple_complexes_within20m=sum(len(json.loads(x))>1 for x in frame.frozen_complexes_with_candidate_within20m),
        context_counts=dict(Counter(frame.topology_context)),
        nearest_candidate_distance_min_m=float(frame.nearest_frozen_candidate_distance_m.min()),
        nearest_candidate_distance_median_m=float(frame.nearest_frozen_candidate_distance_m.median()),
        nearest_candidate_distance_max_m=float(frame.nearest_frozen_candidate_distance_m.max()),
        branching_or_control_nodes=int(frame.branching_or_control_evidence.sum()),
        signal_nodes=int(frame.native_signal.sum()),roundabout_nodes=int(frame.native_roundabout_edge.sum()),
        direct_adjacency_records=len(adjacency),nodes_with_no_membership_assignment=len(rows),
        input_hashes=before,sources_unchanged=all(sha(Path(p))==h for p,h in before.items()),
        native_tiles_unchanged=all(sha(t['path'])==t['sha256'] for t in native.tiles.values()),
        clustering_changed=False,production_changed=False,routing_calls=0,matrix_calls=0,
        runtime_s=time.perf_counter()-started,peak_working_set_mib=psutil.Process().memory_info().peak_wset/1024**2)
    write_json(OUT/'summary.json',summary);write_json(PUBLIC/'summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
