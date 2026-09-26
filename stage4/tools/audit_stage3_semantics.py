"""Frozen data provenance only. No routing, inference, dispatch or product fixes."""
import csv
import json
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import osmium
import pandas as pd
import psutil

from stage4.tools.control_freedom_diagnostic import sha, write_json
from stage4.tools.complete_attribution_pickup import PHASE0

OUT=Path('stage4/output/paper_enhancement/stage3_semantic_provenance')
PUBLIC=Path('stage4/docs/stage3_action_space_attribution/provenance_audit')


def main():
    began=time.perf_counter(); root=Path.cwd(); out=root/OUT; public=root/PUBLIC
    out.mkdir(parents=True,exist_ok=True); public.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists(): raise RuntimeError('Preserve completed audit')
    s=root/'stage3/output/odd_tod'; a=s/'s2a'; b=s/'s2b/final'
    source={}
    def read(path,**kwargs):
        source[str(path.relative_to(root))]=sha(path)
        return pd.read_parquet(path,**kwargs)
    peak=0
    def guard():
        nonlocal peak
        peak=max(peak,psutil.Process().memory_info().rss)
        if peak>2*1024**3 or time.perf_counter()-began>600: raise RuntimeError('RESOURCE_STOP')
    overlays=read(a/'stage3_historical_direction_overlay.parquet')
    edges=read(a/'stage3_full_network_edges.parquet',columns=['stage3_edge_uid','osm_way_id','from_osm_node_id','to_osm_node_id','from_stage3_node_uid','to_stage3_node_uid','valhalla_directed_edge_id'])
    edge={r.stage3_edge_uid:r for r in edges.itertuples(index=False)}
    pbf=Path('D:/pycodes/didi_xian_raw/map_data/xian_roadmap_update.osm.pbf')
    binding=json.loads((a/'osm_cache_binding.json').read_text())
    assert sha(pbf)==binding['pbf_sha256']
    ways={}; needed=set(overlays.observed_osm_way_id.dropna().astype(int))
    class Reader(osmium.SimpleHandler):
        def way(self,w):
            if w.id in needed:
                ways[w.id]={'tags':dict(w.tags),'nodes':[n.ref for n in w.nodes]}
    Reader().apply_file(str(pbf))
    reverse_tokens=read(s/'s4/test31_route_identity_resolution.parquet',filters=[('route_token_type','==','HISTORICAL_REVERSE_OVERLAY')],columns=['order_id','canonical_edge_uid','canonical_traversal_direction','route_sequence'])
    endpoint=defaultdict(list)
    for r in edges.itertuples(index=False):
        if pd.notna(r.from_osm_node_id) and pd.notna(r.to_osm_node_id): endpoint[int(r.osm_way_id),int(r.from_osm_node_id),int(r.to_osm_node_id)].append(r.stage3_edge_uid)
    classifications=[]
    for r in overlays.itertuples(index=False):
        guard()
        way=ways.get(int(r.observed_osm_way_id),{}); tags=way.get('tags',{}); nodes=way.get('nodes',[])
        start,end=r.observed_begin_osm_node_id,r.observed_end_osm_node_id
        oriented=False; direction=None
        if pd.notna(start) and pd.notna(end) and nodes.count(int(start))==1 and nodes.count(int(end))==1:
            oriented=True;direction=1 if nodes.index(int(end))>nodes.index(int(start)) else -1
        ow=tags.get('oneway','').lower(); vehicle_ow=tags.get('oneway:motor_vehicle',tags.get('oneway:motorcar',ow)).lower()
        allowed=1 if vehicle_ow in ('yes','1','true') else (-1 if vehicle_ow in ('-1','reverse') else None)
        implied=False
        if allowed is None and not vehicle_ow and (tags.get('junction')=='roundabout' or tags.get('highway')=='motorway'):
            allowed=1;implied=True
        restriction=oriented and allowed is not None and direction!=allowed
        conditional=any('conditional' in k for k in tags if k.startswith(('oneway','access','motor_vehicle','motorcar','vehicle')))
        exact=endpoint.get((int(r.observed_osm_way_id),int(start),int(end)),[]) if pd.notna(start) and pd.notna(end) else []
        if exact: category='EXPORTED_AUTO_DIRECTION_EXISTS_MAPPING_DISCREPANCY'
        elif restriction and not conditional: category='FROZEN_OSM_DIRECTION_SUPPORT'
        elif not oriented: category='UNRESOLVED_WAY_OR_ENDPOINT_ORIENTATION'
        else: category='OVERLAY_WITHOUT_EXPLICIT_DIRECTION_PROHIBITION'
        classifications.append(dict(canonical_edge_uid=r.canonical_edge_uid,category=category,osm_way_id=int(r.observed_osm_way_id),
            one_way=ow,vehicle_oneway=vehicle_ow,osm_orientation_resolved=oriented,implied_oneway=implied,conditional_tags_present=conditional,
            access_tags={k:v for k,v in tags.items() if k.startswith(('access','motor_vehicle','motorcar','vehicle'))},
            forward_reference_present=pd.notna(r.physical_forward_stage3_edge_uid),exported_same_endpoint_direction_count=len(exact),
            observed_valhalla_edge_id=int(r.observed_valhalla_edge_id),physical_restriction_confirmed=False))
    c=pd.DataFrame(classifications); joined=reverse_tokens.merge(c[['canonical_edge_uid','category']],on='canonical_edge_uid',validate='many_to_one')
    reverse_summary=[]
    for category,g in c.groupby('category'):
        t=joined[joined.category.eq(category)]
        reverse_summary.append(dict(issue=category,unique_overlay_identities=len(g),test31_tokens=len(t),test31_orders_overlapping=t.order_id.nunique(),
                                    source='frozen PBF way tags + observed directed endpoints + exported auto graph',real_world_legal_certification=False))
    cfrows=json.loads((root/PHASE0/'route_counterfactuals.json').read_text()); ids=sorted({r['order_id'] for r in cfrows})
    encounters=read(s/'s4/test31_route_complex_encounters.parquet',filters=[('order_id','in',ids)])
    bad=encounters[encounters.movement_lookup_status.ne('MATCHED_TOPOLOGICAL_MOVEMENT')].copy()
    members=read(b/'stage3_intersection_node_membership.parquet')
    member_sets={k:set(g.stage3_node_uid) for k,g in members.groupby('intersection_complex_uid')}
    movements=read(b/'stage3_route_movement_lookup.parquet',columns=['intersection_complex_uid','incoming_stage3_edge_uid','outgoing_stage3_edge_uid'])
    triples=set(map(tuple,movements.itertuples(index=False,name=None)))
    adjacency=defaultdict(list)
    for r in edges.itertuples(index=False): adjacency[r.from_stage3_node_uid].append(r.to_stage3_node_uid)
    def reachable(start,end,allowed):
        if start not in allowed or end not in allowed: return False
        q=deque([start]); seen={start}
        while q:
            u=q.popleft()
            if u==end:return True
            for v in adjacency.get(u,[]):
                if v in allowed and v not in seen: seen.add(v);q.append(v)
        return False
    movement_rows=[]
    for r in bad.itertuples(index=False):
        guard(); inc=edge.get(r.incoming_stage3_edge_uid); ext=edge.get(r.outgoing_stage3_edge_uid); ms=member_sets.get(r.intersection_complex_uid,set())
        pair=(r.intersection_complex_uid,r.incoming_stage3_edge_uid,r.outgoing_stage3_edge_uid)
        path=False; continuity=False; incoming_role=False;outgoing_role=False
        if inc is None or ext is None: reason='MISSING_EDGE_IDENTITY'
        elif not ms: reason='MISSING_COMPLEX'
        else:
            incoming_role=inc.from_stage3_node_uid not in ms and inc.to_stage3_node_uid in ms
            outgoing_role=ext.from_stage3_node_uid in ms and ext.to_stage3_node_uid not in ms
            path=reachable(inc.to_stage3_node_uid,ext.from_stage3_node_uid,ms)
            chain=[inc]+[edge.get(k) for k in json.loads(r.internal_stage3_edge_uids)]+[ext]
            continuity=all(x is not None and y is not None and x.to_stage3_node_uid==y.from_stage3_node_uid for x,y in zip(chain,chain[1:]))
            if pair in triples: reason='LOOKUP_STATUS_DISAGREES_WITH_FROZEN_KEY'
            elif not incoming_role or not outgoing_role: reason='BOUNDARY_ROLE_DISAGREEMENT'
            elif path: reason='FROZEN_TOPOLOGY_PATH_PRESENT_LOOKUP_OMISSION'
            else: reason='NO_DIRECTED_INTERNAL_PATH_IN_FROZEN_GRAPH'
        movement_rows.append(dict(order_id=r.order_id,occurrence=int(r.movement_occurrence_index),complex_id=r.intersection_complex_uid,
            incoming=r.incoming_stage3_edge_uid,outgoing=r.outgoing_stage3_edge_uid,reason=reason,internal_count=int(r.internal_edge_count),
            directed_path_exists=path,recorded_chain_continuous=continuity,incoming_role_valid=incoming_role,outgoing_role_valid=outgoing_role,
            recoverable_movement_from_frozen_topology=path and incoming_role and outgoing_role))
    m=pd.DataFrame(movement_rows); movement_summary=[]
    for reason,g in m.groupby('reason'):
        movement_summary.append(dict(issue=reason,encounters=len(g),unique_orders=g.order_id.nunique(),
            unique_movement_keys=len(g[['complex_id','incoming','outgoing']].drop_duplicates()),
            continuous_recorded_chain=int(g.recorded_chain_continuous.sum()),recoverable_from_frozen_topology=int(g.recoverable_movement_from_frozen_topology.sum())))
    unresolved=read(s/'s4/test31_route_identity_resolution.parquet',filters=[('order_id','in',ids),('route_token_type','==','UNRESOLVED')])
    mapping=read(a/'stage3_observed_full_network_mapping.parquet')
    u=unresolved.merge(mapping[['canonical_edge_uid','canonical_traversal_direction','mapping_status']],on=['canonical_edge_uid','canonical_traversal_direction'],how='left')
    identity_summary=[dict(issue=str(k),tokens=len(g),unique_orders=g.order_id.nunique()) for k,g in u.groupby('mapping_status',dropna=False)]
    # Keep raw identifiers and human-review samples local; public aggregate example IDs only.
    write_json(out/'reverse_identity_private.json',classifications)
    write_json(out/'movement_private.json',movement_rows)
    samples=[]
    for label,rows in [('reverse',classifications),('movement',movement_rows)]:
        groups=defaultdict(list)
        for row in rows:groups[row.get('category',row.get('reason'))].append(row)
        for issue,items in sorted(groups.items()):
            # deterministic provenance representatives, not a claim of manual validation
            for i,row in enumerate(items[:25]):samples.append(dict(case_id=f'{label}_{len(samples)+1:04d}',issue=issue,manual_review_status='NOT_REVIEWED',detail=row))
    write_json(out/'manual_review_samples_private.json',samples)
    outputs=dict(reverse_classification=reverse_summary,movement_classification=movement_summary,identity_classification=identity_summary)
    for name,rows in outputs.items():
        write_json(public/(name+'.json'),rows)
        if rows:
            with (public/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(status='PASS',reverse_order_count=reverse_tokens.order_id.nunique(),reverse_tokens=len(reverse_tokens),
        reverse_identities_in_test31=reverse_tokens.canonical_edge_uid.nunique(),all_overlay_identities=len(c),
        scoped_orders=len(ids),encounters=len(encounters),unresolved_encounters=len(bad),unresolved_movement_orders=bad.order_id.nunique(),
        raw_identity_unresolved_tokens=len(unresolved),raw_identity_unresolved_orders=unresolved.order_id.nunique(),
        runtime_s=time.perf_counter()-began,peak_sampled_rss_mib=peak/1024**2,route_calls=0,MILPs=0,pbf_sha256=binding['pbf_sha256'],
        source_sha256=source,manual_samples=len(samples),manual_review_completed=False)
    write_json(out/'summary.json',summary);write_json(public/'summary.json',summary)
    print(json.dumps(dict(summary=summary,classifications=outputs),indent=2))


if __name__=='__main__':main()
