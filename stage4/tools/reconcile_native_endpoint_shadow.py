"""Reconcile native endpoint identities to frozen membership, without clustering.

Only reciprocal native hierarchy transitions can supply physical-location aliases.
Even a unique complex anchor is a shadow hypothesis, not movement authorization.
"""
from collections import Counter
import json
from pathlib import Path
import time

import pandas as pd
import psutil

from stage4.tools.read_native_endpoint_candidates import NativeTiles, CONFIG
from stage4.tools.control_freedom_diagnostic import sha, write_json

PUBLIC = Path('stage4/docs/stage3_action_space_attribution/endpoint_shadow_reconciliation')
OUT = Path('stage4/output/paper_enhancement/endpoint_shadow_reconciliation')


def transition_component(native, start):
    """Exact IDs only. Do not equate arbitrary co-located graph nodes."""
    seen = {int(start)}; todo = [int(start)]
    while todo:
        current = todo.pop()
        for target in native.transitions(current):
            if current not in native.transitions(target):
                raise RuntimeError('Nonreciprocal hierarchy transition')
            if target not in seen:
                seen.add(target); todo.append(target)
                if len(seen) > 16:
                    raise RuntimeError('Unexpected hierarchy component size')
    return sorted(seen)


def classify_anchors(component, node_uid, membership):
    known = [n for n in component if n in node_uid]
    complexes = sorted({membership[node_uid[n]] for n in known if node_uid[n] in membership})
    if len(complexes) > 1:
        status = 'CONFLICT_MULTIPLE_FROZEN_COMPLEXES'
    elif complexes:
        status = 'UNIQUE_FROZEN_COMPLEX_ANCHOR'
    elif known:
        status = 'KNOWN_FROZEN_NODES_WITHOUT_COMPLEX'
    else:
        status = 'NO_FROZEN_NODE_ANCHOR'
    return status, known, complexes


def main():
    started = time.perf_counter()
    if (OUT/'summary.json').exists():
        raise RuntimeError('Preserve completed shadow audit')
    OUT.mkdir(parents=True, exist_ok=True); PUBLIC.mkdir(parents=True, exist_ok=True)
    base = Path('stage3/output/odd_tod')
    inputs = [base/'s2a/stage3_full_network_edges.parquet', base/'s2a/stage3_full_network_nodes.parquet',
              base/'s2b/final/stage3_intersection_node_membership.parquet',
              base/'s2b/final/stage3_route_movement_lookup.parquet',
              base/'s2b/final/stage3_edge_complex_boundary_index.parquet',
              Path('stage4/output/paper_enhancement/native_endpoint_candidates_v2/endpoint_candidates.parquet')]
    before = {str(p):sha(p) for p in inputs}
    candidates = pd.read_parquet(inputs[-1])
    nodes = pd.read_parquet(inputs[1], columns=['valhalla_node_id','stage3_node_uid'])
    member = pd.read_parquet(inputs[2])
    if member.stage3_node_uid.duplicated().any(): raise RuntimeError('Nonunique frozen membership')
    node_uid = dict(zip(nodes.valhalla_node_id.astype(int),nodes.stage3_node_uid))
    membership = dict(zip(member.stage3_node_uid,member.intersection_complex_uid))
    native = NativeTiles(json.loads(CONFIG.read_text())['mjolnir']['tile_dir'])
    rows = []; by_edge = {}
    for c in candidates.itertuples(index=False):
        edge = native.edge(c.directed_edge_graph_id)
        for side, nk in [('from','source'),('to','target')]:
            gid = edge[nk]; component = transition_component(native,gid)
            status, known, complexes = classify_anchors(component,node_uid,membership)
            old = getattr(c,'existing_'+side+'_node_id',None)
            old = int(old) if pd.notna(old) else None
            old_complex = membership.get(node_uid.get(old))
            row = dict(stage3_edge_uid=c.stage3_edge_uid,endpoint_side=side,native_node_id=gid,
                       was_missing_side=side==c.endpoint_side,old_node_id=old,
                       transition_component=json.dumps(component),frozen_node_anchors=json.dumps(known),
                       frozen_complex_anchors=json.dumps(complexes),status=status,
                       old_complex=old_complex,old_id_in_native_transition_component=old in component if old is not None else None,
                       old_complex_preserved=old_complex in complexes if old_complex is not None else None)
            rows.append(row); by_edge[(c.stage3_edge_uid,side)] = row
    frame = pd.DataFrame(rows)
    for col in ['native_node_id','old_node_id']: frame[col] = pd.array(frame[col],dtype='Int64')
    frame.to_csv(PUBLIC/'endpoint_shadow.csv',index=False)
    frame.to_parquet(OUT/'endpoint_shadow.parquet',index=False)
    # Original denominator is immutable: no rerun of parser that could erase rows.
    encounter_path = Path('stage4/output/paper_enhancement/stage3_semantic_provenance/movement_private.json')
    encounters = json.loads(encounter_path.read_text()); reconciliation = []
    for e in encounters:
        affected = [by_edge[(uid,side)] for uid in (e['incoming'],e['outgoing']) for side in ('from','to') if (uid,side) in by_edge]
        reasons = []
        if any(r['status']=='NO_FROZEN_NODE_ANCHOR' for r in affected): reasons.append('UNANCHORED_NATIVE_ENDPOINT')
        if any(r['status']=='CONFLICT_MULTIPLE_FROZEN_COMPLEXES' for r in affected): reasons.append('MULTIPLE_COMPLEX_ANCHORS')
        if any(r['old_id_in_native_transition_component'] is False for r in affected): reasons.append('OLD_IDENTITY_NOT_NATIVE_ALIAS')
        if any(r['old_complex_preserved'] is False for r in affected): reasons.append('OLD_COMPLEX_NOT_PRESERVED')
        if not e['recorded_chain_continuous']: reasons.append('RECORDED_CHAIN_DISCONTINUOUS')
        reconciliation.append(dict(order_id=e['order_id'],occurrence=e['occurrence'],complex_id=e['complex_id'],
                                   incoming=e['incoming'],outgoing=e['outgoing'],reasons=reasons,
                                   result='RETAIN_UNKNOWN',movement_repaired=False))
    write_json(OUT/'encounter_reconciliation_private.json',reconciliation)
    missing = frame[frame.was_missing_side]
    summary = dict(status='SHADOW_COMPLETE_NO_PRODUCTION_REPAIR',edges=len(candidates),endpoint_sides=len(frame),
                   missing_side_status_counts=dict(Counter(missing.status)),all_side_status_counts=dict(Counter(frame.status)),
                   original_encounters=len(encounters),unique_movement_keys=len({(e['complex_id'],e['incoming'],e['outgoing']) for e in encounters}),
                   retained_unknown=len(reconciliation),movement_repairs=0,
                   encounter_reason_counts=dict(Counter(r for e in reconciliation for r in e['reasons'])),
                   old_identity_not_native_alias=int(frame.old_id_in_native_transition_component.eq(False).sum()),
                   old_complex_not_preserved=int(frame.old_complex_preserved.eq(False).sum()),
                   input_hashes=before,encounter_source_sha256=sha(encounter_path),
                   source_files_unchanged=all(sha(Path(p))==h for p,h in before.items()),
                   native_tiles_unchanged=all(sha(t['path'])==t['sha256'] for t in native.tiles.values()),
                   clustering_changed=False,production_interface_changed=False,routing_calls=0,
                   runtime_s=time.perf_counter()-started,peak_working_set_mib=psutil.Process().memory_info().peak_wset/1024**2)
    write_json(OUT/'summary.json',summary);write_json(PUBLIC/'summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
