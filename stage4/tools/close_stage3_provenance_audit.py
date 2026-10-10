"""Refine preliminary internal-path checks using frozen endpoint identity evidence."""
import json
from collections import Counter
from pathlib import Path
import pandas as pd
import osmium
from stage4.tools.audit_stage3_semantics import OUT,PUBLIC
from stage4.tools.control_freedom_diagnostic import write_json,sha


def main():
    root=Path.cwd();out=root/OUT;public=root/PUBLIC;s=root/'stage3/output/odd_tod'
    m=pd.DataFrame(json.loads((out/'movement_private.json').read_text()))
    edges=pd.read_parquet(s/'s2a/stage3_full_network_edges.parquet').set_index('stage3_edge_uid')
    nodes=pd.read_parquet(s/'s2a/stage3_full_network_nodes.parquet')
    osm=nodes.dropna(subset=['osm_node_id']).groupby('osm_node_id').stage3_node_uid.agg(list).to_dict()
    vh=nodes.groupby('valhalla_node_id').stage3_node_uid.agg(list).to_dict()
    missing={};events=[]
    for r in m.itertuples(index=False):
        fields=[]
        for uid in (r.incoming,r.outgoing):
            e=edges.loc[uid]
            for prefix,osmfield in [('from','from_osm_node_id'),('to','to_osm_node_id')]:
                if pd.notna(e[prefix+'_stage3_node_uid']):continue
                identity=(uid,prefix)
                a=vh.get(e[prefix+'_valhalla_node_id'],[]);b=osm.get(e[osmfield],[])
                if len(a)==1:status='EXACT_FROZEN_VALHALLA_NODE_AVAILABLE'
                elif len(b)==1:status='UNIQUE_OSM_NODE_CANDIDATE_NOT_DIRECTION_CERTIFIED'
                elif len(b)>1:status='AMBIGUOUS_OSM_NODE_CANDIDATES'
                else:status='NODE_ABSENT_FROM_FROZEN_NODE_TABLE'
                missing[identity]=dict(edge_uid=uid,endpoint=prefix,status=status,osm_node_id=int(e[osmfield]) if pd.notna(e[osmfield]) else None,
                    valhalla_node_id=int(e[prefix+'_valhalla_node_id']) if pd.notna(e[prefix+'_valhalla_node_id']) else None,
                    exact_node_count=len(a),osm_candidate_count=len(b))
                fields.append(status)
        assert fields,'Unexpected complete endpoint pair'
        events.append(dict(order_id=r.order_id,complex_id=r.complex_id,incoming=r.incoming,outgoing=r.outgoing,
            source='MOVEMENT_ENDPOINT_COMPLETE_VS_BOUNDARY_ALL_EDGES',missing_endpoint_statuses=fields,
            internally_connected=r.directed_path_exists,recorded_chain_continuous=r.recorded_chain_continuous,
            exact_recoverable_all_endpoints=all(x=='EXACT_FROZEN_VALHALLA_NODE_AVAILABLE' for x in fields)))
    counts=Counter(x['status'] for x in missing.values())
    authoritative=[dict(issue='ENDPOINT_SCOPE_MISMATCH_BETWEEN_MOVEMENT_AND_BOUNDARY_PRODUCTS',encounters=len(events),unique_orders=m.order_id.nunique(),
        unique_movement_keys=len(m[['complex_id','incoming','outgoing']].drop_duplicates()),internal_path_present=int(m.directed_path_exists.sum()),
        entire_movement_certified=False,exact_endpoint_recovery_encounters=sum(e['exact_recoverable_all_endpoints'] for e in events),
        scientific_consequence='Technical interface unknown; preserve UNKNOWN until endpoint and movement evidence is repaired and verified')]
    write_json(public/'movement_classification.json',authoritative)
    import csv
    with (public/'movement_classification.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(authoritative[0]));writer.writeheader();writer.writerows(authoritative)
    write_json(out/'endpoint_evidence_private.json',list(missing.values()))
    write_json(out/'movement_closure_private.json',events)
    summaries=[dict(issue=k,unique_edge_endpoints=v) for k,v in sorted(counts.items())]
    write_json(public/'endpoint_recoverability.json',summaries)
    needed={v['osm_node_id'] for v in missing.values() if v['osm_node_id'] is not None}
    pbf_nodes=set()
    class NodeReader(osmium.SimpleHandler):
        def node(self,n):
            if n.id in needed:pbf_nodes.add(n.id)
    NodeReader().apply_file('D:/pycodes/didi_xian_raw/map_data/xian_roadmap_update.osm.pbf')
    # Raw export cache, unlike final nodes, precedes endpoint-enrichment filtering.
    raw_nodes=pd.read_parquet(s/'s2a/full_network_export_cache_nodes.parquet')
    raw_ids=set(raw_nodes.valhalla_node_id)
    endpoint_raw=dict(unique_osm_node_ids=len(needed),osm_node_ids_in_frozen_pbf=len(needed&pbf_nodes),
        edge_endpoints_without_valhalla_node_id=sum(v['valhalla_node_id'] is None for v in missing.values()),
        edge_endpoints_with_id_in_raw_node_export=sum(v['valhalla_node_id'] in raw_ids for v in missing.values()),
        interpretation='OSM node existence is not an exact directed Valhalla endpoint mapping; no nearest-node repair performed')
    write_json(public/'endpoint_raw_evidence.json',endpoint_raw)
    cfrows=json.loads((root/'stage4/output/paper_enhancement/stage3_attribution_phase0/route_counterfactuals.json').read_text())
    variants={(r['order_id'],r['condition']):r for r in cfrows}
    branch=Counter()
    for (oid,d),v in variants.items():
        if d!='D3' or not v['av_eligible']:continue
        x=variants[oid,'D1']
        if x['hard_state']!='FEASIBLE' or x['selected_route_type']=='NONE':
            branch[tuple(x.get('remaining_unknown',[]))]+=1
    write_json(public/'branching_summary.json',[dict(D1_original_unknown_reasons=list(k),D3_structurally_available_unique_orders=v) for k,v in branch.items()])
    # Public case sheet uses frozen map identities only, never customer/order IDs.
    examples=[]
    for r in json.loads((out/'manual_review_samples_private.json').read_text()):
        d=r['detail']; safe={k:v for k,v in d.items() if k not in ('order_id','occurrence')}
        if r['case_id'].startswith('movement'):
            safe['classification']='ENDPOINT_SCOPE_MISMATCH'
            safe['reason']='ENDPOINT_SCOPE_MISMATCH_BETWEEN_MOVEMENT_AND_BOUNDARY_PRODUCTS'
            safe.pop('recoverable_movement_from_frozen_topology',None)
            safe['entire_movement_certified']=False
        examples.append(dict(case_id=r['case_id'],status='PROGRAMMATIC_EVIDENCE_ONLY_HUMAN_REVIEW_PENDING',detail=safe))
    write_json(public/'representative_cases.json',examples)
    # Reconcile all graph-source hashes and manually expose audit limitations.
    summary=json.loads((out/'summary.json').read_text())
    for path,digest in summary['source_sha256'].items():assert sha(root/path)==digest
    reverse=json.loads((out/'reverse_identity_private.json').read_text())
    evidence=Counter((r['category'],r['one_way'],r['implied_oneway'],r['conditional_tags_present']) for r in reverse)
    write_json(public/'reverse_tag_evidence.json',[dict(category=k[0],oneway=k[1],implied_oneway=k[2],conditional=k[3],identities=v) for k,v in sorted(evidence.items())])
    qa=dict(status='PASS',all_391_have_incomplete_stage3_endpoint=True,unique_missing_edge_endpoints=len(missing),
        exact_endpoint_recovery_encounters=sum(e['exact_recoverable_all_endpoints'] for e in events),
        source_hashes_unchanged=True,initial_internal_path_result_superseded_as_full_recoverability_claim=True,
        movement_product_changed=False,Stage3_modified=False,routing_calls=0,MILPs=0,
        warning='Internal reachability alone does not establish external endpoint identity, legal movement or route readiness')
    write_json(public/'qa.json',qa)
    summary['movement_classification_stage']='ENDPOINT_AWARE_CLOSURE_COMPLETE'
    summary['authoritative_movement_table']='movement_classification.json'
    summary['closure_code_sha256']=sha(Path(__file__))
    write_json(public/'summary.json',summary)
    ident=json.loads((public/'identity_classification.json').read_text())
    for r in ident:
        if r['issue']=='nan':r['issue']='NULL_SOURCE_AND_CANONICAL_IDENTITY'
    write_json(public/'identity_classification.json',ident)
    with (public/'identity_classification.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(ident[0]));w.writeheader();w.writerows(ident)
    print(json.dumps(dict(endpoint_summary=summaries,qa=qa,reverse_tags=[dict(k=str(k),n=v) for k,v in evidence.items()]),indent=2))


if __name__=='__main__':main()
