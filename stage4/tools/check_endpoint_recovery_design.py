"""Read-only local MVT fragment/endpoint evidence for the 88 audited endpoints.

Grid coincidence is a candidate, not a certified graph endpoint repair.
"""
import json
import math
from collections import defaultdict,Counter
from pathlib import Path
import time
import pandas as pd
import psutil
import valhalla

from stage3.odd_tod.mvt import decode_tile
from stage3.odd_tod.network_foundation import _edge_attributes,graph_id
from stage4.tools.control_freedom_diagnostic import sha,write_json

OUT=Path('stage4/output/paper_enhancement/endpoint_repair_design')
PUBLIC=Path('stage4/docs/stage3_action_space_attribution/endpoint_repair_design')


def main():
    root=Path.cwd(); out=root/OUT; public=root/PUBLIC
    out.mkdir(parents=True,exist_ok=True);public.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise RuntimeError('Preserve existing design verification')
    started=time.perf_counter();peak=0
    missing=json.loads((root/'stage4/output/paper_enhancement/stage3_semantic_provenance/endpoint_evidence_private.json').read_text())
    source=root/'stage3/output/odd_tod/s2a/stage3_full_network_edges.parquet';before=sha(source)
    edges=pd.read_parquet(source,filters=[('stage3_edge_uid','in',list({r['edge_uid'] for r in missing}))])
    targets=set(edges.valhalla_directed_edge_id.astype(int));byuid=edges.set_index('stage3_edge_uid')
    z=12;n=2**z
    def web(lon,lat):return ((lon+180)/360*n,(1-math.asinh(math.tan(math.radians(lat)))/math.pi)/2*n)
    tiles=set()
    for r in edges.itertuples(index=False):
        for lon,lat in json.loads(r.geometry):
            x,y=web(lon,lat)
            for dx in (-1,0,1):
                for dy in (-1,0,1):tiles.add((int(x)+dx,int(y)+dy))
    assert len(tiles)<=100,'Bounded tile inspection limit'
    cfgpath=Path('D:/pycodes/didi_xian_raw/valhalla_data/valhalla.json')
    cfg=json.loads(cfgpath.read_text());cfg['loki']['service_defaults']['mvt_min_zoom_road_class']=[z]*8
    actor=valhalla.Actor(cfg);fragments=defaultdict(set);nodegrid=defaultdict(set);levels={};evidence=[]
    for i,(x,y) in enumerate(sorted(tiles)):
        peak=max(peak,psutil.Process().memory_info().rss)
        if peak>2*1024**3 or time.perf_counter()-started>600:raise RuntimeError('Resource limit')
        tile=decode_tile(actor.tile(dict(tile=dict(z=z,x=x,y=y),filters=dict(action='include',attributes=_edge_attributes()),generalize=0)))
        for node in tile.get('nodes',[]):
            extent=node['extent'];p=node['geometry'];assert extent==4096
            nodegrid[(x*extent+p[0],y*extent+p[1],node['properties']['tile_level'])].add(int(node['id']))
        for f in tile.get('edges',[]):
            p=f['properties'];extent=f['extent'];coords=f['geometry']
            if not coords or not isinstance(coords[0],tuple):continue
            for suffix in ('fwd','bwd'):
                if p.get('edge_id:'+suffix) is None:continue
                eid=graph_id(int(f['id']),int(p['edge_id:'+suffix]))
                if eid not in targets:continue
                points=tuple((x*extent+c[0],y*extent+c[1]) for c in coords)
                if suffix=='bwd':points=tuple(reversed(points))
                fragments[eid].add(points);levels[eid]=p['tile_level']
        if i%10==0:print('tiles',i+1,'/',len(tiles),flush=True)
    for r in missing:
        e=byuid.loc[r['edge_uid']];eid=int(e.valhalla_directed_edge_id);fs=fragments[eid]
        # Unique directed atomic segments; branch/disconnect is retained, never repaired by proximity.
        segments={(a,b) for f in fs for a,b in zip(f,f[1:]) if a!=b}
        incoming=Counter(b for a,b in segments);outgoing=Counter(a for a,b in segments)
        starts=set(outgoing)-set(incoming);ends=set(incoming)-set(outgoing)
        nonlinear=any(v>1 for v in incoming.values()) or any(v>1 for v in outgoing.values())
        terminals=starts if r['endpoint']=='from' else ends
        candidates=set().union(*(nodegrid.get((*p,levels.get(eid)),set()) for p in terminals)) if terminals else set()
        lon=e.start_lon if r['endpoint']=='from' else e.end_lon
        lat=e.start_lat if r['endpoint']=='from' else e.end_lat
        gx,gy=web(lon,lat);grid=(round(gx*4096),round(gy*4096))
        onborder=grid[0]%4096==0 or grid[1]%4096==0
        linear=bool(segments) and len(starts)==1 and len(ends)==1 and not nonlinear
        evidence.append(dict(edge_uid=r['edge_uid'],endpoint=r['endpoint'],graph_id=eid,fragment_count=len(fs),
            stored_endpoint_on_web_tile_boundary=onborder,linear_fragment_graph=linear,
            start_count=len(starts),end_count=len(ends),candidate_node_count=len(candidates),candidate_node_ids=sorted(candidates),
            status='UNIQUE_GRID_CANDIDATE_NEEDS_GRAPH_ENDPOINT_CERTIFICATION' if linear and len(candidates)==1 else 'UNRESOLVED_FRAGMENT_OR_NODE_AMBIGUITY',
            certified_repair=False))
    write_json(out/'endpoint_candidates_private.json',evidence)
    summary=dict(status='DESIGN_EVIDENCE_COMPLETE_NOT_A_REPAIR',endpoints=len(evidence),edges=len(edges),local_mvt_tiles=len(tiles),
        stored_endpoints_on_tile_boundary=sum(r['stored_endpoint_on_web_tile_boundary'] for r in evidence),
        unique_grid_candidates=sum(r['status'].startswith('UNIQUE') for r in evidence),
        fragment_or_node_unresolved=sum(not r['status'].startswith('UNIQUE') for r in evidence),
        certified_repairs=0,counts_by_fragment_count=dict(Counter(r['fragment_count'] for r in evidence)),
        runtime_s=time.perf_counter()-started,peak_sampled_rss_mib=peak/1024**2,
        routing_calls=0,matrix_calls=0,production_modified=False,source_edges_sha256=before,valhalla_config_sha256=sha(cfgpath))
    assert sha(source)==before
    write_json(out/'summary.json',summary);write_json(public/'summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
