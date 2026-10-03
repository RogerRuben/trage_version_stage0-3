"""Read-only, version-locked native GraphId endpoint evidence; never a graph patch.

Layout: official Valhalla 3.8.2 baldr/{nodeinfo,nodetransition,directededge,
graphtileheader,edgeinfo}.h and src/baldr/graphtile.cc. Native header parsing,
complete outgoing ownership, reciprocal opposing edges, and GraphUtils shapes
cross-check the narrow little-endian x64 reader. Coordinates never select IDs.
"""
import json
from pathlib import Path
import struct
import sys
import time

import numpy as np
import pandas as pd
import psutil
from valhalla.baldr import GraphId, GraphTileHeader
from valhalla.baldr.utils.graph_utils import GraphUtils

from stage4.tools.control_freedom_diagnostic import sha, write_json

OUT = Path('stage4/output/paper_enhancement/native_endpoint_candidates_v2')
PUBLIC = Path('stage4/docs/stage3_action_space_attribution/native_endpoint_candidates_v2')
CONFIG = Path('D:/pycodes/didi_xian_raw/valhalla_data/valhalla.json')
MASK46 = (1 << 46) - 1


class NativeTiles:
    """Bounded local read-only tile cache. No actor/routing API is instantiated."""
    def __init__(self, directory):
        if sys.byteorder != 'little' or struct.calcsize('P') != 8:
            raise RuntimeError('Only pinned little-endian x64 ABI supported')
        self.directory = Path(directory)
        self.paths = {}
        self.tiles = {}
        for path in sorted(self.directory.rglob('*.gph')):
            h = GraphTileHeader.from_file(str(path))
            if h.version != '3.8.2' or h.byte_size() != 272:
                raise RuntimeError('Unverified tile version/layout')
            key = int(h.graphid.value)
            if key in self.paths:
                raise RuntimeError('Duplicate native tile ID')
            self.paths[key] = path

    def tile(self, gid):
        g = GraphId(int(gid))
        key = int(GraphId(g.tileid(), g.level(), 0).value)
        if key not in self.tiles:
            path = self.paths[key]
            data = path.read_bytes()
            h = GraphTileHeader.from_bytes(data[:272])
            if len(data) != h.end_offset or struct.unpack_from('<I', data, 224)[0] != h.end_offset:
                raise RuntimeError('Header size cross-check failed')
            node_end = 272 + 32 * h.nodecount
            edge_start = node_end + 8 * h.transitioncount
            edge_end = edge_start + 48 * h.directededgecount
            edgeinfo = struct.unpack_from('<I', data, 104)[0]
            textlist = struct.unpack_from('<I', data, 108)[0]
            if not edge_end <= edgeinfo <= textlist <= len(data):
                raise RuntimeError('Invalid record region boundaries')
            nodes = np.frombuffer(data, dtype='<u8', count=4*h.nodecount, offset=272).reshape(-1, 4)
            edges = np.frombuffer(data, dtype='<u8', count=6*h.directededgecount, offset=edge_start).reshape(-1, 6)
            owner = np.full(h.directededgecount, -1, dtype=np.int32)
            for i, node in enumerate(nodes):
                word = int(node[1]); start = word & ((1 << 21)-1); count = (word >> 21) & 127
                if start + count > len(owner) or np.any(owner[start:start+count] != -1):
                    raise RuntimeError('Non-unique/out-of-bounds native edge ownership')
                owner[start:start+count] = i
            if np.any(owner < 0):
                raise RuntimeError('Unowned native directed edge')
            self.tiles[key] = dict(path=path, sha256=sha(path), data=data, header=h,
                                   nodes=nodes, edges=edges, owner=owner,
                                   edge_start=edge_start, edgeinfo=edgeinfo, textlist=textlist)
            if sum(len(t['data']) for t in self.tiles.values()) > 128*1024**2:
                raise RuntimeError('Native tile memory budget exceeded')
        return self.tiles[key], g

    def node(self, gid):
        t, g = self.tile(gid)
        if g.id() >= len(t['nodes']):
            raise RuntimeError('Native endpoint node index out of bounds')
        n = t['nodes'][g.id()]; w = int(n[0]); w1 = int(n[1])
        base_lon, base_lat = t['header'].base_ll
        lon = base_lon + ((w >> 26) & ((1 << 22)-1))*1e-6 + ((w >> 48)&15)*1e-7
        lat = base_lat + (w & ((1 << 22)-1))*1e-6 + ((w >> 22)&15)*1e-7
        return dict(lon=lon, lat=lat, first_edge=w1 & ((1 << 21)-1), count=(w1 >> 21)&127,
                    tile=t, graph_id=g)

    def edge(self, gid):
        t, g = self.tile(gid)
        if g.id() >= len(t['edges']):
            raise RuntimeError('Native directed edge index out of bounds')
        words = [int(x) for x in t['edges'][g.id()]]
        source = int(GraphId(g.tileid(), g.level(), int(t['owner'][g.id()])).value)
        target = words[0] & MASK46
        a = self.node(source); b = self.node(target)
        opp_index = (words[0] >> 54) & 127
        if opp_index >= b['count']:
            raise RuntimeError('Opposing index outside target outgoing range')
        opp_local = b['first_edge'] + opp_index
        opp = [int(x) for x in b['tile']['edges'][opp_local]]
        if (opp[0] & MASK46) != source or ((opp[0] >> 54)&127) != g.id()-a['first_edge']:
            raise RuntimeError('Opposing edge is not reciprocal')
        offset = t['edgeinfo'] + (words[1] & ((1 << 25)-1))
        if not t['edgeinfo'] <= offset <= t['textlist']-12:
            raise RuntimeError('EdgeInfo offset out of bounds')
        low, mid, high = struct.unpack_from('<III', t['data'], offset)
        if (high >> 28)&3:
            raise RuntimeError('Extended >48-bit OSM way ID not supported by narrow reader')
        way_id = low | ((mid >> 24) << 32) | (((high >> 20)&255) << 40)
        return dict(source=source, target=target, source_node=a, target_node=b, way_id=way_id,
                    forward=bool((words[0] >> 61)&1), access=words[3]&4095,
                    road_class=(words[2] >> 54)&7, length=(words[4] >> 32)&((1 << 24)-1),
                    source_tile=str(t['path']), source_record=t['edge_start']+48*g.id(),
                    opposing_edge_id=int(GraphId(b['graph_id'].tileid(), b['graph_id'].level(), opp_local).value))

    def transitions(self, gid):
        t, g = self.tile(gid)
        self.node(gid)  # bounds check
        word = int(t['nodes'][g.id()][2])
        start = word & ((1 << 21)-1); count = (word >> 21)&7
        if start+count > t['header'].transitioncount:
            raise RuntimeError('Node transition range out of bounds')
        return [struct.unpack_from('<Q',t['data'],272+32*t['header'].nodecount+8*(start+j))[0]&MASK46
                for j in range(count)]


def main():
    started = time.perf_counter()
    if (OUT/'summary.json').exists():
        raise RuntimeError('Preserve completed candidate evidence; choose a new version explicitly')
    OUT.mkdir(parents=True, exist_ok=True); PUBLIC.mkdir(parents=True, exist_ok=True)
    missing_path = Path('stage4/output/paper_enhancement/stage3_semantic_provenance/endpoint_evidence_private.json')
    missing = json.loads(missing_path.read_text())
    source = Path('stage3/output/odd_tod/s2a/stage3_full_network_edges.parquet')
    node_source = source.with_name('stage3_full_network_nodes.parquet')
    frozen = {str(p): sha(p) for p in (source, node_source, CONFIG, missing_path)}
    edges = pd.read_parquet(source, filters=[('stage3_edge_uid', 'in', [r['edge_uid'] for r in missing])]).set_index('stage3_edge_uid')
    known = pd.read_parquet(node_source, columns=['valhalla_node_id', 'stage3_node_uid'])
    known_ids = set(known.valhalla_node_id.astype(int))
    native = NativeTiles(json.loads(CONFIG.read_text())['mjolnir']['tile_dir'])
    graph = GraphUtils(CONFIG)
    rows = []
    for item in missing:
        e = edges.loc[item['edge_uid']]; gid = int(e.valhalla_directed_edge_id)
        row = dict(stage3_edge_uid=item['edge_uid'], directed_edge_graph_id=gid, endpoint_side=item['endpoint'], old_node_id=None,
                   native_identity_verified=False, existing_endpoint_conflict=False)
        errors = []
        try:
            r = native.edge(gid)
            row.update({k:r[k] for k in ('source_tile','source_record','opposing_edge_id','way_id','forward','access')})
            row.update(native_from_node_id=r['source'], native_to_node_id=r['target'])
            side = 'source' if item['endpoint']=='from' else 'target'
            row['candidate_node_id'] = r[side]
            row['candidate_in_frozen_node_table'] = r[side] in known_ids
            if r['way_id'] != int(e.osm_way_id): errors.append('OSM_WAY_ID_CONFLICT')
            # Export F/R is relative to the MVT feature, NOT EdgeInfo's stored way
            # direction. A marker mismatch is not a direction mismatch. Verify the
            # already-oriented geometry against the native source/target instead.
            row['mvt_direction_marker'] = e.direction
            row['mvt_marker_agrees_native_forward'] = r['forward'] == (e.direction=='F')
            geometry_error = max(abs(e.start_lon-r['source_node']['lon']),abs(e.start_lat-r['source_node']['lat']),
                                 abs(e.end_lon-r['target_node']['lon']),abs(e.end_lat-r['target_node']['lat']))
            row['oriented_export_endpoint_error_deg'] = geometry_error
            # z12/extent4096 export has ~2 m quantization. This is only a
            # consistency veto; it never supplies or chooses any GraphId.
            if geometry_error > 2e-5: errors.append('ORIENTED_EXPORT_GEOMETRY_CONFLICT')
            if r['access'] != int(e.valhalla_access_mask): errors.append('ACCESS_CONFLICT')
            if r['road_class'] != int(e.valhalla_road_class_code): errors.append('ROAD_CLASS_CONFLICT')
            if r['length'] != int(e.length_m): errors.append('LENGTH_CONFLICT')
            for endpoint, nk in (('from','source'),('to','target')):
                old = e[endpoint+'_valhalla_node_id']
                if pd.notna(old):
                    row['existing_'+endpoint+'_node_id'] = int(old)
                if pd.notna(old) and int(old) != r[nk]:
                    row['existing_endpoint_conflict'] = True
                    row['existing_conflict_side'] = endpoint
                    row['existing_conflict_is_explicit_hierarchy_transition'] = r[nk] in native.transitions(int(old))
                    row['existing_conflict_same_level'] = GraphId(int(old)).level() == GraphId(r[nk]).level()
                    errors.append('EXISTING_'+endpoint.upper()+'_IDENTITY_CONFLICT')
            shape = graph.get_edge_shape(GraphId(gid))
            # Native shape can be stored in way order; direction selects orientation.
            expected = [(r['source_node']['lon'], r['source_node']['lat']), (r['target_node']['lon'], r['target_node']['lat'])]
            direct = max(abs(float(shape[i][j])-expected[k][j]) for i,k in ((0,0),(-1,1)) for j in (0,1))
            reverse = max(abs(float(shape[i][j])-expected[k][j]) for i,k in ((-1,0),(0,1)) for j in (0,1))
            row['native_shape_endpoint_error_deg'] = min(direct,reverse)
            if min(direct,reverse) > 2e-6: errors.append('NATIVE_SHAPE_ENDPOINT_CONFLICT')
            row['native_ownership_and_opposing_verified'] = True
            row['native_identity_verified'] = not any(not x.startswith('EXISTING_') for x in errors)
        except (RuntimeError, KeyError, IndexError, ValueError) as ex:
            errors.append(type(ex).__name__+': '+str(ex))
        row['conflict_reason'] = '|'.join(errors)
        row['evidence_status'] = 'VERIFIED_NATIVE_ENDPOINT' if not errors else 'STOP_CONFLICT_OR_UNRESOLVED'
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_parquet(OUT/'endpoint_candidates.parquet', index=False)
    # Static network identities only; no orders, vehicles, or trajectory coordinates.
    frame.to_csv(PUBLIC/'endpoint_candidates.csv', index=False)
    unchanged = all(sha(Path(p))==h for p,h in frozen.items()) and all(sha(t['path'])==t['sha256'] for t in native.tiles.values())
    verified = int((frame.evidence_status=='VERIFIED_NATIVE_ENDPOINT').sum())
    summary = dict(status='PASS_NATIVE_IDENTITIES' if verified==len(rows) and unchanged else 'STOP_VERIFICATION_FAILED',
                   endpoints=len(rows), verified=verified, conflicts_or_unresolved=len(rows)-verified,
                   native_identities_read_and_cross_checked=int(frame.native_identity_verified.sum()),
                   existing_endpoint_conflicts=int(frame.existing_endpoint_conflict.sum()),
                   conflicts_with_explicit_hierarchy_transition=int(frame.get('existing_conflict_is_explicit_hierarchy_transition',pd.Series(dtype=bool)).fillna(False).sum()),
                   native_missing_side_unique_nodes=int(frame.candidate_node_id.nunique()),
                   mvt_marker_disagreement_not_a_direction_failure=int((~frame.mvt_marker_agrees_native_forward).sum()),
                   candidates_already_in_frozen_node_table=int(frame.get('candidate_in_frozen_node_table',pd.Series(dtype=bool)).sum()),
                   input_hashes=frozen, native_tiles=[dict(path=str(t['path']),sha256=t['sha256'],nodes=t['header'].nodecount,edges=t['header'].directededgecount) for t in native.tiles.values()],
                   sources_unchanged=unchanged, production_modified=False, routing_calls=0, matrix_calls=0,
                   runtime_s=time.perf_counter()-started, peak_working_set_mib=psutil.Process().memory_info().peak_wset/1024**2)
    write_json(OUT/'summary.json',summary);write_json(PUBLIC/'summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
