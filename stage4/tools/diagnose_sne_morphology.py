"""Train-complex-only, CPU-only morphology diagnostic; no dispatch imports.

All design constants below are fixed before looking at SNE outputs. Grid details
stay in ignored output; docs contain aggregates and a bounded map QA pack.
"""
from __future__ import annotations

import gc
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import psutil
from pyproj import Transformer
from shapely.geometry import LineString

ROOT = Path(__file__).resolve().parents[2]
BASE = Path("stage3/output/odd_tod")
OUT = ROOT / "stage4/output/sne_morphology"
DOC = ROOT / "stage4/docs/static_exposure_diagnostic/sne"
SOURCES = {
    "edges": BASE / "s2a/stage3_full_network_edges.parquet",
    "nodes": BASE / "s2a/stage3_full_network_nodes.parquet",
    "members": BASE / "s2b/final/stage3_intersection_node_membership.parquet",
    "complexes": BASE / "s2b/final/stage3_intersection_complexes.parquet",
    "train": BASE / "s3/train_static_complex_reference.parquet",
    "profile": Path("stage3/config/stage3_av_capability_profiles.json"),
    "crs_config": Path("stage3/config/stage3_s2b_intersection_complex.json"),
}
SETTINGS = [(s, p) for s in (500, 1000, 2000) for p in (0., .5)]
METRICS = ("author_norm", "axial_norm")


def check_memory():
    rss = psutil.Process().memory_info().rss
    if rss > 2 * 1024**3:
        raise MemoryError("2 GiB soft budget reached; no automatic budget expansion")


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def dist(x):
    x = pd.Series(x).dropna()
    return {"n": len(x), **({k: float(x.quantile(q)) for k, q in
        [("min", 0), ("p25", .25), ("p50", .5), ("p75", .75), ("p90", .9), ("max", 1)]} if len(x) else {})}


def corr(a, b):
    z = pd.DataFrame({"a": np.asarray(a), "b": np.asarray(b)}).dropna()
    r = z.a.rank().corr(z.b.rank()) if len(z) > 1 and z.a.nunique() > 1 and z.b.nunique() > 1 else np.nan
    return {"n": len(z), "spearman": float(r) if np.isfinite(r) else None}


def entropy(hist):
    a = np.asarray(hist, dtype=float)
    total = a.sum(axis=-1)
    p = a / np.where(total > 0, total, 1)[..., None]
    logp = np.zeros_like(p)
    np.log(p, out=logp, where=p > 0)
    h = -(p * logp).sum(axis=-1)
    return np.where(total > 0, h, np.nan)


def bearing(delta):
    return np.degrees(np.arctan2(delta[..., 0], delta[..., 1])) % 360


def bins(angles, axial=False):
    # Exactly the author's roll-one + pair bin assignment, with 180 folding
    # only for our separately named axial quantity.
    return ((np.floor(np.asarray(angles) / 5).astype(int) + 1) % 72) // 2 % (18 if axial else 36)


def key_geometry(coords):
    a = np.asarray(coords, dtype='<f8')
    return min(a.tobytes(), a[::-1].tobytes())


def clip_segments(start, end, size, phase):
    """Split only grid-crossing straight segments, never edge x grid expansion."""
    delta = end - start
    lengths = np.linalg.norm(delta, axis=1)
    ok = lengths > 0
    start, end, delta, lengths = start[ok], end[ok], delta[ok], lengths[ok]
    original = np.flatnonzero(ok)
    shift = size * phase
    c0 = np.floor((start - shift) / size).astype(np.int64)
    c1 = np.floor((end - shift) / size).astype(np.int64)
    same = (c0 == c1).all(axis=1)
    keys, weights, ids = [c0[same]], [lengths[same]], [original[same]]
    extra_k, extra_w, extra_i = [], [], []
    for i in np.flatnonzero(~same):
        ts = [0., 1.]
        for axis in (0, 1):
            if delta[i, axis] == 0:
                continue
            lo, hi = sorted((start[i, axis], end[i, axis]))
            for grid in range(math.floor((lo - shift) / size) + 1,
                              math.ceil((hi - shift) / size)):
                t = (grid * size + shift - start[i, axis]) / delta[i, axis]
                if 0 < t < 1:
                    ts.append(t)
        ts = np.unique(ts)
        for a, b in zip(ts[:-1], ts[1:]):
            point = start[i] + (a + b) / 2 * delta[i]
            extra_k.append(np.floor((point - shift) / size).astype(np.int64))
            extra_w.append((b - a) * lengths[i])
            extra_i.append(original[i])
    if extra_k:
        keys.append(np.asarray(extra_k)); weights.append(np.asarray(extra_w)); ids.append(np.asarray(extra_i))
    return np.concatenate(keys), np.concatenate(weights), np.concatenate(ids)


def synthetic_checks():
    empty = float(entropy(np.zeros(18)))
    assert math.isnan(empty)
    hist = np.bincount(bins([0, 90], True), weights=[100, 100], minlength=18)
    assert np.isclose(entropy(hist), math.log(2))
    split = np.bincount(bins([0, 0, 90], True), weights=[30, 70, 100], minlength=18)
    assert np.allclose(hist, split)
    line = np.array([[0., 0.], [0., 100.]])
    assert key_geometry(line) == key_geometry(line[::-1])
    author1 = entropy(np.bincount(bins([0, 90]), minlength=36))
    author2 = entropy(np.bincount(bins([0, 0, 90]), minlength=36))
    start = np.array([[0., 0.], [0., 600.], [-60., -40.]])
    end = np.array([[0., 600.], [600., 600.], [650., 810.]])
    for s, p in SETTINGS:
        _, length, _ = clip_segments(start, end, s, p)
        assert np.isclose(length.sum(), np.linalg.norm(end - start, axis=1).sum(), atol=1e-8)
    rotation = [float(entropy(np.bincount(bins((np.array([4., 6., 90.]) + r) % 360, True), minlength=18)))
                for r in (0, 3, 7, 15)]
    return {"status": "PASS", "author_split_H_before_after": [float(author1), float(author2)],
            "axial_split_and_reverse_key_invariant": True, "cross_grid_length_conservation": True,
            "orthogonal_H": math.log(2), "rotation_offsets_deg": [0, 3, 7, 15], "rotation_H": rotation}


def prepare():
    config = json.loads((ROOT / SOURCES['crs_config']).read_text())
    assert config['metric_crs'] == 'EPSG:32649'
    transform = Transformer.from_crs('EPSG:4326', 'EPSG:32649', always_xy=True)
    df = pd.read_parquet(ROOT / SOURCES['edges'], columns=[
        'geometry', 'osm_way_id', 'bridge_effective', 'tunnel_effective', 'valhalla_layer',
        'valhalla_road_class', 'from_stage3_node_uid', 'to_stage3_node_uid'])
    class_names = sorted(df.valhalla_road_class.fillna('UNKNOWN').unique())
    codes = {c: i for i, c in enumerate(class_names)}
    mid, angles, allclass, shapes, uniqueclass = [], [], [], [], []
    seen, endpoint_keys = {}, {}
    duplicates = invalid = degenerate_chord = same_direction = endpoint_geometry_disagreement = 0
    bounds = [float('inf'), float('inf'), -float('inf'), -float('inf')]
    for i, r in enumerate(df.itertuples(index=False)):
        coords = np.asarray(json.loads(r.geometry), dtype=float)
        if len(coords) < 2 or not np.isfinite(coords).all():
            invalid += 1; continue
        x, y = transform.transform(coords[:, 0], coords[:, 1])
        points = np.column_stack([x, y])
        if np.linalg.norm(np.diff(points, axis=0), axis=1).sum() == 0:
            invalid += 1; continue
        bounds = [min(bounds[0], x.min()), min(bounds[1], y.min()), max(bounds[2], x.max()), max(bounds[3], y.max())]
        # The author interpolates the midpoint in WGS84, not along UTM length.
        midpoint = LineString(coords).interpolate(.5, normalized=True)
        mid.append(transform.transform(midpoint.x, midpoint.y))
        chord = points[-1] - points[0]
        degenerate_chord += int(np.linalg.norm(chord) == 0)
        angles.append(float(bearing(chord)))  # match atan2(0,0)=0 in the author formula
        rc = codes[r.valhalla_road_class if pd.notna(r.valhalla_road_class) else 'UNKNOWN']
        allclass.append(rc)
        identity = (str(r.osm_way_id), str(r.bridge_effective), str(r.tunnel_effective), str(r.valhalla_layer))
        signature = identity + (key_geometry(coords),)
        endpoints = identity + tuple(sorted([str(r.from_stage3_node_uid), str(r.to_stage3_node_uid)]))
        if endpoints in endpoint_keys and endpoint_keys[endpoints] != signature[-1]:
            endpoint_geometry_disagreement += 1
        endpoint_keys.setdefault(endpoints, signature[-1])
        if signature in seen:
            duplicates += 1
            same_direction += int(seen[signature] == coords.tobytes())
        else:
            seen[signature] = coords.tobytes()
            shapes.append(points); uniqueclass.append(rc)
        if i % 10000 == 0:
            check_memory()
    info = {"directed_edge_rows": len(df), "valid_author_edges": len(mid), "invalid_geometry_edges": invalid,
            "zero_endpoint_chord_edges_author_kept": degenerate_chord,
            "unique_geometry_identity_rows": len(shapes), "exact_duplicate_rows_removed": duplicates,
            "same_direction_duplicate_rows": same_direction,
            "same_identity_endpoint_different_geometry_rows_retained": endpoint_geometry_disagreement,
            "dedup_identity": "osm_way_id + bridge + tunnel + layer + exact coordinates up to reversal; no rounding",
            "bounds_utm": [float(v) for v in bounds], "class_names": class_names}
    del seen, endpoint_keys, df
    nodes = pd.read_parquet(ROOT / SOURCES['nodes'], columns=['stage3_node_uid', 'lon', 'lat'])
    assert nodes.stage3_node_uid.is_unique
    nodes['x'], nodes['y'] = transform.transform(nodes.lon.to_numpy(), nodes.lat.to_numpy())
    members = pd.read_parquet(ROOT / SOURCES['members'], columns=['intersection_complex_uid', 'stage3_node_uid'])
    assert not members.duplicated().any()
    members = members.merge(nodes[['stage3_node_uid', 'x', 'y']], on='stage3_node_uid', how='left', validate='many_to_one')
    assert not members[['x', 'y']].isna().any().any()
    centers = members.groupby('intersection_complex_uid')[['x', 'y']].mean()
    assert len(centers) == 43685
    train = pd.read_parquet(ROOT / SOURCES['train'], columns=['intersection_complex_uid', 'A_c', 'M_c', 'D_c', 'L_c'])
    assert len(train) == 2425 and train.intersection_complex_uid.is_unique
    centers = centers.join(train.set_index('intersection_complex_uid'))
    del nodes
    gc.collect()
    starts = np.concatenate([p[:-1] for p in shapes])
    ends = np.concatenate([p[1:] for p in shapes])
    geomid = np.repeat(np.arange(len(shapes)), [len(p) - 1 for p in shapes])
    classes = np.asarray(uniqueclass)[geomid]
    info['local_segments'] = len(starts)
    return np.asarray(mid), np.asarray(angles), np.asarray(allclass), shapes, starts, ends, geomid, classes, centers, members, info


def calculate(size, phase, mid, angles, starts, ends, geomid, classes, centers, members, info):
    keys, length, seg = clip_segments(starts, ends, size, phase)
    expected = np.linalg.norm(ends - starts, axis=1).sum()
    assert abs(length.sum() - expected) < max(1e-6, expected * 1e-12)
    akeys = np.floor((mid - size * phase) / size).astype(np.int64)
    combined = np.concatenate([keys, akeys])
    unique, inverse = np.unique(combined, axis=0, return_inverse=True)
    axidx, auidx = inverse[:len(keys)], inverse[len(keys):]
    axial = np.zeros((len(unique), 18)); author = np.zeros((len(unique), 36))
    np.add.at(axial, (axidx, bins(bearing(ends[seg] - starts[seg]), True)), length)
    np.add.at(author, (auidx, bins(angles)), 1)
    unique_support = pd.DataFrame({'cell': axidx, 'geometry': geomid[seg]}).drop_duplicates().groupby('cell').size()
    classhist = np.zeros((len(unique), len(info['class_names'])))
    np.add.at(classhist, (axidx, classes[seg]), length)
    grid = pd.DataFrame(unique, columns=['gx', 'gy'])
    grid['author_count'] = author.sum(axis=1)
    grid['author_norm'] = entropy(author) / math.log(36)
    grid.loc[grid.author_count < 2, 'author_norm'] = np.nan
    grid['axial_norm'] = entropy(axial) / math.log(18)
    grid['length_m'] = axial.sum(axis=1)
    grid['geometry_count'] = unique_support.reindex(range(len(grid)), fill_value=0).to_numpy()
    for j, name in enumerate(info['class_names']):
        grid['road_length_share_' + name] = classhist[:, j] / np.where(grid.length_m > 0, grid.length_m, 1)
    grid['dominant_road_class'] = np.asarray(info['class_names'])[classhist.argmax(axis=1)]
    grid.loc[grid.length_m == 0, 'dominant_road_class'] = 'UNKNOWN'
    shift = size * phase
    xmin, ymin, xmax, ymax = info['bounds_utm']
    grid['network_bbox_edge_proxy'] = ((grid.gx * size + shift < xmin) | (grid.gy * size + shift < ymin)
        | ((grid.gx + 1) * size + shift > xmax) | ((grid.gy + 1) * size + shift > ymax))
    # This proxy is not certification of tile completeness; no unknown -> zero fill.
    c = centers.copy()
    c[['gx', 'gy']] = np.floor((c[['x', 'y']] - shift) / size).astype(np.int64)
    c = c.reset_index().merge(grid, on=['gx', 'gy'], how='left', validate='many_to_one').set_index('intersection_complex_uid')
    mkeys = np.floor((members[['x', 'y']] - shift) / size).astype(np.int64)
    mg = members[['intersection_complex_uid']].copy()
    mg[['gx', 'gy']] = mkeys
    crossing = mg.drop_duplicates().groupby('intersection_complex_uid').size().gt(1)
    c['members_cross_grid'] = crossing.reindex(c.index)
    tag = f'{size}m_p{phase:g}'
    grid.to_parquet(OUT / f'grid_{tag}.parquet', index=False)
    c.to_parquet(OUT / f'complex_{tag}.parquet')
    t = c[c.A_c.notna()].copy()
    summary = {'setting': tag, 'cell_count': len(grid), 'length_conservation_error_m': float(length.sum() - expected),
               'train_complex_count': len(t), 'full_network_complex_count': len(c),
               'full_network_support': {m: int(c[m].notna().sum()) for m in METRICS},
               'train_unique_mapped_cells': len(t[['gx', 'gy']].drop_duplicates()),
               'train_cross_grid_member_count': int(t.members_cross_grid.sum()),
               'train_bbox_edge_proxy_count': int(t.network_bbox_edge_proxy.fillna(True).sum()),
               'train_support': {m: int(t[m].notna().sum()) for m in METRICS},
               'train_distribution': {m: dist(t[m]) for m in METRICS},
               'method_correlation': corr(t.author_norm, t.axial_norm),
               'descriptor_correlations': {m: {d: corr(t[m], t[d + '_c']) for d in 'AMDL'} for m in METRICS},
               'density_correlations': {m: corr(t[m], t.length_m) for m in METRICS}}
    # Fixed support/density strata, not selected against output relationships.
    t['support_stratum'] = pd.cut(t.geometry_count, [-1, 1, 9, 49, np.inf], labels=['0-1', '2-9', '10-49', '50+'])
    t['length_stratum'] = pd.qcut(t.length_m, 4, duplicates='drop').astype(str)
    strata = []
    for label in ['support_stratum', 'length_stratum', 'dominant_road_class', 'network_bbox_edge_proxy']:
        for value, sub in t.groupby(label, observed=True):
            strata.append({'variable': label, 'value': str(value), 'n': len(sub),
                           'distribution': {m: dist(sub[m]) for m in METRICS},
                           'descriptor_correlations': {m: {d: corr(sub[m], sub[d + '_c']) for d in 'AMDL'} for m in METRICS}})
    summary['strata'] = strata
    return t, summary


def pairs_and_figures(primary, shapes):
    # Same A/M/D, same 10m L band AND absolute L difference <=10m; adjacency
    # after sorting on L then stable UID gives bounded, entropy-blind pairs.
    p = primary.dropna(subset=list(METRICS)).copy()
    p['L_band'] = np.floor(p.L_c / 10).astype(int)
    rows = []
    for _, g in p.reset_index().groupby(['A_c', 'M_c', 'D_c', 'L_band'], sort=True):
        g = g.sort_values(['L_c', 'intersection_complex_uid'])
        for i in range(0, len(g) - 1, 2):
            a, b = g.iloc[i], g.iloc[i + 1]
            if abs(a.L_c - b.L_c) > 10: continue
            rows.append({'a': a.intersection_complex_uid, 'b': b.intersection_complex_uid,
                         'axial_abs_diff': abs(a.axial_norm - b.axial_norm),
                         'author_abs_diff': abs(a.author_norm - b.author_norm),
                         'same_grid': bool(a.gx == b.gx and a.gy == b.gy)})
    pairs = pd.DataFrame(rows)
    pairs.to_parquet(OUT / 'conditional_pairs.parquet', index=False)
    ordered = pairs.sort_values(['axial_abs_diff', 'a', 'b'])
    selected = pd.concat([ordered.head(6).assign(selection='agreement'), ordered.tail(6).iloc[::-1].assign(selection='divergence')]).drop_duplicates(['a', 'b'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from shapely import STRtree
    from shapely.geometry import box
    geoms = [LineString(s) for s in shapes]
    tree = STRtree(geoms)
    qa = []
    for number, (_, pair) in enumerate(selected.iterrows(), 1):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
        for ax, key in zip(axes, ['a', 'b']):
            row = primary.loc[pair[key]]
            left, bottom = row.gx * 1000, row.gy * 1000
            idx = tree.query(box(left, bottom, left + 1000, bottom + 1000))
            ax.add_collection(LineCollection([shapes[i] for i in idx], colors='#597889', linewidths=.7))
            ax.scatter([row.x], [row.y], c='#c94b40', s=26, zorder=3)
            ax.set(xlim=(left, left + 1000), ylim=(bottom, bottom + 1000), aspect='equal')
            ax.tick_params(labelbottom=False, labelleft=False)
            ax.set_title(f'A/M/D={row.A_c:g}/{row.M_c:g}/{row.D_c:g}, L={row.L_c:.1f} m\n'
                         f'author={row.author_norm:.3f}; axial={row.axial_norm:.3f}\n'
                         f'road length={row.length_m/1000:.1f} km; geometries={row.geometry_count:g}', fontsize=9)
        fig.suptitle(f'Pair {number:02d} | {pair.selection} | 1 km frozen-network context\n'
                     'Red: complex center; same-scale panels; morphology only, not AV capability', fontsize=11)
        file = f'pair_{number:02d}.png'
        fig.savefig(DOC / file, dpi=130); plt.close(fig)
        qa.append({'pair': number, 'selection': pair.selection, 'file': file,
                   'axial_abs_diff': float(pair.axial_abs_diff), 'author_abs_diff': float(pair.author_abs_diff),
                   'same_grid': bool(pair.same_grid)})
        check_memory()
    return {'pair_count': len(pairs), 'same_grid_pairs': int(pairs.same_grid.sum()),
            'axial_absolute_difference': dist(pairs.axial_abs_diff),
            'author_absolute_difference': dist(pairs.author_abs_diff),
            'different_grid_axial_difference': dist(pairs.loc[~pairs.same_grid, 'axial_abs_diff']), 'qa': qa}


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True); DOC.mkdir(parents=True, exist_ok=True)
    source_hashes = {k: sha(ROOT / p) for k, p in SOURCES.items()}
    checks = synthetic_checks()
    prepared = prepare()
    mid, angles, allclass, shapes, starts, ends, geomid, classes, centers, members, info = prepared
    print(json.dumps(info), flush=True)
    tables, results = {}, []
    for size, phase in SETTINGS:
        check_memory()
        table, summary = calculate(size, phase, mid, angles, starts, ends, geomid, classes, centers, members, info)
        tag = summary['setting']; tables[tag] = table; results.append(summary)
        print(tag, summary['train_support'], 'RSS MiB', round(psutil.Process().memory_info().rss/2**20), flush=True)
        gc.collect()
    sensitivity = []
    for a, b in itertools.combinations(tables, 2):
        for m in METRICS:
            t = pd.concat([tables[a][m], tables[b][m]], axis=1, keys=['a', 'b']).dropna()
            sensitivity.append({'a': a, 'b': b, 'metric': m, **corr(t.a, t.b),
                                'absolute_difference': dist((t.a - t.b).abs())})
    pairs = pairs_and_figures(tables['1000m_p0'], shapes)
    assert source_hashes == {k: sha(ROOT / p) for k, p in SOURCES.items()}
    mem = psutil.Process().memory_info()
    payload = {'status': 'FIRST_STAGE_MORPHOLOGY_DIAGNOSTIC_COMPLETE',
               'code_base': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               'sources': {k: {'path': p.as_posix(), 'sha256': source_hashes[k]} for k, p in SOURCES.items()},
               'crs': 'EPSG:32649', 'origin_m': [0, 0], 'settings': SETTINGS,
               'boundary_certification': 'UNKNOWN: network bbox edge is only a proxy, not exact tile/PBF coverage',
               'network': info, 'synthetic_checks': checks, 'settings_results': results,
               'cross_setting_sensitivity': sensitivity, 'conditional_pairs': pairs,
               'inputs_unchanged': True, 'runtime_s': time.perf_counter() - started,
               'peak_working_set_mib': getattr(mem, 'peak_wset', mem.rss)/2**20}
    write_json(DOC / 'summary.json', payload)
    print('COMPLETE', payload['runtime_s'], payload['peak_working_set_mib'], flush=True)


if __name__ == '__main__':
    main()
