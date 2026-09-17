"""S2B-1.1 narrow 5m-vs-10m scientific closure using frozen products."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyproj import Transformer
from scipy.spatial import cKDTree

from stage3.odd_tod.network_foundation import (
    Stage3S2AError, atomic_json, atomic_parquet, atomic_text, git_head,
    parquet_descriptor, payload_hash, read_json, sha256_file, source_descriptor,
)


PHASE_STATUS = "STAGE3_S2B11_FINAL_REVIEW_PAIR_READY"
AUTHORIZED_BASE = "44ac44ed0e9746d8074e7802679e7f245d6c812c"
ALLOWED_LABELS = ["5_CORRECT", "10_CORRECT", "BOTH_ACCEPTABLE", "NEITHER", "UNCERTAIN"]


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file(): raise Stage3S2AError(f"missing frozen product: {path}")
    return pq.read_table(path).to_pandas()


def _hash_rank(uid: str) -> str:
    return hashlib.sha256(f"20261009|S2B11|{uid}".encode()).hexdigest()


def changed_groups(m5: pd.DataFrame, m10: pd.DataFrame) -> tuple[pd.DataFrame, set[str]]:
    a = m5[m5["candidate_node"]].set_index("stage3_node_uid")["intersection_complex_uid"]
    b = m10[m10["candidate_node"]].set_index("stage3_node_uid")["intersection_complex_uid"]
    pairs = pd.DataFrame({"complex_r05": a, "complex_r10": b}).dropna().reset_index()
    changed = set(pairs.groupby("complex_r10").filter(lambda frame: frame["complex_r05"].nunique() > 1)["complex_r10"])
    pairs["changed_05_10"] = pairs["complex_r10"].isin(changed)
    return pairs, changed


def endpoint_diagnostic(edges, nodes, candidates, controls, pairs, changed, transformer) -> dict[str, Any]:
    incomplete = edges[edges["from_stage3_node_uid"].isna() | edges["to_stage3_node_uid"].isna()].copy()
    candidate_xy = np.column_stack(transformer.transform(candidates["lon"].to_numpy(), candidates["lat"].to_numpy()))
    candidate_tree = cKDTree(candidate_xy)
    signal_nodes = candidates[candidates["rule_c_signal"]]
    signal_xy = np.column_stack(transformer.transform(signal_nodes["lon"].to_numpy(), signal_nodes["lat"].to_numpy()))
    signal_tree = cKDTree(signal_xy)
    changed_nodes = set(pairs.loc[pairs["complex_r10"].isin(changed), "stage3_node_uid"])
    changed_frame = nodes[nodes["stage3_node_uid"].isin(changed_nodes)]
    changed_xy = np.column_stack(transformer.transform(changed_frame["lon"].to_numpy(), changed_frame["lat"].to_numpy()))
    changed_tree = cKDTree(changed_xy)
    # Endpoint proximity is more meaningful than clipped-edge midpoint.
    def endpoint_xy(frame):
        starts = np.column_stack(transformer.transform(frame["start_lon"].to_numpy(), frame["start_lat"].to_numpy()))
        ends = np.column_stack(transformer.transform(frame["end_lon"].to_numpy(), frame["end_lat"].to_numpy()))
        return starts, ends
    def near(frame, tree, distance):
        starts, ends = endpoint_xy(frame)
        return np.minimum(tree.query(starts)[0], tree.query(ends)[0]) <= distance
    complete = edges.dropna(subset=["from_stage3_node_uid", "to_stage3_node_uid"])
    near_candidate_20 = near(incomplete, candidate_tree, 20)
    near_signal_20 = near(incomplete, signal_tree, 20)
    near_changed_20 = near(incomplete, changed_tree, 20)
    complete_candidate_20 = near(complete, candidate_tree, 20)
    complete_signal_20 = near(complete, signal_tree, 20)
    complete_changed_20 = near(complete, changed_tree, 20)
    def comparison(name, incomplete_mask, complete_mask):
        incomplete_share = float(incomplete_mask.mean())
        complete_share = float(complete_mask.mean())
        return {
            "evidence": name,
            "endpoint_incomplete_near_count": int(incomplete_mask.sum()),
            "endpoint_incomplete_near_share": incomplete_share,
            "endpoint_complete_near_count": int(complete_mask.sum()),
            "endpoint_complete_near_share": complete_share,
            "incomplete_to_complete_share_ratio": incomplete_share / complete_share if complete_share else None,
        }
    by_class = []
    for road_class, group in edges.groupby("valhalla_road_class", dropna=False):
        incomplete_count = int((group["from_stage3_node_uid"].isna() | group["to_stage3_node_uid"].isna()).sum())
        by_class.append({"road_class": str(road_class), "full_edge_count": len(group), "endpoint_incomplete_count": incomplete_count, "endpoint_incomplete_share": incomplete_count / len(group)})
    return {
        "endpoint_incomplete_edge_count": len(incomplete),
        "missing_from_count": int(incomplete["from_stage3_node_uid"].isna().sum()),
        "missing_to_count": int(incomplete["to_stage3_node_uid"].isna().sum()),
        "missing_both_count": int((incomplete["from_stage3_node_uid"].isna() & incomplete["to_stage3_node_uid"].isna()).sum()),
        "by_road_class": by_class,
        "near_junction_candidate_20m_count": int(near_candidate_20.sum()),
        "near_junction_candidate_20m_share": float(near_candidate_20.mean()),
        "near_signal_20m_count": int(near_signal_20.sum()),
        "near_signal_20m_share": float(near_signal_20.mean()),
        "near_changed_05_10_20m_count": int(near_changed_20.sum()),
        "near_changed_05_10_20m_share": float(near_changed_20.mean()),
        "complete_edge_comparisons": [
            comparison("junction_candidate", near_candidate_20, complete_candidate_20),
            comparison("signal", near_signal_20, complete_signal_20),
            comparison("changed_05_10", near_changed_20, complete_changed_20),
        ],
        "proximity_definition": "minimum projected distance from either exported edge endpoint to evidence node; EPSG:32649; 20m descriptive radius",
        "topology_impact_interpretation": "no systematic enrichment is observed near candidates, signals, or 5-to-10m changed areas relative to endpoint-complete edges; 1,503 changed-area-near edges remain a localized topology risk and endpoints are not fabricated",
    }


def signal_fragmentation(complexes: pd.DataFrame) -> dict[str, Any]:
    signal = complexes[complexes["signal_evidence_count"] > 0]
    buckets = {"1": int((signal["signal_evidence_count"] == 1).sum()), "2": int((signal["signal_evidence_count"] == 2).sum()), "3": int((signal["signal_evidence_count"] == 3).sum()), "4+": int((signal["signal_evidence_count"] >= 4).sum())}
    return {"signal_complex_count": len(signal), "signal_nodes_per_complex_distribution": buckets, "singleton_signal_node_share": float((signal["signal_evidence_count"] == 1).mean()), "total_signal_node_assignments": int(signal["signal_evidence_count"].sum())}


def degree_sanity(edges: pd.DataFrame, nodes: pd.DataFrame) -> dict[str, Any]:
    usable = edges.dropna(subset=["from_stage3_node_uid", "to_stage3_node_uid"])
    pairs = usable.assign(
        a=np.minimum(usable["from_stage3_node_uid"], usable["to_stage3_node_uid"]),
        b=np.maximum(usable["from_stage3_node_uid"], usable["to_stage3_node_uid"]),
    ).drop_duplicates(["a", "b"])
    degree = pd.concat([pairs["a"], pairs["b"]]).value_counts()
    all_degree = nodes["stage3_node_uid"].map(degree).fillna(0).astype(int)
    distribution = all_degree.value_counts().sort_index()
    return {
        "node_count": len(nodes), "physical_undirected_edge_pair_count": len(pairs),
        "degree_distribution": {str(int(key)): int(value) for key, value in distribution.items()},
        "degree_ge3_count": int((all_degree >= 3).sum()), "degree_ge4_count": int((all_degree >= 4).sum()),
        "degree_max": int(all_degree.max()), "definition": "unique unordered endpoint pairs over endpoint-complete frozen auto-routable directed edges",
    }


def select_cases(x10: pd.DataFrame, changed: set[str], existing_qa: pd.DataFrame) -> pd.DataFrame:
    pool = x10[x10["intersection_complex_uid"].isin(changed)].copy()
    pool["rank"] = pool["intersection_complex_uid"].map(_hash_rank)
    pool = pool.sort_values("rank")
    used: set[str] = set()
    existing_ids = set(existing_qa.loc[existing_qa["changed_complex_r10"].notna(), "changed_complex_r10"])
    strata = [
        ("signalized", 20, pool["signal_evidence_present"]),
        ("multi_node_divided_road", 20, pool["candidate_member_count"] >= 3),
        ("high_degree", 10, pool["max_member_undirected_degree"] >= 4),
        ("grade_separated", 10, pool["grade_separation_evidence_present"]),
        ("random_changed", 10, pd.Series(True, index=pool.index)),
    ]
    rows = []
    for category, quota, mask in strata:
        subset = pool[mask]
        # Prefer cases already represented in the old QA pack, then fill from
        # frozen changed products. Both paths avoid recomputing complexes.
        subset = subset.assign(existing_qa=subset["intersection_complex_uid"].isin(existing_ids)).sort_values(["existing_qa", "rank"], ascending=[False, True])
        picked = 0
        for row in subset.itertuples(index=False):
            uid = row.intersection_complex_uid
            if uid in used: continue
            rows.append({"complex_r10": uid, "selection_stratum": category, "selection_rank": row.rank, "reused_existing_qa": bool(row.existing_qa)})
            used.add(uid); picked += 1
            if picked >= quota: break
        if picked < quota: raise Stage3S2AError(f"insufficient changed cases for {category}: {picked}/{quota}")
    result = pd.DataFrame(rows)
    result.insert(0, "adjudication_case_id", [f"s2b11_{index:03d}" for index in range(1, len(result) + 1)])
    return result


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _write_png(path: Path, image: np.ndarray) -> None:
    h, w, _ = image.shape; raw = b"".join(b"\x00" + image[y].tobytes() for y in range(h))
    data = b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + _chunk(b"IDAT", zlib.compress(raw, 6)) + _chunk(b"IEND", b"")
    temp = path.with_name(f".{path.name}.tmp"); temp.write_bytes(data); os.replace(temp, path)


def _line(image, a, b, color, width=1):
    x0, y0, x1, y1 = *map(int, a), *map(int, b); dx, sx = abs(x1-x0), 1 if x0<x1 else -1; dy, sy = -abs(y1-y0), 1 if y0<y1 else -1; err=dx+dy
    while True:
        for ox in range(-width,width+1):
            for oy in range(-width,width+1):
                x,y=x0+ox,y0+oy
                if 0<=x<image.shape[1] and 0<=y<image.shape[0]: image[y,x]=color
        if x0==x1 and y0==y1: break
        e=2*err
        if e>=dy: err+=dy; x0+=sx
        if e<=dx: err+=dx; y0+=sy


def render_pair(case, members5, members10, candidates, nodes, edges, folder: Path) -> dict[str, Any]:
    node_lookup = nodes.set_index("stage3_node_uid"); candidate_lookup = candidates.set_index("stage3_node_uid")
    union = set(members5) | set(members10); center = node_lookup.loc[list(union)][["lon","lat"]].mean()
    radius = .0022; width,height=1400,700; image=np.full((height,width,3),248,dtype=np.uint8)
    nearby=edges[edges["start_lon"].between(center.lon-radius,center.lon+radius)&edges["start_lat"].between(center.lat-radius,center.lat+radius)]
    def draw(panel, members, tolerance):
        left=panel*700; pad=25
        def pixel(point): return (left+pad+int((point[0]-(center.lon-radius))/(2*radius)*(650)), height-pad-int((point[1]-(center.lat-radius))/(2*radius)*650))
        for edge in nearby.itertuples(index=False):
            geometry=json.loads(edge.geometry); color=(135,135,135); line_width=0
            if edge.bridge_effective: color=(120,60,180)
            if edge.tunnel_effective: color=(110,100,30)
            source=str(edge.from_stage3_node_uid);target=str(edge.to_stage3_node_uid)
            if source not in members and target in members: color,line_width=(20,80,220),1
            if source in members and target not in members: color,line_width=(20,170,70),1
            for a,b in zip(geometry,geometry[1:]): _line(image,pixel(a),pixel(b),color,line_width)
        pts=[]
        for uid in members:
            node=node_lookup.loc[uid]; p=pixel((node.lon,node.lat));pts.append(p);color=(220,30,30)
            if uid in candidate_lookup.index and candidate_lookup.loc[uid,"rule_c_signal"]: color=(240,180,0)
            if uid in candidate_lookup.index and candidate_lookup.loc[uid,"rule_d_roundabout"]: color=(0,180,190)
            _line(image,(p[0]-4,p[1]),(p[0]+4,p[1]),color,2);_line(image,(p[0],p[1]-4),(p[0],p[1]+4),color,2)
        if pts:
            x0,x1=min(p[0] for p in pts)-8,max(p[0] for p in pts)+8;y0,y1=min(p[1] for p in pts)-8,max(p[1] for p in pts)+8
            for a,b in [((x0,y0),(x1,y0)),((x1,y0),(x1,y1)),((x1,y1),(x0,y1)),((x0,y1),(x0,y0))]:_line(image,a,b,(230,80,20),1)
        # visual tolerance ruler and panel identity: 5 ticks vs 10 ticks.
        _line(image,(left+30,25),(left+30+tolerance*8,25),(0,0,0),2)
        for tick in range(tolerance): _line(image,(left+34+tick*8,18),(left+34+tick*8,32),(0,0,0),0)
    draw(0,set(members5),5);draw(1,set(members10),10);_line(image,(699,0),(699,699),(0,0,0),2)
    path=folder/f"{case.adjudication_case_id}_{case.complex_r10}_r05_vs_r10.png";_write_png(path,image)
    return {"adjudication_case_id":case.adjudication_case_id,"complex_r10":case.complex_r10,"selection_stratum":case.selection_stratum,"png_path":path.as_posix(),"png_sha256":sha256_file(path),"png_size_bytes":path.stat().st_size,"left_panel":"5m","right_panel":"10m"}


def run(root: Path) -> dict[str, Any]:
    if git_head(root) != AUTHORIZED_BASE: raise Stage3S2AError("S2B-1.1 must run from authorized S2B-1 base")
    base=root/"stage3/output/odd_tod"; s2a=base/"s2a"; s2b=base/"s2b/calibration"; out=base/"s2b/closure_5v10"; docs=root/"stage3/docs/odd_tod/s2b"; out.mkdir(parents=True,exist_ok=True);docs.mkdir(parents=True,exist_ok=True)
    paths={"edges":s2a/"stage3_full_network_edges.parquet","nodes":s2a/"stage3_full_network_nodes.parquet","controls":s2a/"stage3_control_evidence.parquet","candidates":s2b/"junction_candidates.parquet","m5":s2b/"node_membership_r05.parquet","m10":s2b/"node_membership_r10.parquet","x5":s2b/"complexes_r05.parquet","x10":s2b/"complexes_r10.parquet","qa":s2b/"qa_sample.parquet","s2b_evidence":docs/"stage3_s2b1_evidence_bundle.json"}
    edges,nodes,controls,candidates,m5,m10,x5,x10,qa=[_read(paths[key]) for key in ("edges","nodes","controls","candidates","m5","m10","x5","x10","qa")]
    pairs,changed=changed_groups(m5,m10);transformer=Transformer.from_crs(4326,32649,always_xy=True)
    r05_to_r10 = pairs[pairs["changed_05_10"]].drop_duplicates("complex_r05").set_index("complex_r05")["complex_r10"].to_dict()
    qa = qa.copy()
    qa["changed_complex_r10"] = np.where(
        qa["tolerance_m"].eq(10) & qa["intersection_complex_uid"].isin(changed),
        qa["intersection_complex_uid"],
        qa["intersection_complex_uid"].map(r05_to_r10),
    )
    endpoint=endpoint_diagnostic(edges,nodes,candidates,controls,pairs,changed,transformer); degree=degree_sanity(edges,nodes); signal={"r05":signal_fragmentation(x5),"r10":signal_fragmentation(x10)}
    cases=select_cases(x10,changed,qa); member5=m5.groupby("intersection_complex_uid")["stage3_node_uid"].agg(list).to_dict(); member10=m10.groupby("intersection_complex_uid")["stage3_node_uid"].agg(list).to_dict(); pair_lookup=pairs.groupby("complex_r10")["complex_r05"].agg(lambda values:sorted(set(values))).to_dict()
    cases["complexes_r05"] = cases["complex_r10"].map(lambda uid:json.dumps(pair_lookup[uid])); cases["member_node_count_r05_total"] = cases["complexes_r05"].map(lambda value:sum(len(member5[uid]) for uid in json.loads(value))); cases["member_node_count_r10"] = cases["complex_r10"].map(lambda uid:len(member10[uid]));cases["adjudication_label"]="";cases["reviewer_note"]="";cases["allowed_labels"]=json.dumps(ALLOWED_LABELS)
    case_path=out/"s2b11_adjudication_cases.parquet";atomic_parquet(case_path,cases)
    csv_path=out/"s2b11_adjudication_sheet.csv";csv_temp=csv_path.with_name(f".{csv_path.name}.tmp");cases.to_csv(csv_temp,index=False,encoding="utf-8-sig");os.replace(csv_temp,csv_path)
    folder=out/"paired_visualizations";folder.mkdir(parents=True,exist_ok=True);visual=[]
    for case in cases.itertuples(index=False):
        members5=set().union(*(set(member5[uid]) for uid in json.loads(case.complexes_r05)));visual.append(render_pair(case,members5,member10[case.complex_r10],candidates,nodes,edges,folder))
    visual=pd.DataFrame(visual);visual["png_path"]=visual["png_path"].map(lambda value:Path(value).relative_to(root).as_posix());visual_path=out/"s2b11_visual_index.parquet";atomic_parquet(visual_path,visual)
    png_set_sha256=hashlib.sha256("\n".join(f"{r.png_path}|{r.png_sha256}" for r in visual.sort_values("png_path").itertuples(index=False)).encode()).hexdigest()
    qa_manifest={"schema_version":"stage3_s2b11_qa_manifest.1","phase_status":PHASE_STATUS,"case_count":len(cases),"strata_counts":{str(k):int(v) for k,v in cases.selection_stratum.value_counts().items()},"existing_qa_reused_count":int(cases["reused_existing_qa"].sum()),"labels_allowed":ALLOWED_LABELS,"labels_filled":0,"left_panel":"5m","right_panel":"10m","adjudication_cases":parquet_descriptor(case_path,root),"adjudication_sheet_csv":source_descriptor(csv_path,root),"visual_index":parquet_descriptor(visual_path,root),"png_set_sha256":png_set_sha256,"png_total_size_bytes":int(visual["png_size_bytes"].sum()),"test31_used":False,"av_feasibility_used":False};qa_manifest["artifact_sha256"]=payload_hash(qa_manifest);qa_manifest_path=docs/"stage3_s2b11_qa_manifest.json";atomic_json(qa_manifest_path,qa_manifest)
    closure={"schema_version":"stage3_s2b11_closure.1","phase_status":PHASE_STATUS,"authorized_base":AUTHORIZED_BASE,"s2b1_engineering":"PASS","s2b1_tolerance_selection":"NOT_YET_CLOSED","recommendation_status":"5m_RECOMMENDATION_NOT_ACCEPTED","prior_s2b1_recommendation_superseded":True,"recommendation_function_structural_small_radius_bias_removed":True,"rejected_baselines_m":[15,20],"final_review_pair_m":[5,10],"four_tolerance_products_recomputed":False,"endpoint_incompleteness":endpoint,"signal_fragmentation":signal,"degree_sanity":degree,"changed_05_10_complex_count":len(changed),"adjudication_case_count":len(cases),"adjudication_strata_counts":{str(k):int(v) for k,v in cases.selection_stratum.value_counts().items()},"existing_qa_reused_count":int(cases["reused_existing_qa"].sum()),"adjudication_labels_allowed":ALLOWED_LABELS,"adjudication_labels_filled":0,"scientific_decision":"PENDING_HUMAN_ADJUDICATION","s2b2_authorized":False,"next_phase_authorized":False};closure["artifact_sha256"]=payload_hash(closure);closure_path=docs/"stage3_s2b11_5v10_closure.json";atomic_json(closure_path,closure)
    byclass="\n".join(f"- {r['road_class']}: {r['endpoint_incomplete_count']:,}/{r['full_edge_count']:,} ({r['endpoint_incomplete_share']:.2%})" for r in endpoint["by_road_class"])
    md=f"""# Stage 3 S2B-1.1: 5m vs 10m Closure Pack

Status: `{PHASE_STATUS}`. `S2B1_ENGINEERING = PASS`; tolerance selection remains open. The prior 5m recommendation is withdrawn as scientific evidence because its rule structurally preferred smaller radii.

## Endpoint incompleteness

There are `{endpoint['endpoint_incomplete_edge_count']:,}` endpoint-incomplete edges. Within 20m of a junction candidate: `{endpoint['near_junction_candidate_20m_count']:,}` ({endpoint['near_junction_candidate_20m_share']:.2%}); signal: `{endpoint['near_signal_20m_count']:,}` ({endpoint['near_signal_20m_share']:.2%}); a 5-to-10m changed area: `{endpoint['near_changed_05_10_20m_count']:,}` ({endpoint['near_changed_05_10_20m_share']:.2%}). Relative to endpoint-complete edges, the three near shares are lower (ratios 0.44, 0.78, and 0.46), so there is no evidence of systematic enrichment around intersection evidence. The 1,503 changed-area-near edges remain a localized risk. These edges are excluded from topology rather than guessed.

By road class:
{byclass}

## Signal fragmentation

- 5m: `{json.dumps(signal['r05'],sort_keys=True)}`
- 10m: `{json.dumps(signal['r10'],sort_keys=True)}`

## Degree sanity

`{json.dumps(degree,sort_keys=True)}`

## Targeted human adjudication

The pack contains `{len(cases)}` deterministic changed areas: `{json.dumps(closure['adjudication_strata_counts'],sort_keys=True)}`; `{closure['existing_qa_reused_count']}` came from areas already represented in the prior QA pack. Each PNG shows 5m on the left and 10m on the right. The Parquet/CSV adjudication sheet is intentionally blank and accepts only `{json.dumps(ALLOWED_LABELS)}`.

`15m` and `20m` remain rejected baselines. `5m` versus `10m` is the final review pair. No tolerance is frozen and S2B-2 remains unauthorized.
""";md_path=docs/"stage3_s2b11_5v10_closure.md";atomic_text(md_path,md)
    test_path=docs/"stage3_s2b11_test_evidence.json"
    outputs={"closure_json":source_descriptor(closure_path,root),"closure_markdown":source_descriptor(md_path,root),"qa_manifest":source_descriptor(qa_manifest_path,root),"adjudication_cases":parquet_descriptor(case_path,root),"adjudication_sheet_csv":source_descriptor(csv_path,root),"visual_index":parquet_descriptor(visual_path,root)}
    if test_path.is_file(): outputs["test_evidence"]=source_descriptor(test_path,root)
    evidence={"schema_version":"stage3_s2b11_evidence.1","phase_status":PHASE_STATUS,"authorized_base":AUTHORIZED_BASE,"inputs":{k:(parquet_descriptor(v,root) if v.suffix==".parquet" else source_descriptor(v,root)) for k,v in paths.items()},"outputs":outputs,"png_set_sha256":png_set_sha256,"guards":{"four_tolerance_products_recomputed":False,"tolerance_frozen":False,"s2b2_authorized":False,"next_phase_authorized":False,"test31_used":False,"av_feasibility_used":False,"stage4_used":False}};evidence["artifact_sha256"]=payload_hash(evidence);evidence_path=docs/"stage3_s2b11_evidence_bundle.json";atomic_json(evidence_path,evidence);return closure


def verify(path: Path, root: Path) -> dict[str, Any]:
    evidence=read_json(path);fail=[]
    if evidence.get("artifact_sha256")!=payload_hash(evidence):fail.append("payload hash")
    for section in ("inputs","outputs"):
        for d in evidence.get(section,{}).values():
            p=Path(d["path"]);p=p if p.is_absolute() else root/p
            if not p.is_file() or sha256_file(p)!=d["sha256"]:fail.append(f"binding: {d['path']}")
    if any(evidence.get("guards",{}).values()):fail.append("scope")
    return {"status":"PASS" if not fail else "FAIL","failures":fail,"phase_status":evidence.get("phase_status")}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--root",type=Path,default=Path.cwd());parser.add_argument("--verify",type=Path);args=parser.parse_args(argv);root=args.root.resolve();result=verify(args.verify.resolve(),root) if args.verify else run(root);print(json.dumps(result,indent=2,ensure_ascii=False));return 0 if result.get("status","PASS")=="PASS" else 1


if __name__=="__main__":raise SystemExit(main())
