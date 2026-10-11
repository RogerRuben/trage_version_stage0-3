"""Derive typed historical target shares from existing Stage1/M3/Stage3 products.

Exact Stage1 OSM pickup-node identity and 15-minute bins join the existing target
network. No geometric nearest-road repair, Test31 read, routing or inference.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import psutil

from stage4.analysis.capability_chain_instances import atomic_json, atomic_parquet, sha
from stage4.dispatch.repositioning_policy import REFERENCE_REL, REFERENCE_MANIFEST_REL

OUT = Path("stage4/output/capability_chain_planning/operations_v5_input")
REFERENCE = OUT / "type_demand_reference.parquet"
MANIFEST = OUT / "manifest.json"
DOC = Path("stage4/docs/capability_chain_planning/operations_v5")
TEMPLATES = Path("stage4/output/symmetric_flexibility_v1/input/train_request_templates.parquet")
DATES = ("20161010", "20161017", "20161024")
PROFILE_COLUMNS = {"C": "compatible_C", "M": "compatible_M", "A": "compatible_A"}


def aggregate_targets(templates, node_rows, reference):
    """Missing identity is omitted from target estimation, never capability failure."""
    if (templates[["date", "order_id"]].duplicated().any()
            or node_rows[["date", "order_id"]].duplicated().any()
            or reference[["time_bin_index", "node_id"]].duplicated().any()):
        raise ValueError("nonunique historical identity or reference key")
    work = templates.merge(node_rows, on=["date", "order_id"], validate="one_to_one", how="left")
    if work["stage1_core_eligible"].isna().any() or not work.stage1_core_eligible.all():
        raise ValueError("template has no high-quality Stage1 source")
    node = pd.to_numeric(work.start_node, errors="coerce")
    valid_node = node.notna() & np.isfinite(node) & node.gt(0) & node.mod(1).eq(0)
    times = pd.to_datetime(pd.to_numeric(work.departure_time, errors="coerce"), unit="s", utc=True).dt.tz_convert("Asia/Shanghai")
    if times.isna().any():
        raise ValueError("missing Stage1 release time")
    within_train = times.dt.strftime("%Y%m%d").between("20161009", "20161024")
    bins = times.dt.hour * 4 + times.dt.minute // 15
    valid = work.loc[valid_node & within_train].copy()
    valid["node_id"] = node.loc[valid.index].astype("int64")
    valid["time_bin_index"] = bins.loc[valid.index].astype(int)
    target_columns = ["time_bin_index", "node_id", "lon_wgs84", "lat_wgs84"]
    valid = valid.merge(reference[target_columns], on=["time_bin_index", "node_id"],
                        validate="many_to_one", how="left")
    supported = valid.lon_wgs84.notna() & valid.lat_wgs84.notna()
    facts = dict(template_rows=len(work), missing_or_invalid_start_node=int((~valid_node).sum()),
        outside_train_actual_date=int((~within_train).sum()), missing_existing_target_identity=int((~supported).sum()),
        mapped_rows=int(supported.sum()), mapping="EXACT_STAGE1_START_NODE_AND_LOCAL_15_MINUTE_BIN")
    valid = valid.loc[supported]
    pieces, profiles = [], {}
    for profile, column in PROFILE_COLUMNS.items():
        if work[column].isna().any():
            raise ValueError("missing historical research compatibility")
        subset = valid.loc[valid[column]]
        counts = subset.groupby(target_columns, sort=True).size().rename("train_pickup_count").reset_index()
        counts["demand_share"] = counts.train_pickup_count / counts.groupby("time_bin_index").train_pickup_count.transform("sum")
        counts["profile_id"] = profile
        if not np.allclose(counts.groupby("time_bin_index").demand_share.sum(), 1.):
            raise ValueError("typed target shares do not normalize")
        pieces.append(counts)
        profiles[profile] = dict(source_compatible_orders=int(work[column].sum()),
            mapped_compatible_orders=len(subset), target_node_bins=len(counts),
            supported_15_minute_bins=int(counts.time_bin_index.nunique()))
    return pd.concat(pieces, ignore_index=True), dict(facts, profiles=profiles)


def build(root):
    root, started = Path(root).resolve(), perf_counter()
    base, source = root / REFERENCE_REL, root / TEMPLATES
    # Never invoke the old reference builder's raw/PBF pipeline implicitly.
    if not base.is_file() or not (root / REFERENCE_MANIFEST_REL).is_file():
        raise FileNotFoundError("existing historical reference is required; no upstream production authorized")
    old_manifest = json.loads((root / REFERENCE_MANIFEST_REL).read_text(encoding="utf-8"))
    if old_manifest.get("test31_included") is not False or old_manifest["reference_sha256"] != sha(base):
        raise ValueError("existing historical reference binding is invalid")
    before = {REFERENCE_REL.as_posix(): sha(base), TEMPLATES.as_posix(): sha(source)}
    templates = pd.read_parquet(source, columns=["date", "order_id", "common_eligible", *PROFILE_COLUMNS.values()])
    templates["date"] = templates.date.astype(str)
    templates["order_id"] = templates.order_id.astype(str)
    if set(templates.date) != set(DATES) or not templates.common_eligible.all():
        raise ValueError("only the existing three historical Monday templates are allowed")
    nodes, digest, files = [], hashlib.sha256(), []
    for date in DATES:
        paths = sorted((root / f"stage1/input_v1/split=train/date={date}").glob("bucket=*/order_base.parquet"))
        if not paths:
            raise FileNotFoundError(f"existing Stage1 Train date missing: {date}")
        frames = []
        for path in paths:
            value = sha(path)
            relative = path.relative_to(root).as_posix()
            digest.update(f"{relative}|{value}\n".encode())
            files.append((path, value))
            frames.append(pd.read_parquet(path, columns=["order_id", "start_node", "departure_time", "stage1_core_eligible"]))
        frame = pd.concat(frames, ignore_index=True)
        frame["order_id"] = frame.order_id.astype(str)
        frame = frame.loc[frame.order_id.isin(templates.loc[templates.date.eq(date), "order_id"])].copy()
        frame["date"] = date
        nodes.append(frame)
        del frames, frame
        gc.collect()
    reference = pd.read_parquet(base, columns=["time_bin_index", "node_id", "lon_wgs84", "lat_wgs84"])
    typed, facts = aggregate_targets(templates, pd.concat(nodes, ignore_index=True), reference)
    destination = root / REFERENCE
    if any(sha(path) != value for path, value in files) or any(sha(root / path) != value for path, value in before.items()):
        raise ValueError("historical sources changed during derivation")
    memory = psutil.Process().memory_info()
    peak = max(memory.rss, getattr(memory, "peak_wset", memory.rss)) / 2**20
    if peak > 512:
        raise MemoryError("typed reference preprocessing exceeded 512 MiB")
    atomic_parquet(destination, typed)
    manifest = dict(version="type_conditioned_reference_v1", **facts,
        source_partition_dates=list(DATES), actual_date_cutoff="20161024",
        HV_reference_train_dates=old_manifest["train_dates"], HV_reference_unchanged=True,
        source_hashes=before, Stage1_source_file_count=len(files), Stage1_source_digest=digest.hexdigest(),
        reference_sha256=sha(destination), rows=len(typed), test31_included=False,
        no_new_inference=True, no_routing=True, no_map_matching=True,
        passenger_acceptance_treatment="CONSTANT_P70_CANCELS_IN_NORMALIZED_TARGET_SHARES",
        empty_typed_bin_policy="NO_TYPED_OPTIONAL_MOVE_NOT_CAPABILITY_FAILURE",
        runtime_s=perf_counter() - started, peak_rss_mib=peak)
    atomic_json(root / MANIFEST, manifest)
    atomic_json(root / DOC / "preparation_summary.json", manifest)
    return manifest


def load(root, base_reference, templates_path=TEMPLATES):
    root = Path(root).resolve()
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    if (manifest["version"] != "type_conditioned_reference_v1" or manifest["test31_included"]
            or manifest["source_partition_dates"] != list(DATES)
            or manifest["reference_sha256"] != sha(root / REFERENCE)
            or manifest["source_hashes"][REFERENCE_REL.as_posix()] != sha(root / REFERENCE_REL)
            or manifest["source_hashes"][TEMPLATES.as_posix()] != sha(root / templates_path)):
        raise ValueError("typed historical target provenance mismatch")
    digest, file_count = hashlib.sha256(), 0
    for date in DATES:
        for path in sorted((root / f"stage1/input_v1/split=train/date={date}").glob("bucket=*/order_base.parquet")):
            digest.update(f"{path.relative_to(root).as_posix()}|{sha(path)}\n".encode())
            file_count += 1
    if file_count != manifest["Stage1_source_file_count"] or digest.hexdigest() != manifest["Stage1_source_digest"]:
        raise ValueError("typed target Stage1 sources changed")
    typed = pd.read_parquet(root / REFERENCE)
    base_keys = set(zip(base_reference.time_bin_index, base_reference.node_id))
    if any(key not in base_keys for key in zip(typed.time_bin_index, typed.node_id)):
        raise ValueError("typed targets introduced a new physical destination")
    return typed, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(build(args.root), ensure_ascii=False))


if __name__ == "__main__":
    main()
