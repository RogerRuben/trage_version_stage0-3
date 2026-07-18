from __future__ import annotations

import json

import pandas as pd

from stage0.v5.archive import sampled_orders_path, sampling_run_id, stable_sample
from stage0.v5.config import Stage0Config, stable_hash
from stage0.v5.pipeline import _partition_done
from stage0.v5.pipeline import export_case_traces
from stage0.v5.retention import prune_point_work
from stage0.v5.gates import require_test_freeze


def test_stable_hash_sampling_is_order_invariant():
    orders = pd.DataFrame({"order_id": ["c", "a", "b"], "eligible": [True] * 3})
    left = stable_sample(orders, "20161010", 2, 20261009).order_id.tolist()
    right = stable_sample(orders.sample(frac=1, random_state=1), "20161010", 2, 20261009).order_id.tolist()
    assert left == right
    assert stable_hash("20161010", "a", seed=1) == stable_hash("20161010", "a", seed=1)


def test_sampling_runs_are_namespaced_by_scope_and_size(tmp_path):
    small = sampling_run_id(["20161010"], 2000, 20261009)
    large = sampling_run_id(["20161010"], 10000, 20261009)
    other_dates = sampling_run_id(["20161010", "20161014"], 2000, 20261009)
    assert len({small, large, other_dates}) == 3
    assert sampled_orders_path(tmp_path, small, "20161010") != sampled_orders_path(tmp_path, large, "20161010")


def test_train_validation_test_dates_are_disjoint():
    config = Stage0Config.load(__import__("pathlib").Path("stage0/config/stage0_v5.yaml"))
    split = config.section("split")
    train, validation, test = map(lambda name: set(map(str, split[name])), ("train", "validation", "test"))
    assert not (train & validation or train & test or validation & test)


def test_compact_pruning_is_dry_run_and_requires_products(tmp_path):
    values = Stage0Config.load(__import__("pathlib").Path("stage0/config/stage0_v5.yaml")).values.copy()
    values = {**values, "paths": {**values["paths"], "output": str(tmp_path / "out"), "work": str(tmp_path / "work")}}
    from stage0.v5.config import config_hash
    config = Stage0Config(tmp_path / "config.yaml", values, config_hash(values))
    result = prune_point_work(config, tmp_path, execute=False)
    assert result["dry_run"] and result["missing_prerequisites"]


def test_resume_partition_requires_manifest_and_every_product(tmp_path):
    output = tmp_path
    date, bucket, digest = "20161010", 1, "abc"
    manifest = output / "manifests/partitions" / f"day={date}" / f"part={bucket:03d}.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"status": "PASS", "config_hash": digest}))
    assert not _partition_done(output, date, bucket, digest, 2000)


def test_test_dates_require_current_frozen_manifest(tmp_path):
    source = __import__("pathlib").Path("stage0/config/stage0_v5.yaml")
    loaded = Stage0Config.load(source)
    values = {**loaded.values, "paths": {**loaded.values["paths"], "output": str(tmp_path / "out")}}
    from stage0.v5.config import config_hash
    config = Stage0Config(source, values, config_hash(values))
    import pytest
    with pytest.raises(RuntimeError, match="locked"):
        require_test_freeze(config, tmp_path, ["20161028"])
    manifest = tmp_path / "out/manifests/stage0_freeze_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"status": "FROZEN", "config_hash": config.digest}), encoding="utf-8")
    require_test_freeze(config, tmp_path, ["20161028"])
    require_test_freeze(config, tmp_path, ["20161010"])


def test_rejected_case_traces_are_bounded_by_failure_reason(tmp_path):
    source = __import__("pathlib").Path("stage0/config/stage0_v5.yaml")
    loaded = Stage0Config.load(source)
    values = {
        **loaded.values,
        "paths": {**loaded.values["paths"], "output": str(tmp_path / "out"), "work": str(tmp_path / "work")},
        "runtime": {
            **loaded.values["runtime"],
            "case_trace_per_failure_reason_per_day": 2,
            "case_trace_representative_per_day": 3,
        },
    }
    from stage0.v5.config import config_hash
    config = Stage0Config(source, values, config_hash(values))
    date, run_id = "20161010", "sample-run"
    quality = pd.DataFrame({
        "date": [date] * 30,
        "order_id": [f"o{i}" for i in range(30)],
        "route_quality": ["rejected"] * 20 + ["strict_core"] * 10,
        "hard_error_flags": ['["gap"]'] * 10 + ['["direction"]'] * 10 + ["[]"] * 10,
        "soft_quality_flags": ["[]"] * 30,
    })
    quality_path = tmp_path / "out/route_quality" / f"day={date}" / "part=000.parquet"
    quality_path.parent.mkdir(parents=True)
    quality.to_parquet(quality_path, index=False)
    route_path = tmp_path / "out/route_parts" / f"day={date}" / "part=000.parquet"
    route_path.parent.mkdir(parents=True)
    pd.DataFrame({
        "order_id": quality.order_id, "edge_uid": "edge-a", "route_sequence": 0,
        "route_source": "observed",
    }).to_parquet(route_path, index=False)
    points = pd.DataFrame({
        "order_id": quality.order_id, "timestamp": range(30), "edge_uid": "edge-a",
        "candidate_rank": 1, "gps_to_edge_distance_m": 2.0,
    })
    point_path = tmp_path / "work/matched_diagnostics" / run_id / f"day={date}" / "part=000.parquet"
    point_path.parent.mkdir(parents=True)
    points.to_parquet(point_path, index=False)
    result = export_case_traces(config, tmp_path, [date], run_id)
    assert result["case_trace_total"] == 7
    retained = pd.read_parquet(tmp_path / "out/case_traces" / run_id / f"day={date}" / "points.parquet")
    assert retained.order_id.nunique() == 7
    assert {"edge_uid", "candidate_rank", "gps_to_edge_distance_m"} <= set(retained.columns)
    retained_routes = pd.read_parquet(tmp_path / "out/case_traces" / run_id / f"day={date}" / "route_parts.parquet")
    assert retained_routes.order_id.nunique() == 7
