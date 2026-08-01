"""Global Stage 2 v4 preflight over the frozen Stage 1 production."""

from __future__ import annotations

import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .config import Stage2V4Config
from .contracts import STAGE2_V4_PREFLIGHT_SCHEMA_VERSION, Stage2V4ContractError
from .io import sha256_file
from .release import bind_stage1_release
from .stage1_adapter import build_route_alignment, discover_stage1_buckets


def _accumulate(total: Counter[str], values: dict[str, int]) -> None:
    for key, value in values.items():
        if key == "maximum_traversal_span_length":
            total[key] = max(total[key], int(value))
        else:
            total[key] += int(value)


def run_preflight(
    config: Stage2V4Config,
    *,
    stage1_release: str | Path,
    stage1_output: str | Path,
    stage1_input: str | Path,
    stage1_models: str | Path,
    alignment_output: str | Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    binding = bind_stage1_release(
        stage1_release,
        stage1_output,
        stage1_input,
        stage1_models,
        config,
    )
    refs = discover_stage1_buckets(stage1_output, stage1_input)
    expected = config.section("stage1_release")
    if len(refs) != expected["bucket_count"]:
        raise Stage2V4ContractError(
            f"expected {expected['bucket_count']} buckets, found {len(refs)}"
        )

    target = Path(alignment_output)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.tmp-",
        suffix=".parquet",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    writer: pq.ParquetWriter | None = None
    counters: Counter[str] = Counter()
    by_split: dict[str, Counter[str]] = {}
    by_date: dict[str, Counter[str]] = {}
    try:
        for ref in refs:
            result = build_route_alignment(ref)
            table = pa.Table.from_pandas(result.traversal_alignment, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
            _accumulate(counters, result.counters)
            _accumulate(by_split.setdefault(ref.split, Counter()), result.counters)
            _accumulate(by_date.setdefault(ref.date, Counter()), result.counters)
        if writer is None:
            raise Stage2V4ContractError("no Stage 1 buckets were available")
        writer.close()
        writer = None
        os.replace(temporary, target)
    except Exception:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise

    failures: list[str] = []
    reconciliations = {
        "order_count": expected["accepted_order_count"],
        "traversal_label_count": expected["traversal_label_count"],
        "route_token_count": expected["route_sequence_count"],
    }
    for name, wanted in reconciliations.items():
        if counters[name] != wanted:
            failures.append(f"{name}: expected {wanted}, got {counters[name]}")
    for name in (
        "orphan_traversal_label_count",
        "decision_time_missing_count",
        "self_order_history_candidate_count",
        "route_token_conservation_error",
        "label_row_conservation_error",
    ):
        if counters[name]:
            failures.append(f"{name}: expected 0, got {counters[name]}")

    report = {
        "schema_version": STAGE2_V4_PREFLIGHT_SCHEMA_VERSION,
        "engineering_status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "stage2_config_sha256": config.digest,
        "stage1_release": {
            "manifest_sha256": sha256_file(stage1_release),
            "tag": expected["release_tag"],
            "commit": binding.resolved_release_commit,
            "config_sha256": expected["config_sha256"],
            "code_sha": expected["code_sha"],
            "model_id": expected["model_id"],
            "output_manifest_aggregate_sha256": expected[
                "output_manifest_aggregate_sha256"
            ],
            "stage0_release": expected["stage0_release"],
        },
        "bucket_count": len(refs),
        "counters": dict(sorted(counters.items())),
        "by_split": {
            key: dict(sorted(value.items())) for key, value in sorted(by_split.items())
        },
        "by_date": {
            key: dict(sorted(value.items())) for key, value in sorted(by_date.items())
        },
        "alignment": {
            "path": target.as_posix(),
            "sha256": sha256_file(target),
            "mapping": (
                "one_to_one"
                if counters["multi_token_traversal_count"] == 0
                else "span"
            ),
            "span_length_min": 1,
            "span_length_max": counters["maximum_traversal_span_length"],
        },
        "decision_time": {
            "source": "stage0_order_departure_time",
            "missing_count": counters["decision_time_missing_count"],
        },
        "runtime_s": time.perf_counter() - started,
    }
    if failures:
        raise Stage2V4ContractError("Stage 2 v4 preflight failed: " + "; ".join(failures))
    return report
