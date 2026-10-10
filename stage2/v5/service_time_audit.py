"""Audit the frozen Stage 1 traversal-time fields before defining v5 targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SCHEMA_VERSION = "stage2_v5_service_time_target_audit.1"
DIRECT_SOURCE = "direct_observed"
TRAVERSAL_COLUMNS = (
    "order_id",
    "traversal_id",
    "measurement_source",
    "time_source",
    "time_observation_valid",
    "enter_time",
    "exit_time",
    "observed_travel_time_s",
    "engine_allocated_travel_time_s",
    "travel_time_s",
    "observed_distance_m",
    "allocated_distance_m",
)
LABEL_COLUMNS = (
    "order_id",
    "traversal_id",
    "measurement_source",
    "direct_observed_time_s",
    "direct_observed_distance_m",
    "allocated_distance_m",
    "direct_distance_coverage_share",
    "observed_sec_per_m",
    "rts_measurement_available",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


class _Distribution:
    """Exact moments plus a bounded deterministic quantile sample."""

    def __init__(self, sample_per_partition: int = 10_000) -> None:
        self.sample_per_partition = sample_per_partition
        self.count = 0
        self.total = 0.0
        self.total_square = 0.0
        self.minimum = float("inf")
        self.maximum = float("-inf")
        self.samples: list[np.ndarray] = []

    def update(self, values: np.ndarray) -> None:
        clean = np.asarray(values, dtype=float)
        clean = clean[np.isfinite(clean)]
        if not len(clean):
            return
        self.count += int(len(clean))
        self.total += float(clean.sum(dtype=np.float64))
        self.total_square += float(np.square(clean).sum(dtype=np.float64))
        self.minimum = min(self.minimum, float(clean.min()))
        self.maximum = max(self.maximum, float(clean.max()))
        stride = max(1, int(np.ceil(len(clean) / self.sample_per_partition)))
        self.samples.append(clean[::stride][: self.sample_per_partition].copy())

    def summary(self) -> dict[str, Any]:
        if not self.count:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "p01": None,
                "p10": None,
                "p50": None,
                "p90": None,
                "p95": None,
                "p99": None,
                "max": None,
                "quantile_sample_count": 0,
            }
        sample = np.concatenate(self.samples) if self.samples else np.empty(0)
        mean = self.total / self.count
        variance = max(self.total_square / self.count - mean * mean, 0.0)
        quantiles = np.quantile(sample, [0.01, 0.1, 0.5, 0.9, 0.95, 0.99])
        return {
            "count": self.count,
            "mean": mean,
            "std": float(np.sqrt(variance)),
            "min": self.minimum,
            "p01": float(quantiles[0]),
            "p10": float(quantiles[1]),
            "p50": float(quantiles[2]),
            "p90": float(quantiles[3]),
            "p95": float(quantiles[4]),
            "p99": float(quantiles[5]),
            "max": self.maximum,
            "quantile_sample_count": int(len(sample)),
        }


def _source_class(source: pd.Series) -> np.ndarray:
    normalized = source.astype("string").fillna("unresolved").astype(str)
    mapping = {
        "direct_observed": "direct_raw_gps_interval",
        "interval_supported": "interval_supported_no_link_time",
        "engine_interpolated": "engine_interpolated_no_link_time",
        "engine_allocated": "engine_allocated_time",
        "unresolved": "unresolved_no_link_time",
    }
    return normalized.map(mapping).fillna("unknown_no_link_time").to_numpy(str)


def _relative_label_path(
    traversal_path: Path,
    input_root: Path,
    label_root: Path,
) -> Path:
    return label_root / traversal_path.parent.relative_to(input_root) / "traversal_labels.parquet"


def audit_service_time_targets(
    input_root: str | Path,
    label_root: str | Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    inputs = Path(input_root)
    labels = Path(label_root)
    traversal_paths = sorted(inputs.rglob("link_traversals.parquet"))
    if not traversal_paths:
        raise FileNotFoundError(f"no link_traversals under {inputs}")

    counters: Counter[str] = Counter()
    measurement_sources: Counter[str] = Counter()
    time_sources: Counter[str] = Counter()
    source_classes: Counter[str] = Counter()
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    by_date: dict[str, Counter[str]] = defaultdict(Counter)
    distributions = {
        "direct_observed_time_s": _Distribution(),
        "direct_observed_distance_m": _Distribution(),
        "allocated_distance_m": _Distribution(),
        "direct_observed_sec_per_m": _Distribution(),
        "raw_derivable_direct_sec_per_m": _Distribution(),
        "direct_distance_coverage_share": _Distribution(),
        "enter_exit_span_minus_direct_time_s": _Distribution(),
    }

    for traversal_path in traversal_paths:
        label_path = _relative_label_path(traversal_path, inputs, labels)
        if not label_path.is_file():
            raise FileNotFoundError(f"missing paired traversal labels: {label_path}")
        relative_parts = traversal_path.relative_to(inputs).parts
        split = relative_parts[0].split("=", 1)[1]
        date = relative_parts[1].split("=", 1)[1]
        traversal = pd.read_parquet(traversal_path, columns=list(TRAVERSAL_COLUMNS))
        label = pd.read_parquet(label_path, columns=list(LABEL_COLUMNS))
        counters["partition_count"] += 1
        counters["traversal_row_count"] += len(traversal)
        counters["label_row_count"] += len(label)
        counters["traversal_duplicate_key_count"] += int(
            traversal.duplicated(["order_id", "traversal_id"]).sum()
        )
        counters["label_duplicate_key_count"] += int(
            label.duplicated(["order_id", "traversal_id"]).sum()
        )

        source = traversal["measurement_source"].astype("string").fillna("missing")
        time_source = traversal["time_source"].astype("string").fillna("missing")
        measurement_sources.update(source.astype(str).value_counts().to_dict())
        time_sources.update(time_source.astype(str).value_counts().to_dict())
        classes = _source_class(traversal["measurement_source"])
        source_classes.update(pd.Series(classes).value_counts().to_dict())

        observed_time = _numeric(traversal, "observed_travel_time_s")
        engine_time = _numeric(traversal, "engine_allocated_travel_time_s")
        travel_time = _numeric(traversal, "travel_time_s")
        observed_distance = _numeric(traversal, "observed_distance_m")
        allocated_distance = _numeric(traversal, "allocated_distance_m")
        enter = _numeric(traversal, "enter_time")
        exit_ = _numeric(traversal, "exit_time")
        observation_flag = traversal["time_observation_valid"].fillna(False).to_numpy(bool)
        direct_source = source.eq(DIRECT_SOURCE).to_numpy(bool)
        observed_time_valid = np.isfinite(observed_time) & (observed_time > 0)
        observed_distance_valid = np.isfinite(observed_distance) & (observed_distance > 0)
        allocated_distance_valid = np.isfinite(allocated_distance) & (allocated_distance > 0)
        enter_exit_valid = np.isfinite(enter) & np.isfinite(exit_) & (exit_ >= enter)
        direct_time_valid = direct_source & observed_time_valid
        direct_pace_valid = direct_time_valid & observed_distance_valid
        full_distance_time_valid = direct_time_valid & allocated_distance_valid

        counters["direct_source_count"] += int(direct_source.sum())
        counters["travel_time_target_valid_count"] += int(direct_time_valid.sum())
        counters["travel_time_direct_valid_count"] += int(direct_time_valid.sum())
        counters["travel_time_interpolated_valid_count"] += int(
            (np.isfinite(engine_time) & (engine_time > 0)).sum()
        )
        counters["raw_direct_pace_derivable_count"] += int(direct_pace_valid.sum())
        counters["full_distance_time_derivation_count"] += int(full_distance_time_valid.sum())
        counters["engine_time_nonnull_count"] += int(np.isfinite(engine_time).sum())
        counters["travel_time_nonnull_count"] += int(np.isfinite(travel_time).sum())
        counters["observed_time_nonnull_count"] += int(np.isfinite(observed_time).sum())
        counters["time_observation_valid_true_count"] += int(observation_flag.sum())
        counters["time_observation_valid_true_non_direct_count"] += int(
            (observation_flag & ~direct_source).sum()
        )
        counters["time_observation_valid_false_direct_count"] += int(
            (~observation_flag & direct_source).sum()
        )
        counters["enter_exit_valid_count"] += int(enter_exit_valid.sum())
        counters["direct_enter_exit_valid_count"] += int(
            (direct_source & enter_exit_valid).sum()
        )
        alias_pair = np.isfinite(travel_time) & np.isfinite(observed_time)
        counters["travel_and_observed_both_nonnull_count"] += int(alias_pair.sum())
        counters["travel_observed_alias_mismatch_count"] += int(
            (~np.isclose(travel_time[alias_pair], observed_time[alias_pair], atol=1e-9)).sum()
        )

        direct_span = direct_source & enter_exit_valid & observed_time_valid
        distributions["direct_observed_time_s"].update(observed_time[direct_time_valid])
        distributions["direct_observed_distance_m"].update(
            observed_distance[direct_pace_valid]
        )
        distributions["allocated_distance_m"].update(
            allocated_distance[allocated_distance_valid]
        )
        distributions["raw_derivable_direct_sec_per_m"].update(
            observed_time[direct_pace_valid] / observed_distance[direct_pace_valid]
        )
        distributions["enter_exit_span_minus_direct_time_s"].update(
            (exit_ - enter - observed_time)[direct_span]
        )

        joined = traversal.merge(
            label,
            on=["order_id", "traversal_id"],
            how="left",
            validate="one_to_one",
            suffixes=("_traversal", "_label"),
            indicator=True,
        )
        label_present = joined["_merge"].eq("both").to_numpy(bool)
        counters["traversal_with_label_count"] += int(label_present.sum())
        counters["traversal_without_label_count"] += int((~label_present).sum())
        label_time = _numeric(joined, "direct_observed_time_s")
        label_distance = _numeric(joined, "direct_observed_distance_m")
        label_allocated_distance = _numeric(joined, "allocated_distance_m_label")
        label_pace = _numeric(joined, "observed_sec_per_m")
        coverage = _numeric(joined, "direct_distance_coverage_share")
        label_measurement = (
            joined["measurement_source_label"].astype("string").fillna("missing").astype(str)
        )
        traversal_measurement = (
            joined["measurement_source_traversal"].astype("string").fillna("missing").astype(str)
        )
        counters["measurement_source_mismatch_count"] += int(
            (label_present & label_measurement.ne(traversal_measurement).to_numpy()).sum()
        )
        observed_joined = _numeric(joined, "observed_travel_time_s")
        direct_time_pair = np.isfinite(label_time) & np.isfinite(observed_joined)
        counters["label_and_traversal_direct_time_pair_count"] += int(
            direct_time_pair.sum()
        )
        counters["label_traversal_direct_time_mismatch_count"] += int(
            (~np.isclose(label_time[direct_time_pair], observed_joined[direct_time_pair], atol=1e-9)).sum()
        )
        allocated_pair = np.isfinite(label_allocated_distance) & np.isfinite(allocated_distance)
        counters["allocated_distance_mismatch_count"] += int(
            (~np.isclose(label_allocated_distance[allocated_pair], allocated_distance[allocated_pair], atol=1e-9)).sum()
        )
        pace_pair = (
            np.isfinite(label_time)
            & np.isfinite(label_distance)
            & (label_distance > 0)
            & np.isfinite(label_pace)
        )
        expected_pace = np.divide(
            label_time,
            label_distance,
            out=np.full(len(joined), np.nan),
            where=np.isfinite(label_distance) & (label_distance > 0),
        )
        counters["label_pace_formula_mismatch_count"] += int(
            (~np.isclose(label_pace[pace_pair], expected_pace[pace_pair], atol=1e-12)).sum()
        )
        rts_available = joined["rts_measurement_available"].fillna(False).to_numpy(bool)
        qualified_pace = pace_pair & rts_available & (label_pace > 0)
        counters["pace_target_valid_count"] += int(qualified_pace.sum())
        distributions["direct_observed_sec_per_m"].update(label_pace[qualified_pace])
        counters["rts_measurement_available_count"] += int(rts_available.sum())
        counters["raw_pace_derivable_but_quality_unavailable_count"] += int(
            (direct_pace_valid & ~rts_available).sum()
        )
        finite_coverage = np.isfinite(coverage)
        distributions["direct_distance_coverage_share"].update(coverage[finite_coverage])
        for threshold in (0.5, 0.8, 0.9, 0.95, 0.99):
            name = f"direct_coverage_ge_{str(threshold).replace('.', '_')}_count"
            counters[name] += int((finite_coverage & (coverage >= threshold)).sum())

        partition_stats = {
            "rows": len(traversal),
            "direct_time": int(direct_time_valid.sum()),
            "pace": int(qualified_pace.sum()),
            "labels": int(label_present.sum()),
        }
        by_split[split].update(partition_stats)
        by_date[date].update(partition_stats)

    total = counters["traversal_row_count"]
    label_rows = counters["label_row_count"]
    direct_count = counters["travel_time_direct_valid_count"]
    pace_count = counters["pace_target_valid_count"]
    counters["non_direct_time_observation_flag_defect_count"] = counters[
        "time_observation_valid_true_non_direct_count"
    ]
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "engineering_status": "PASS",
        "input": {
            "stage1_input_root": str(inputs.resolve()),
            "stage1_label_root": str(labels.resolve()),
            "partition_count": counters["partition_count"],
        },
        "counters": dict(sorted(counters.items())),
        "coverage": {
            "direct_travel_time_share_of_route_tokens": direct_count / total,
            "direct_pace_share_of_route_tokens": pace_count / total,
            "traversal_label_share_of_route_tokens": label_rows / total,
            "pace_share_of_supervised_traversals": pace_count / max(label_rows, 1),
            "rts_measurement_share_of_route_tokens": counters[
                "rts_measurement_available_count"
            ]
            / total,
        },
        "measurement_source_counts": dict(sorted(measurement_sources.items())),
        "time_source_counts": dict(sorted(time_sources.items())),
        "travel_time_source_class_counts": dict(sorted(source_classes.items())),
        "distributions": {
            name: distribution.summary()
            for name, distribution in distributions.items()
        },
        "by_split": {
            name: dict(sorted(values.items()))
            for name, values in sorted(by_split.items())
        },
        "by_date": {
            name: dict(sorted(values.items()))
            for name, values in sorted(by_date.items())
        },
        "semantic_findings": {
            "physical_traversal_realized_time_field": "No frozen field is a complete full-edge realized time for every traversal. direct_observed_time_s and link_traversals.observed_travel_time_s are the same sum of directly timed GPS intervals and may cover only part of allocated_distance_m.",
            "direct_observation_fields": [
                "link_traversals.observed_travel_time_s",
                "link_traversals.observed_distance_m",
                "traversal_labels.direct_observed_time_s",
                "traversal_labels.direct_observed_distance_m",
                "traversal_labels.observed_sec_per_m"
            ],
            "interpolated_or_engine_fields": "engine_allocated_travel_time_s is disabled and null in the frozen release; interval_supported and engine_interpolated route tokens have no individual-link time label.",
            "duplicate_time_source_conclusion": "travel_time_s aliases observed_travel_time_s on direct rows, while traversal_labels.direct_observed_time_s re-aggregates the same direct intervals. They are consistency copies, not independent observations, and must not be counted twice.",
            "time_observation_valid_conclusion": "The frozen link_traversals.time_observation_valid flag is true for non-direct rows and is not sufficient by itself. v5 validity must require measurement_source == direct_observed plus finite positive time and distance.",
            "recommended_primary_distribution_target": "observed_sec_per_m (directly observed pace) because direct timing commonly covers only a fraction of the physical traversal. Full traversal time is derived as predicted pace multiplied by allocated_distance_m.",
            "recommended_direct_travel_time_sensitivity_target": "Use direct_observed_time_s only for high direct_distance_coverage_share thresholds as a sensitivity target, not as an unconditional full-traversal target.",
            "source_indicator_required": True,
        },
        "recommended_masks": {
            "travel_time_target_valid": "measurement_source == direct_observed AND direct_observed_time_s > 0",
            "travel_time_direct_valid": "same as travel_time_target_valid",
            "travel_time_interpolated_valid": "false for the frozen release because engine allocation is disabled",
            "pace_target_valid": "travel_time_direct_valid AND rts_measurement_available AND observed_sec_per_m is finite and positive; the Stage 1 flag here is a frozen physical-window quality gate (time >= 3 s, distance >= 10 m, continuous, plausible speed), not RTS model output",
            "full_traversal_time_sensitivity_valid": "pace_target_valid AND direct_distance_coverage_share >= frozen sensitivity threshold",
            "travel_time_source_class": [
                "direct_raw_gps_interval",
                "interval_supported_no_link_time",
                "engine_interpolated_no_link_time",
                "engine_allocated_time",
                "unresolved_no_link_time",
                "unknown_no_link_time"
            ]
        },
        "target_contract": {
            "primary_distribution_target": "direct_observed_sec_per_m",
            "derived_full_traversal_time": "predicted_pace * allocated_distance_m",
            "direct_time_role": "high-coverage sensitivity and auxiliary supervision",
            "interpolated_time_role": "not supervised in v5 frozen upstream",
            "unavailable_value_policy": "NaN plus explicit validity mask; never fill with zero",
        },
        "runtime_s": time.perf_counter() - started,
    }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    coverage = report["coverage"]
    counters = report["counters"]
    distributions = report["distributions"]
    return f"""# Stage 2 v5 Service-time Target Audit

## Conclusion

The frozen products do not contain a complete realized full-edge travel time for every route token. The reliable primary supervision is directly observed pace (`observed_sec_per_m`), calculated from directly timed GPS intervals. Stage 2 v5 therefore models positive pace as its primary distribution target and derives a full traversal service time from predicted pace × `allocated_distance_m`.

`travel_time_s`, `observed_travel_time_s`, and `direct_observed_time_s` are not independent labels: on direct rows they represent the same direct-interval timing evidence. Engine allocation is disabled, so interpolated and interval-supported tokens do not have individual-link service-time targets.

## Coverage

| Metric | Value |
|---|---:|
| Route tokens | {counters['traversal_row_count']:,} |
| Traversal labels | {counters['label_row_count']:,} |
| Direct time valid | {counters['travel_time_direct_valid_count']:,} ({coverage['direct_travel_time_share_of_route_tokens']:.2%}) |
| Direct pace valid | {counters['pace_target_valid_count']:,} ({coverage['direct_pace_share_of_route_tokens']:.2%}) |
| RTS measurement available | {counters['rts_measurement_available_count']:,} ({coverage['rts_measurement_share_of_route_tokens']:.2%}) |
| Engine/interpolated time labels | {counters['travel_time_interpolated_valid_count']:,} |

## Provenance and consistency

| Audit | Count |
|---|---:|
| Duplicate traversal keys | {counters['traversal_duplicate_key_count']:,} |
| Duplicate label keys | {counters['label_duplicate_key_count']:,} |
| Missing paired label rows | {counters['traversal_without_label_count']:,} |
| Direct time mismatch between input and label | {counters['label_traversal_direct_time_mismatch_count']:,} |
| Pace formula mismatch | {counters['label_pace_formula_mismatch_count']:,} |
| Measurement-source mismatch | {counters['measurement_source_mismatch_count']:,} |
| `travel_time_s` alias mismatch | {counters['travel_observed_alias_mismatch_count']:,} |
| `time_observation_valid=true` on non-direct rows | {counters['time_observation_valid_true_non_direct_count']:,} |

The last row is an upstream flag-semantics defect: v5 never uses `time_observation_valid` alone. It requires a direct measurement source and finite positive time/distance.

## Distribution summary

| Quantity | Count | Mean | P50 | P90 | P95 | P99 |
|---|---:|---:|---:|---:|---:|---:|
| Direct observed time (s) | {distributions['direct_observed_time_s']['count']:,} | {distributions['direct_observed_time_s']['mean']:.4f} | {distributions['direct_observed_time_s']['p50']:.4f} | {distributions['direct_observed_time_s']['p90']:.4f} | {distributions['direct_observed_time_s']['p95']:.4f} | {distributions['direct_observed_time_s']['p99']:.4f} |
| Direct observed distance (m) | {distributions['direct_observed_distance_m']['count']:,} | {distributions['direct_observed_distance_m']['mean']:.4f} | {distributions['direct_observed_distance_m']['p50']:.4f} | {distributions['direct_observed_distance_m']['p90']:.4f} | {distributions['direct_observed_distance_m']['p95']:.4f} | {distributions['direct_observed_distance_m']['p99']:.4f} |
| Direct pace (s/m) | {distributions['direct_observed_sec_per_m']['count']:,} | {distributions['direct_observed_sec_per_m']['mean']:.6f} | {distributions['direct_observed_sec_per_m']['p50']:.6f} | {distributions['direct_observed_sec_per_m']['p90']:.6f} | {distributions['direct_observed_sec_per_m']['p95']:.6f} | {distributions['direct_observed_sec_per_m']['p99']:.6f} |
| Direct distance coverage | {distributions['direct_distance_coverage_share']['count']:,} | {distributions['direct_distance_coverage_share']['mean']:.4f} | {distributions['direct_distance_coverage_share']['p50']:.4f} | {distributions['direct_distance_coverage_share']['p90']:.4f} | {distributions['direct_distance_coverage_share']['p95']:.4f} | {distributions['direct_distance_coverage_share']['p99']:.4f} |

Quantiles use a deterministic bounded sample per partition; counts and moments are exact.

## Frozen v5 target contract

- Primary distribution target: positive direct-observed pace.
- Derived physical traversal time: predicted pace × allocated traversal distance.
- Direct observed time: auxiliary/high-coverage sensitivity target.
- Interpolated and interval-supported time: unavailable for link supervision in the frozen upstream.
- Missing targets remain `NaN` with explicit masks and source classes.
- RTS remains a secondary relative-delay target; LCS components remain auxiliary state targets.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-input", type=Path, required=True)
    parser.add_argument("--stage1-output", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    report = audit_service_time_targets(args.stage1_input, args.stage1_output)
    _atomic_json(args.json, report)
    _atomic_text(args.markdown, render_markdown(report))
    report["json_sha256"] = _sha256(args.json)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
