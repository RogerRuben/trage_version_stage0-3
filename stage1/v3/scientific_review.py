"""Streaming scientific diagnostics for frozen Stage 1 v3 outputs."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from .config import Stage1V3Config
from .input_adapter import iter_stage0_buckets
from .io import atomic_write_json, stage1_v3_code_identity
from .schema import ContractError, OUTPUT_BUCKET_SCHEMA_VERSION


class _Moments:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.minimum = float("inf")
        self.maximum = float("-inf")

    def update(self, values: pd.Series | np.ndarray) -> None:
        clean = pd.to_numeric(
            pd.Series(values), errors="coerce"
        ).to_numpy(dtype=np.float64)
        clean = clean[np.isfinite(clean)]
        if not clean.size:
            return
        self.count += int(clean.size)
        self.total += float(clean.sum())
        self.total_sq += float(np.square(clean).sum())
        self.minimum = min(self.minimum, float(clean.min()))
        self.maximum = max(self.maximum, float(clean.max()))

    def record(self) -> dict[str, Any]:
        if not self.count:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }
        mean = self.total / self.count
        variance = max(self.total_sq / self.count - mean * mean, 0.0)
        return {
            "count": self.count,
            "mean": mean,
            "std": float(np.sqrt(variance)),
            "min": self.minimum,
            "max": self.maximum,
        }


def _peak_rss_mb() -> float:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss / (1024 * 1024))
    except (ImportError, OSError):
        return float("nan")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def run_scientific_review(
    input_root: str | Path,
    output_root: str | Path,
    report_dir: str | Path,
    config: Stage1V3Config,
) -> dict[str, Any]:
    """Review all outputs without refitting or mutating any model/input."""

    started = time.perf_counter()
    output = Path(output_root)
    reports = Path(report_dir)
    reports.mkdir(parents=True, exist_ok=True)
    refs = list(iter_stage0_buckets(input_root, config))
    availability: Counter[tuple[str, str, str]] = Counter()
    missing: Counter[tuple[str, str, str]] = Counter()
    fallback: Counter[tuple[str, str]] = Counter()
    tail: Counter[tuple[str, str, str, str]] = Counter()
    coverage_bins: Counter[tuple[str, str, str]] = Counter()
    unseen: Counter[tuple[str, str]] = Counter()
    distance: Counter[tuple[str, str]] = Counter()
    distance_orders: dict[str, set[str]] = defaultdict(set)
    moments: dict[tuple[str, str], _Moments] = defaultdict(_Moments)
    crawl_cross: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "count": 0.0,
            "sum_crawl": 0.0,
            "sum_stop": 0.0,
            "sum_crawl_sq": 0.0,
            "sum_stop_sq": 0.0,
            "sum_product": 0.0,
            "maximum_sum": 0.0,
        }
    )
    peak_rss = _peak_rss_mb()
    bucket_count = 0
    interval_mutual_exclusion_failures = 0
    distance_label_failures = 0
    unseen_support_failures = 0
    canonical_highway_missing = 0
    acceleration_comparison_count = 0
    acceleration_comparison_difference_count = 0
    ordinary_acceleration_rms = _Moments()
    weighted_acceleration_rms_sample = _Moments()
    acceleration_rms_delta = _Moments()

    traversal_columns = [
        "order_id",
        "traversal_id",
        "crawl_time_share",
        "stop_time_share",
        "speed_cv_bounded",
        "acceleration_pair_count",
        "acceleration_weight_s",
        "acceleration_rms_mps2",
        "acceleration_rms_bounded",
        "maximum_absolute_acceleration_mps2",
        "direct_distance_exceeds_allocated",
        "lcs_available",
        "lcs_unavailable_reason",
        "lcs_raw",
        "lcs_pct",
        "lcs_tail_event",
        "rts_available",
        "rts_unavailable_reason",
        "rts_measurement_available",
        "rts_raw",
        "rts_pct",
        "rts_tail_event",
        "observed_sec_per_m",
        "reference_sec_per_m",
        "excess_time_ratio",
        "reference_level_used",
        "edge_hour_support_level",
        "edge_time_bin_30m_observation_count",
        "directed_edge_model_scope",
        "edge_observation_count",
        "edge_hour_observation_count",
        "canonical_highway",
        "synthetic_reverse_edge",
        "osm_direction_disagreement",
    ]
    order_columns = [
        "order_id",
        "lcs_available",
        "lcs_unavailable_reason",
        "lcs_mean",
        "lcs_tail_event_present",
        "rts_available",
        "rts_unavailable_reason",
        "rts_mean",
        "rts_tail_event_present",
    ]
    quality_columns = [
        "order_id",
        "observed_time_share",
        "observed_distance_share",
    ]

    for ref in refs:
        bucket_dir = (
            output
            / f"split={ref.split}"
            / f"date={ref.date}"
            / f"bucket={ref.bucket:05d}"
        )
        manifest_path = bucket_dir / "manifest.json"
        if not manifest_path.is_file():
            raise ContractError(f"scientific review missing bucket {bucket_dir}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != OUTPUT_BUCKET_SCHEMA_VERSION
            or manifest.get("engineering_status") != "PASS"
        ):
            raise ContractError(
                f"scientific review rejected bucket manifest {bucket_dir}"
            )
        traversals = pd.read_parquet(
            bucket_dir / "traversal_labels.parquet",
            columns=traversal_columns,
        )
        orders = pd.read_parquet(
            bucket_dir / "order_labels.parquet", columns=order_columns
        )
        quality = pd.read_parquet(
            bucket_dir / "order_label_quality.parquet",
            columns=quality_columns,
        )
        intervals = pd.read_parquet(
            bucket_dir / "interval_labels.parquet",
            columns=[
                "order_id",
                "traversal_id",
                "acceleration_mps2",
                "is_stop",
                "is_crawl",
                "is_low_speed_total",
            ],
        )
        interval_mutual_exclusion_failures += int(
            (
                (intervals["is_stop"].eq(True) & intervals["is_crawl"].eq(True))
                | intervals["is_low_speed_total"]
                .eq(True)
                .ne(
                    intervals["is_stop"].eq(True)
                    | intervals["is_crawl"].eq(True)
                )
            ).sum()
        )
        if acceleration_comparison_count < 10_000:
            finite_acceleration = intervals.loc[
                np.isfinite(
                    pd.to_numeric(
                        intervals["acceleration_mps2"], errors="coerce"
                    )
                ),
                ["order_id", "traversal_id", "acceleration_mps2"],
            ].copy()
            finite_acceleration["acceleration_mps2"] = pd.to_numeric(
                finite_acceleration["acceleration_mps2"], errors="coerce"
            )
            ordinary = (
                finite_acceleration.assign(
                    _a2=np.square(
                        finite_acceleration["acceleration_mps2"]
                    )
                )
                .groupby(
                    ["order_id", "traversal_id"], sort=False
                )["_a2"]
                .mean()
                .pow(0.5)
                .rename("ordinary_rms")
                .reset_index()
            )
            sample_limit = 10_000 - acceleration_comparison_count
            ordinary = ordinary.head(sample_limit)
            comparison = ordinary.merge(
                traversals[
                    [
                        "order_id",
                        "traversal_id",
                        "acceleration_rms_mps2",
                    ]
                ],
                on=["order_id", "traversal_id"],
                how="inner",
                validate="one_to_one",
            ).dropna()
            old = comparison["ordinary_rms"]
            new = comparison["acceleration_rms_mps2"]
            ordinary_acceleration_rms.update(old)
            weighted_acceleration_rms_sample.update(new)
            acceleration_rms_delta.update(new - old)
            acceleration_comparison_count += len(comparison)
            acceleration_comparison_difference_count += int(
                (
                    ~np.isclose(
                        old, new, atol=1e-12, rtol=1e-12
                    )
                ).sum()
            )

        split = ref.split
        for dimension in ("lcs", "rts"):
            flags = traversals[f"{dimension}_available"].eq(True)
            availability[(split, dimension, "available")] += int(flags.sum())
            availability[(split, dimension, "unavailable")] += int(
                (~flags).sum()
            )
            for reason, count in traversals.loc[
                ~flags, f"{dimension}_unavailable_reason"
            ].fillna("<NULL>").astype(str).value_counts().items():
                missing[(split, dimension, str(reason))] += int(count)
            event = traversals[f"{dimension}_tail_event"].astype("boolean")
            tail[(split, "traversal", dimension, "unavailable")] += int(
                event.isna().sum()
            )
            tail[(split, "traversal", dimension, "available_no_tail")] += int(
                event.eq(False).sum()
            )
            tail[(split, "traversal", dimension, "available_tail")] += int(
                event.eq(True).sum()
            )
            order_event = orders[
                f"{dimension}_tail_event_present"
            ].astype("boolean")
            tail[(split, "order", dimension, "unavailable")] += int(
                order_event.isna().sum()
            )
            tail[(split, "order", dimension, "available_no_tail")] += int(
                order_event.eq(False).sum()
            )
            tail[(split, "order", dimension, "available_tail")] += int(
                order_event.eq(True).sum()
            )

        for name in (
            "crawl_time_share",
            "stop_time_share",
            "speed_cv_bounded",
            "acceleration_pair_count",
            "acceleration_weight_s",
            "acceleration_rms_mps2",
            "acceleration_rms_bounded",
            "maximum_absolute_acceleration_mps2",
            "lcs_raw",
            "lcs_pct",
            "rts_raw",
            "rts_pct",
            "observed_sec_per_m",
            "reference_sec_per_m",
            "excess_time_ratio",
            "edge_time_bin_30m_observation_count",
        ):
            moments[(split, name)].update(traversals[name])

        crawl = pd.to_numeric(
            traversals["crawl_time_share"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        stop = pd.to_numeric(
            traversals["stop_time_share"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        valid = np.isfinite(crawl) & np.isfinite(stop)
        if valid.any():
            cross = crawl_cross[split]
            x, y = crawl[valid], stop[valid]
            cross["count"] += int(valid.sum())
            cross["sum_crawl"] += float(x.sum())
            cross["sum_stop"] += float(y.sum())
            cross["sum_crawl_sq"] += float(np.square(x).sum())
            cross["sum_stop_sq"] += float(np.square(y).sum())
            cross["sum_product"] += float((x * y).sum())
            cross["maximum_sum"] = max(
                cross["maximum_sum"], float((x + y).max())
            )

        exceeds = traversals["direct_distance_exceeds_allocated"].eq(True)
        distance[(split, "traversal_count")] += int(exceeds.sum())
        distance[(split, "direct_traversal_count")] += len(traversals)
        distance_orders[split].update(
            traversals.loc[exceeds, "order_id"].astype(str)
        )
        distance_label_failures += int(
            (
                exceeds
                & (
                    traversals["lcs_available"].eq(True)
                    | traversals["rts_available"].eq(True)
                    | traversals["rts_measurement_available"].eq(True)
                )
            ).sum()
        )
        canonical_highway_missing += int(
            traversals["canonical_highway"].isna().sum()
        )
        for level, count in traversals[
            "edge_hour_support_level"
        ].fillna("<NULL>").astype(str).value_counts().items():
            fallback[(split, str(level))] += int(count)
        evaluation_unseen = traversals[
            "directed_edge_model_scope"
        ].eq("evaluation_unseen")
        unseen[(split, "traversal_count")] += int(evaluation_unseen.sum())
        nonzero_edge_support = int(
            (
                evaluation_unseen
                & pd.to_numeric(
                    traversals["edge_observation_count"], errors="coerce"
                ).ne(0)
            ).sum()
        )
        nonzero_edge_hour_support = int(
            (
                evaluation_unseen
                & pd.to_numeric(
                    traversals["edge_hour_observation_count"], errors="coerce"
                ).ne(0)
            ).sum()
        )
        unseen[(split, "nonzero_edge_support_count")] += nonzero_edge_support
        unseen[
            (split, "nonzero_edge_hour_support_count")
        ] += nonzero_edge_hour_support
        unseen_support_failures += (
            nonzero_edge_support + nonzero_edge_hour_support
        )

        for name in ("observed_time_share", "observed_distance_share"):
            values = pd.to_numeric(quality[name], errors="coerce")
            bins = pd.cut(
                values,
                bins=[-np.inf, 0.25, 0.5, 0.75, 1.0, np.inf],
                labels=["<=0.25", "(0.25,0.5]", "(0.5,0.75]", "(0.75,1]", ">1"],
            )
            for label, count in bins.value_counts(sort=False).items():
                coverage_bins[(split, name, str(label))] += int(count)
        bucket_count += 1
        peak_rss = max(peak_rss, _peak_rss_mb())

    availability_rows = [
        {
            "split": split,
            "dimension": dimension,
            "status": status,
            "count": count,
        }
        for (split, dimension, status), count in sorted(availability.items())
    ]
    missing_rows = [
        {
            "split": split,
            "dimension": dimension,
            "reason": reason,
            "count": count,
        }
        for (split, dimension, reason), count in sorted(missing.items())
    ]
    component_names = {
        "crawl_time_share",
        "stop_time_share",
        "speed_cv_bounded",
        "acceleration_pair_count",
        "acceleration_weight_s",
        "acceleration_rms_mps2",
        "acceleration_rms_bounded",
        "maximum_absolute_acceleration_mps2",
        "lcs_raw",
        "lcs_pct",
    }
    lcs_rows = [
        {"split": split, "metric": metric, **summary.record()}
        for (split, metric), summary in sorted(moments.items())
        if metric in component_names
    ]
    rts_names = {
        "rts_raw",
        "rts_pct",
        "observed_sec_per_m",
        "reference_sec_per_m",
        "excess_time_ratio",
    }
    rts_rows = [
        {"split": split, "metric": metric, **summary.record()}
        for (split, metric), summary in sorted(moments.items())
        if metric in rts_names
    ]
    distance_rows = []
    for split in ("train", "validation", "test"):
        affected = int(distance[(split, "traversal_count")])
        total = int(distance[(split, "direct_traversal_count")])
        distance_rows.append(
            {
                "split": split,
                "affected_traversal_count": affected,
                "affected_order_count": len(distance_orders[split]),
                "direct_traversal_count": total,
                "affected_traversal_share": affected / total if total else None,
            }
        )
    fallback_totals = Counter()
    for (split, _level), count in fallback.items():
        fallback_totals[split] += count
    fallback_rows = [
        {
            "split": split,
            "record_type": "fallback_level",
            "fallback_level": level,
            "count": count,
            "share": (
                count / fallback_totals[split]
                if fallback_totals[split]
                else None
            ),
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
        for (split, level), count in sorted(fallback.items())
    ]
    fallback_rows.extend(
        {
            "split": split,
            "record_type": "edge_time_bin_30m_distribution",
            "fallback_level": None,
            "count": record["count"],
            "share": None,
            "mean": record["mean"],
            "std": record["std"],
            "min": record["min"],
            "max": record["max"],
        }
        for (split, metric), summary in sorted(moments.items())
        if metric == "edge_time_bin_30m_observation_count"
        for record in [summary.record()]
    )
    tail_rows = [
        {
            "split": split,
            "unit": unit,
            "dimension": dimension,
            "status": status,
            "count": count,
        }
        for (split, unit, dimension, status), count in sorted(tail.items())
    ]
    unseen_rows = [
        {"split": split, "metric": metric, "count": count}
        for (split, metric), count in sorted(unseen.items())
    ]
    drift_rows = [
        {"split": split, "metric": metric, **summary.record()}
        for (split, metric), summary in sorted(moments.items())
        if metric in {"lcs_raw", "lcs_pct", "rts_raw", "rts_pct"}
    ]
    coverage_rows = [
        {
            "split": split,
            "metric": metric,
            "coverage_bin": coverage_bin,
            "count": count,
        }
        for (split, metric, coverage_bin), count in sorted(
            coverage_bins.items()
        )
    ]
    csv_payloads = {
        "availability_by_split.csv": availability_rows,
        "missing_reason_counts.csv": missing_rows,
        "lcs_component_distribution.csv": lcs_rows,
        "rts_distribution.csv": rts_rows,
        "distance_exceed_audit.csv": distance_rows,
        "support_fallback_counts.csv": fallback_rows,
        "tail_event_status_counts.csv": tail_rows,
        "unseen_edge_audit.csv": unseen_rows,
        "temporal_drift_metrics.csv": drift_rows,
        "order_coverage_bins.csv": coverage_rows,
    }
    for name, rows in csv_payloads.items():
        _write_csv(reports / name, rows)

    crawl_stop = {}
    for split, values in sorted(crawl_cross.items()):
        count = values["count"]
        numerator = (
            count * values["sum_product"]
            - values["sum_crawl"] * values["sum_stop"]
        )
        denominator = np.sqrt(
            max(
                count * values["sum_crawl_sq"]
                - values["sum_crawl"] ** 2,
                0.0,
            )
            * max(
                count * values["sum_stop_sq"]
                - values["sum_stop"] ** 2,
                0.0,
            )
        )
        crawl_stop[split] = {
            "count": int(count),
            "crawl_plus_stop_max": values["maximum_sum"],
            "correlation": (
                float(numerator / denominator) if denominator > 0 else None
            ),
        }

    runtime_s = time.perf_counter() - started
    engineering_checks = {
        "interval_crawl_stop_mutual_exclusion_failure_count": (
            interval_mutual_exclusion_failures
        ),
        "distance_exceed_label_failure_count": distance_label_failures,
        "evaluation_unseen_nonzero_support_failure_count": (
            unseen_support_failures
        ),
        "canonical_highway_missing_count": canonical_highway_missing,
    }
    review = {
        "schema_version": "stage1_v3_scientific_review.1",
        "engineering_status": (
            "PASS"
            if not any(engineering_checks.values())
            else "FAIL"
        ),
        "scientific_status": (
            "REVIEW_COMPLETED_BASELINE_NOT_CAUSALLY_VALIDATED"
        ),
        "config_sha": config.digest,
        "stage1_code_sha": stage1_v3_code_identity(),
        "bucket_count": bucket_count,
        "runtime_s": runtime_s,
        "peak_rss_mb": peak_rss,
        "engineering_checks": engineering_checks,
        "crawl_stop_by_split": crawl_stop,
        "acceleration_rms_comparison": {
            "sample_count": acceleration_comparison_count,
            "different_from_ordinary_count": (
                acceleration_comparison_difference_count
            ),
            "ordinary_rms": ordinary_acceleration_rms.record(),
            "weighted_rms": weighted_acceleration_rms_sample.record(),
            "weighted_minus_ordinary": acceleration_rms_delta.record(),
        },
        "distance_exceed_by_split": distance_rows,
        "support_fallback_counts": fallback_rows,
        "tail_event_status_counts": tail_rows,
        "unseen_edge_audit": unseen_rows,
        "statistical_attachments": sorted(csv_payloads),
        "nonblocking_limitations": [
            "LCS scalar is an equal-weight baseline, not a unique ground truth",
            "ordinary temporal drift does not invalidate held-out evaluation",
            "IIS, PMIS, and dynamic GNS scalar labels remain unavailable",
            "order labels are auxiliary summaries; traversal is the primary unit",
        ],
    }
    atomic_write_json(reports / "stage1_v3_scientific_review.json", review)
    markdown = [
        "# Stage 1 v3 Scientific Review",
        "",
        f"- Engineering status: `{review['engineering_status']}`",
        f"- Scientific status: `{review['scientific_status']}`",
        f"- Buckets reviewed: {bucket_count}",
        f"- Runtime: {runtime_s:.3f} s",
        f"- Peak RSS: {peak_rss:.1f} MB",
        "",
        "The review is descriptive. It validates formula semantics and reports "
        "distribution shift; it does not claim causal or predictive validity.",
        "",
        "## Formula audits",
        "",
        f"- Crawl/stop interval overlap failures: {interval_mutual_exclusion_failures}",
        f"- Distance-exceed label failures: {distance_label_failures}",
        f"- Evaluation-unseen support leakage failures: {unseen_support_failures}",
        f"- Missing canonical highway values: {canonical_highway_missing}",
        "",
        "Detailed counts and distributions are in the ten CSV attachments and "
        "`stage1_v3_scientific_review.json`.",
        "",
    ]
    (reports / "stage1_v3_scientific_review.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    return review
