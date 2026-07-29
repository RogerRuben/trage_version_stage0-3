"""Visit-aware traversal-to-order aggregation for Stage 1 v3."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .schema import ContractError

if TYPE_CHECKING:
    from .config import Stage1V3Config


DIMENSIONS = ("lcs", "rts", "gns", "iis", "pmis")
CORE_IDENTITY_DIMENSIONS = ("lcs", "rts")
DIMENSION_STATISTICS = (
    "mean",
    "max",
    "tail",
    "persistence",
    "coverage_share",
    "available",
    "unavailable_reason",
)
ORDER_LABEL_COLUMNS = (
    "split",
    "date",
    "order_id",
    *(
        f"{dimension}_{statistic}"
        for dimension in DIMENSIONS
        for statistic in DIMENSION_STATISTICS
    ),
    "all_dimension_mask",
    "valid_core_dimension_count",
    "core_composition_signature",
    "composition_signature",
    "core_composite_status",
)
ORDER_QUALITY_COLUMNS = (
    "split",
    "date",
    "order_id",
    "direct_interval_count",
    "unique_timed_edge_count",
    "direct_observed_time_s",
    "direct_observed_distance_m",
    "order_duration_s",
    "route_distance_m",
    "observed_time_share",
    "observed_distance_share",
    "observed_time_exceeds_order_duration",
    "observed_distance_exceeds_route_distance",
    "direct_coverage_pass",
    "lcs_missing_reason",
    "rts_missing_reason",
    "gns_missing_reason",
    *(f"{dimension}_coverage_share" for dimension in DIMENSIONS),
)


def _weighted_summary(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    tail_threshold: float,
) -> dict[str, float]:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return {
            "mean": float("nan"),
            "max": float("nan"),
            "tail": float("nan"),
            "persistence": float("nan"),
        }
    clean_values = values[valid]
    clean_weights = weights[valid]
    high = clean_values >= tail_threshold
    return {
        "mean": float(np.average(clean_values, weights=clean_weights)),
        "max": float(clean_values.max()),
        "tail": (
            float(np.average(clean_values[high], weights=clean_weights[high]))
            if high.any()
            else float("nan")
        ),
        "persistence": float(clean_weights[high].sum() / clean_weights.sum()),
    }


def _require_unique(frame: pd.DataFrame, keys: list[str], product: str) -> None:
    missing = sorted(set(keys) - set(frame.columns))
    if missing:
        raise ContractError(f"{product} missing key columns: {missing}")
    if frame.duplicated(keys, keep=False).any():
        raise ContractError(f"duplicate {product} key: {keys}")


def _availability(
    group: pd.DataFrame,
    dimension: str,
) -> np.ndarray:
    value_column = f"{dimension}_pct"
    if value_column not in group:
        return np.zeros(len(group), dtype=bool)
    values = pd.to_numeric(group[value_column], errors="coerce").to_numpy(
        dtype=np.float64
    )
    flag_column = f"{dimension}_available"
    flags = (
        group[flag_column].eq(True).to_numpy(dtype=bool)
        if flag_column in group
        else np.isfinite(values)
    )
    return flags & np.isfinite(values)


def _dimension_reason(
    *,
    dimension: str,
    available: bool,
    order_coverage_pass: bool,
    unavailable_reasons: dict[str, str],
) -> str:
    if available:
        return ""
    if dimension in unavailable_reasons:
        return unavailable_reasons[dimension]
    if not order_coverage_pass:
        return "ORDER_DIRECT_COVERAGE_BELOW_THRESHOLD"
    return f"NO_AVAILABLE_{dimension.upper()}_TRAVERSAL"


def aggregate_order_labels(
    traversal_labels: pd.DataFrame,
    route_parts: pd.DataFrame,
    link_traversals: pd.DataFrame,
    order_base: pd.DataFrame,
    config: "Stage1V3Config",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate labels without zero-imputing missing dynamic evidence.

    No cross-dimension composite is emitted in the v3 review candidate.  LCS
    and RTS retain independent availability and coverage semantics; GNS remains
    unavailable pending a separately frozen static extension.
    """

    _require_unique(order_base, ["order_id"], "order_base")
    _require_unique(
        traversal_labels,
        ["order_id", "traversal_id"],
        "traversal_labels",
    )
    _require_unique(route_parts, ["order_id", "route_sequence"], "route_parts")
    _require_unique(
        link_traversals,
        ["order_id", "traversal_id"],
        "link_traversals",
    )
    required_order = {
        "order_id",
        "split",
        "date",
        "departure_time",
        "arrival_time",
        "stage1_core_eligible",
    }
    missing_order = sorted(required_order - set(order_base.columns))
    if missing_order:
        raise ContractError(f"order_base missing aggregation columns: {missing_order}")
    required_traversal = {
        "order_id",
        "traversal_id",
        "route_sequence",
        "canonical_edge_uid",
        "observed_directed_edge_uid",
        "direct_interval_count",
        "direct_observed_time_s",
        "direct_observed_distance_m",
        *{
            f"{dimension}_{suffix}"
            for dimension in DIMENSIONS
            for suffix in ("raw", "pct", "available", "unavailable_reason")
        },
    }
    missing_traversal = sorted(required_traversal - set(traversal_labels.columns))
    if missing_traversal:
        raise ContractError(
            f"traversal_labels missing aggregation columns: {missing_traversal}"
        )
    if "length_m" not in route_parts:
        raise ContractError("route_parts missing length_m")
    required_link_traversal = {
        "order_id",
        "traversal_id",
        "route_sequence",
        "canonical_edge_uid",
        "observed_directed_edge_uid",
        "measurement_source",
    }
    missing_link_traversal = sorted(
        required_link_traversal - set(link_traversals.columns)
    )
    if missing_link_traversal:
        raise ContractError(
            "link_traversals missing aggregation columns: "
            f"{missing_link_traversal}"
        )

    known_orders = set(order_base["order_id"].astype(str))
    traversal_orders = set(traversal_labels["order_id"].astype(str))
    route_orders = set(route_parts["order_id"].astype(str))
    orphan = sorted((traversal_orders | route_orders) - known_orders)
    if orphan:
        raise ContractError(f"orphan order_id values in Stage1 labels: {orphan[:5]}")
    if route_orders != known_orders:
        raise ContractError("route_parts must contain every core order exactly")
    if traversal_orders != known_orders:
        raise ContractError("traversal_labels must contain every core order")
    eligible = order_base["stage1_core_eligible"]
    if (
        not eligible.map(lambda value: isinstance(value, (bool, np.bool_))).all()
        or not eligible.eq(True).all()
    ):
        raise ContractError("Stage1 v3 input contains non-core-eligible orders")

    direct_traversals = link_traversals.loc[
        link_traversals["measurement_source"].eq("direct_observed"),
        [
            "order_id",
            "traversal_id",
            "route_sequence",
            "canonical_edge_uid",
            "observed_directed_edge_uid",
        ],
    ].copy()
    label_identity = traversal_labels[
        [
            "order_id",
            "traversal_id",
            "route_sequence",
            "canonical_edge_uid",
            "observed_directed_edge_uid",
        ]
    ].copy()
    identity = label_identity.merge(
        direct_traversals,
        on=["order_id", "traversal_id"],
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_label", "_input"),
    )
    if not identity["_merge"].eq("both").all():
        raise ContractError(
            "traversal_labels do not exactly match direct link_traversals"
        )
    for column in (
        "route_sequence",
        "canonical_edge_uid",
        "observed_directed_edge_uid",
    ):
        if not identity[f"{column}_label"].astype(str).eq(
            identity[f"{column}_input"].astype(str)
        ).all():
            raise ContractError(
                f"traversal label identity differs on {column}"
            )

    for unavailable in ("gns", "iis", "pmis"):
        for suffix in ("raw", "pct"):
            name = f"{unavailable}_{suffix}"
            if name in traversal_labels and pd.to_numeric(
                traversal_labels[name], errors="coerce"
            ).notna().any():
                raise ContractError(
                    f"{unavailable.upper()} must remain unavailable in Stage1 v3"
                )
        flag = f"{unavailable}_available"
        if flag in traversal_labels:
            values = traversal_labels[flag]
            if not values.map(
                lambda value: isinstance(value, (bool, np.bool_))
            ).all():
                raise ContractError(
                    f"{flag} must contain only non-null booleans"
                )
            if values.eq(True).any():
                raise ContractError(
                    f"{unavailable.upper()} must remain unavailable in this pipeline"
                )

    for dimension in ("lcs", "rts"):
        required = {
            f"{dimension}_raw",
            f"{dimension}_pct",
            f"{dimension}_available",
            f"{dimension}_unavailable_reason",
        }
        missing = sorted(required - set(traversal_labels.columns))
        if missing:
            raise ContractError(
                f"traversal_labels missing {dimension} fields: {missing}"
            )
        flags = traversal_labels[f"{dimension}_available"]
        if not flags.map(
            lambda value: isinstance(value, (bool, np.bool_))
        ).all():
            raise ContractError(
                f"{dimension}_available must contain only non-null booleans"
            )
        raw = pd.to_numeric(
            traversal_labels[f"{dimension}_raw"], errors="coerce"
        )
        pct = pd.to_numeric(
            traversal_labels[f"{dimension}_pct"], errors="coerce"
        )
        available = flags.eq(True)
        if (
            (available & (~np.isfinite(raw) | ~np.isfinite(pct))).any()
            or (~available & (raw.notna() | pct.notna())).any()
        ):
            raise ContractError(
                f"{dimension} availability and numeric values are inconsistent"
            )
        finite_pct = pct[np.isfinite(pct)]
        if (finite_pct.lt(0.0) | finite_pct.gt(1.0)).any():
            raise ContractError(f"{dimension}_pct must be in [0, 1]")

    for column in (
        "direct_observed_time_s",
        "direct_observed_distance_m",
        "direct_interval_count",
    ):
        values = pd.to_numeric(traversal_labels[column], errors="coerce")
        if (~np.isfinite(values) | values.lt(0)).any():
            raise ContractError(
                f"traversal_labels.{column} must be finite and non-negative"
            )
    interval_values = pd.to_numeric(
        traversal_labels["direct_interval_count"], errors="coerce"
    )
    if interval_values.ne(np.floor(interval_values)).any():
        raise ContractError("direct_interval_count must be integral")

    route_lengths = pd.to_numeric(route_parts["length_m"], errors="coerce")
    if (~np.isfinite(route_lengths) | route_lengths.lt(0)).any():
        raise ContractError("route_parts.length_m must be finite and non-negative")

    coverage_cfg = config.section("coverage")
    minimum_intervals = int(coverage_cfg["minimum_direct_interval_count"])
    minimum_edges = int(coverage_cfg["minimum_unique_timed_edge_count"])
    minimum_time_share = float(coverage_cfg["minimum_observed_time_share"])
    minimum_distance_share = float(
        coverage_cfg["minimum_observed_distance_share"]
    )
    aggregation_cfg = config.section("aggregation")
    tail_threshold = float(aggregation_cfg["tail_percentile_threshold"])
    if not 0 <= tail_threshold <= 1:
        raise ContractError("aggregation.tail_percentile_threshold must be in [0, 1]")

    unavailable_reasons = {
        "gns": "EDGE_STATIC_FEATURE_EXTENSION_NOT_FITTED",
        "iis": str(config.section("iis")["unavailable_reason"]),
        "pmis": str(config.section("pmis")["unavailable_reason"]),
    }

    route_distance_by_order = (
        route_parts.assign(
            _order_id_key=route_parts["order_id"].astype(str),
            _route_distance=route_lengths
        )
        .groupby("_order_id_key", sort=False)["_route_distance"]
        .sum(min_count=1)
        .to_dict()
    )
    labels_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []

    grouped_labels = {
        str(order_id): group
        for order_id, group in traversal_labels.groupby(
            "order_id", sort=False, dropna=False
        )
    }
    for order in order_base.itertuples(index=False):
        order_id = str(order.order_id)
        group = grouped_labels.get(order_id, traversal_labels.iloc[0:0])
        direct_time = pd.to_numeric(
            group.get(
                "direct_observed_time_s", pd.Series(index=group.index, dtype=float)
            ),
            errors="coerce",
        ).fillna(0.0)
        direct_distance = pd.to_numeric(
            group.get(
                "direct_observed_distance_m",
                pd.Series(index=group.index, dtype=float),
            ),
            errors="coerce",
        ).fillna(0.0)
        interval_count = int(
            pd.to_numeric(
                group.get(
                    "direct_interval_count",
                    pd.Series(index=group.index, dtype=float),
                ),
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )
        timed_edges = int(
            group.loc[
                direct_time.gt(0)
                & group["observed_directed_edge_uid"].notna(),
                "observed_directed_edge_uid",
            ].nunique()
        ) if len(group) else 0
        observed_time_s = float(direct_time.sum())
        observed_distance_m = float(direct_distance.sum())
        departure_time = float(order.departure_time)
        arrival_time = float(order.arrival_time)
        if (
            not np.isfinite(departure_time)
            or not np.isfinite(arrival_time)
            or arrival_time <= departure_time
        ):
            raise ContractError("order_base timestamps must be finite and increasing")
        order_duration_s = arrival_time - departure_time
        route_distance_m = float(route_distance_by_order.get(order_id, np.nan))
        if not np.isfinite(route_distance_m) or route_distance_m <= 0:
            raise ContractError("each core order must have positive route distance")
        time_tolerance = float(
            config.section("direct")["duration_tolerance_s"]
        )
        observed_time_exceeds = (
            observed_time_s > order_duration_s + time_tolerance
        )
        if observed_time_exceeds:
            raise ContractError(
                f"direct observed time exceeds order duration for {order_id}"
            )
        distance_tolerance = float(
            config.section("direct")["distance_identity_tolerance_m"]
        )
        observed_distance_exceeds = (
            observed_distance_m > route_distance_m + distance_tolerance
        )
        time_share = (
            observed_time_s / order_duration_s
            if order_duration_s > 0
            else float("nan")
        )
        distance_share = (
            observed_distance_m / route_distance_m
            if np.isfinite(route_distance_m) and route_distance_m > 0
            else float("nan")
        )
        coverage_pass = (
            interval_count >= minimum_intervals
            and timed_edges >= minimum_edges
            and np.isfinite(time_share)
            and time_share >= minimum_time_share
            and np.isfinite(distance_share)
            and distance_share >= minimum_distance_share
        )

        label_row: dict[str, Any] = {
            "split": str(order.split),
            "date": str(order.date),
            "order_id": order_id,
        }
        mask: dict[str, bool] = {}
        dimension_coverage: dict[str, float] = {}
        for dimension in DIMENSIONS:
            valid = _availability(group, dimension)
            if dimension == "lcs":
                weights = direct_time.to_numpy(dtype=np.float64)
                denominator = order_duration_s
                valid_weight = float(weights[valid].sum()) if valid.any() else 0.0
            elif dimension == "rts":
                weights = direct_distance.to_numpy(dtype=np.float64)
                denominator = route_distance_m
                valid_weight = float(weights[valid].sum()) if valid.any() else 0.0
            else:
                weights = np.zeros(len(group), dtype=np.float64)
                denominator = float("nan")
                valid_weight = 0.0

            available = bool(valid.any()) and coverage_pass
            if dimension in {"gns", "iis", "pmis"}:
                available = False
            values = (
                    pd.to_numeric(group[f"{dimension}_pct"], errors="coerce").to_numpy(
                        dtype=np.float64,
                        copy=True,
                    )
                if f"{dimension}_pct" in group
                else np.full(len(group), np.nan, dtype=np.float64)
            )
            if not available:
                values[:] = np.nan
            summary = _weighted_summary(
                values,
                weights,
                tail_threshold=tail_threshold,
            )
            for statistic, value in summary.items():
                label_row[f"{dimension}_{statistic}"] = value
            coverage_share = (
                valid_weight / denominator
                if np.isfinite(denominator) and denominator > 0
                else float("nan")
            )
            label_row[f"{dimension}_coverage_share"] = coverage_share
            label_row[f"{dimension}_available"] = available
            label_row[f"{dimension}_unavailable_reason"] = _dimension_reason(
                dimension=dimension,
                available=available,
                order_coverage_pass=coverage_pass,
                unavailable_reasons=unavailable_reasons,
            )
            mask[dimension] = available
            dimension_coverage[dimension] = coverage_share

        core_available = [
            dimension for dimension in CORE_IDENTITY_DIMENSIONS if mask[dimension]
        ]
        all_available = [dimension for dimension in DIMENSIONS if mask[dimension]]
        label_row["all_dimension_mask"] = json.dumps(
            mask, sort_keys=True, separators=(",", ":")
        )
        label_row["valid_core_dimension_count"] = len(core_available)
        label_row["core_composition_signature"] = (
            "+".join(core_available) if core_available else "NONE"
        )
        label_row["composition_signature"] = (
            "+".join(all_available) if all_available else "NONE"
        )
        label_row["core_composite_status"] = str(
            config.data["core_composite_status"]
        )
        labels_rows.append(label_row)

        quality_rows.append(
            {
                "split": str(order.split),
                "date": str(order.date),
                "order_id": order_id,
                "direct_interval_count": interval_count,
                "unique_timed_edge_count": timed_edges,
                "direct_observed_time_s": observed_time_s,
                "direct_observed_distance_m": observed_distance_m,
                "order_duration_s": order_duration_s,
                "route_distance_m": route_distance_m,
                "observed_time_share": time_share,
                "observed_distance_share": distance_share,
                "observed_time_exceeds_order_duration": observed_time_exceeds,
                "observed_distance_exceeds_route_distance": (
                    observed_distance_exceeds
                ),
                "direct_coverage_pass": coverage_pass,
                "lcs_missing_reason": label_row["lcs_unavailable_reason"],
                "rts_missing_reason": label_row["rts_unavailable_reason"],
                "gns_missing_reason": label_row["gns_unavailable_reason"],
                **{
                    f"{dimension}_coverage_share": value
                    for dimension, value in dimension_coverage.items()
                },
            }
        )
    return (
        pd.DataFrame(labels_rows).reindex(columns=ORDER_LABEL_COLUMNS),
        pd.DataFrame(quality_rows).reindex(columns=ORDER_QUALITY_COLUMNS),
    )
