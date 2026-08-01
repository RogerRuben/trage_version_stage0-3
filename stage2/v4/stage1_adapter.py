"""Stage 1 v3 adapter and route/traversal alignment audit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import (
    COMPONENT_MASKS,
    DECISION_TIME_SOURCE,
    ORDER_PRIMARY_KEY,
    ORDER_REQUIRED_COLUMNS,
    PHYSICAL_TRAVERSAL_REQUIRED_COLUMNS,
    ROUTE_JOIN_KEY,
    ROUTE_PRIMARY_KEY,
    ROUTE_REQUIRED_COLUMNS,
    TRAVERSAL_PRIMARY_KEY,
    TRAVERSAL_REQUIRED_COLUMNS,
    Stage2V4ContractError,
    require_columns,
)


@dataclass(frozen=True, order=True)
class Stage1BucketRef:
    split: str
    date: str
    bucket: int
    output_path: Path
    input_path: Path

    @property
    def identity(self) -> str:
        return f"{self.split}/{self.date}/{self.bucket:05d}"


@dataclass
class AlignmentResult:
    route_tokens: pd.DataFrame
    traversal_alignment: pd.DataFrame
    counters: dict[str, int]


def discover_stage1_buckets(
    output_root: str | Path,
    input_root: str | Path,
) -> tuple[Stage1BucketRef, ...]:
    output = Path(output_root)
    source_input = Path(input_root)
    refs: list[Stage1BucketRef] = []
    for manifest in sorted(output.glob("split=*/date=*/bucket=*/manifest.json")):
        parts = {
            part.split("=", 1)[0]: part.split("=", 1)[1]
            for part in manifest.parent.relative_to(output).parts
        }
        try:
            split = parts["split"]
            date = parts["date"]
            bucket = int(parts["bucket"])
        except (KeyError, ValueError) as exc:
            raise Stage2V4ContractError(
                f"invalid Stage 1 bucket partition: {manifest.parent}"
            ) from exc
        input_path = (
            source_input
            / f"split={split}"
            / f"date={date}"
            / f"bucket={bucket:05d}"
        )
        if not input_path.is_dir():
            raise Stage2V4ContractError(
                f"Stage 1 input bucket is missing for {split}/{date}/{bucket:05d}"
            )
        refs.append(
            Stage1BucketRef(
                split=split,
                date=date,
                bucket=bucket,
                output_path=manifest.parent,
                input_path=input_path,
            )
        )
    return tuple(refs)


def _require_unique(frame: pd.DataFrame, key: tuple[str, ...], product: str) -> None:
    duplicate_count = int(frame.duplicated(list(key), keep=False).sum())
    if duplicate_count:
        raise Stage2V4ContractError(
            f"{product} has {duplicate_count} rows on duplicated primary keys {key}"
        )


def _require_partition(
    frame: pd.DataFrame,
    ref: Stage1BucketRef,
    product: str,
) -> None:
    if not frame["split"].astype(str).eq(ref.split).all():
        raise Stage2V4ContractError(f"{product} split disagrees with {ref.identity}")
    if not frame["date"].astype(str).eq(ref.date).all():
        raise Stage2V4ContractError(f"{product} date disagrees with {ref.identity}")


def _finite_unit_interval(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))
        & numeric.between(0.0, 1.0, inclusive="both").fillna(False).to_numpy(),
        index=series.index,
        dtype=bool,
    )


def _add_component_masks(frame: pd.DataFrame) -> None:
    has_direct = (
        pd.to_numeric(frame["direct_interval_count"], errors="coerce").fillna(0).gt(0)
        & pd.to_numeric(frame["direct_observed_time_s"], errors="coerce").fillna(0).gt(0)
    )
    frame[COMPONENT_MASKS["crawl_time_share"]] = (
        _finite_unit_interval(frame["crawl_time_share"]) & has_direct
    )
    frame[COMPONENT_MASKS["stop_time_share"]] = (
        _finite_unit_interval(frame["stop_time_share"]) & has_direct
    )
    frame[COMPONENT_MASKS["speed_cv_bounded"]] = (
        _finite_unit_interval(frame["speed_cv_bounded"])
        & pd.to_numeric(frame["direct_interval_count"], errors="coerce").fillna(0).ge(2)
    )
    frame[COMPONENT_MASKS["acceleration_rms_bounded"]] = (
        _finite_unit_interval(frame["acceleration_rms_bounded"])
        & pd.to_numeric(frame["acceleration_pair_count"], errors="coerce").fillna(0).gt(0)
        & pd.to_numeric(frame["acceleration_weight_s"], errors="coerce").fillna(0).gt(0)
    )
    frame["lcs_target_valid"] = (
        frame["lcs_available"].fillna(False).astype(bool)
        & _finite_unit_interval(frame["lcs_raw"])
        & _finite_unit_interval(frame["lcs_pct"])
    )
    frame["rts_target_valid"] = (
        frame["rts_available"].fillna(False).astype(bool)
        & frame["rts_measurement_available"].fillna(False).astype(bool)
        & _finite_unit_interval(frame["rts_raw"])
        & _finite_unit_interval(frame["rts_pct"])
    )
    for target, mask in COMPONENT_MASKS.items():
        frame[target] = pd.to_numeric(frame[target], errors="coerce").where(frame[mask])
    for target in ("lcs_raw", "lcs_pct"):
        frame[target] = pd.to_numeric(frame[target], errors="coerce").where(
            frame["lcs_target_valid"]
        )
    for target in ("rts_raw", "rts_pct"):
        frame[target] = pd.to_numeric(frame[target], errors="coerce").where(
            frame["rts_target_valid"]
        )
    for tail, available in (
        ("lcs_tail_event", "lcs_target_valid"),
        ("rts_tail_event", "rts_target_valid"),
    ):
        values = frame[tail].astype("boolean")
        frame[tail] = values.where(frame[available], pd.NA)


def build_route_alignment(ref: Stage1BucketRef) -> AlignmentResult:
    """Left-join labels onto the complete route skeleton for one bucket."""

    route = pd.read_parquet(ref.output_path / "route_sequence_context.parquet")
    traversals = pd.read_parquet(ref.output_path / "traversal_labels.parquet")
    orders = pd.read_parquet(ref.input_path / "order_base.parquet")
    physical = pd.read_parquet(ref.input_path / "link_traversals.parquet")
    require_columns(route.columns, ROUTE_REQUIRED_COLUMNS, "route_sequence_context")
    require_columns(traversals.columns, TRAVERSAL_REQUIRED_COLUMNS, "traversal_labels")
    require_columns(orders.columns, ORDER_REQUIRED_COLUMNS, "order_base")
    require_columns(
        physical.columns,
        PHYSICAL_TRAVERSAL_REQUIRED_COLUMNS,
        "link_traversals",
    )
    _require_partition(route, ref, "route_sequence_context")
    _require_partition(traversals, ref, "traversal_labels")
    _require_partition(orders, ref, "order_base")
    _require_unique(route, ROUTE_PRIMARY_KEY, "route_sequence_context")
    _require_unique(traversals, TRAVERSAL_PRIMARY_KEY, "traversal_labels")
    _require_unique(orders, ORDER_PRIMARY_KEY, "order_base")
    _require_unique(traversals, ROUTE_JOIN_KEY, "traversal_labels route mapping")
    if not orders["stage1_core_eligible"].fillna(False).astype(bool).all():
        raise Stage2V4ContractError(
            f"{ref.identity} contains a non-core order in the frozen Stage 1 input"
        )

    physical = physical.loc[
        :,
        sorted(PHYSICAL_TRAVERSAL_REQUIRED_COLUMNS),
    ].copy()
    physical["split"] = ref.split
    physical["date"] = ref.date
    _require_unique(
        physical,
        ("split", "date", "order_id", "traversal_id"),
        "link_traversals",
    )
    _require_unique(physical, ROUTE_JOIN_KEY, "link_traversals route mapping")
    route = route.merge(
        physical,
        on=list(ROUTE_JOIN_KEY),
        how="left",
        validate="one_to_one",
    )
    if route["traversal_id"].isna().any():
        raise Stage2V4ContractError(
            f"{ref.identity} route token lacks a physical traversal identity"
        )
    if len(route) != len(physical):
        raise Stage2V4ContractError(
            f"{ref.identity} route/physical traversal row conservation failed"
        )
    label_identity = traversals.loc[
        :,
        [*ROUTE_JOIN_KEY, "traversal_id"],
    ].merge(
        physical.loc[:, [*ROUTE_JOIN_KEY, "traversal_id"]],
        on=list(ROUTE_JOIN_KEY),
        how="left",
        suffixes=("_label", "_physical"),
        validate="one_to_one",
    )
    if not label_identity["traversal_id_label"].eq(
        label_identity["traversal_id_physical"]
    ).all():
        raise Stage2V4ContractError(
            f"{ref.identity} label traversal_id disagrees with physical traversal"
        )

    route_keys = route.loc[:, list(ROUTE_JOIN_KEY)]
    orphan_probe = traversals.loc[:, list(ROUTE_JOIN_KEY)].merge(
        route_keys,
        on=list(ROUTE_JOIN_KEY),
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    orphan_count = int(orphan_probe["_merge"].ne("both").sum())
    if orphan_count:
        raise Stage2V4ContractError(
            f"{ref.identity} has {orphan_count} traversal labels outside the route skeleton"
        )

    label_columns = [
        column
        for column in traversals.columns
        if column not in route.columns
    ]
    joined = route.merge(
        traversals.loc[:, [*ROUTE_JOIN_KEY, *label_columns]],
        on=list(ROUTE_JOIN_KEY),
        how="left",
        validate="one_to_one",
        indicator="_label_join",
    )
    if len(joined) != len(route):
        raise Stage2V4ContractError(f"{ref.identity} route token conservation failed")
    joined["label_available"] = joined["_label_join"].eq("both")
    joined.drop(columns="_label_join", inplace=True)
    if int(joined["label_available"].sum()) != len(traversals):
        raise Stage2V4ContractError(f"{ref.identity} label row conservation failed")

    _add_component_masks(joined)
    order_context = orders.loc[
        :,
        [*ORDER_PRIMARY_KEY, "departure_time", "start_node", "end_node"],
    ].rename(columns={"departure_time": "decision_time"})
    decision_numeric = pd.to_numeric(order_context["decision_time"], errors="coerce")
    if not np.isfinite(decision_numeric.to_numpy(dtype=float, na_value=np.nan)).all():
        raise Stage2V4ContractError(f"{ref.identity} has missing decision_time")
    order_context["decision_time_source"] = DECISION_TIME_SOURCE
    joined = joined.merge(
        order_context,
        on=list(ORDER_PRIMARY_KEY),
        how="left",
        validate="many_to_one",
    )
    if joined["decision_time"].isna().any():
        raise Stage2V4ContractError(f"{ref.identity} route token lacks order context")

    labeled = joined["label_available"]
    span_start = joined["route_sequence"].where(labeled).astype("Int64")
    span_end = joined["route_sequence_end"].where(labeled).astype("Int64")
    alignment = joined.loc[
        labeled,
        [
            "split",
            "date",
            "order_id",
            "traversal_id",
            "route_sequence",
            "observed_directed_edge_uid",
        ],
    ].copy()
    alignment["traversal_span_start_sequence"] = span_start.loc[labeled].to_numpy()
    alignment["traversal_span_end_sequence"] = span_end.loc[labeled].to_numpy()
    alignment["traversal_span_length"] = (
        alignment["traversal_span_end_sequence"]
        - alignment["traversal_span_start_sequence"]
        + 1
    ).astype("int64")
    alignment["alignment_status"] = np.where(
        alignment["traversal_span_length"].eq(1),
        "one_to_one",
        "span",
    )
    alignment["traversal_id"] = alignment["traversal_id"].astype("int64")
    alignment["route_sequence"] = alignment["route_sequence"].astype("int64")

    counters = {
        "order_count": int(len(orders)),
        "route_token_count": int(len(route)),
        "traversal_label_count": int(len(traversals)),
        "unlabeled_route_token_count": int((~labeled).sum()),
        "orphan_traversal_label_count": orphan_count,
        "multi_token_traversal_count": int(
            alignment["traversal_span_length"].gt(1).sum()
        ),
        "maximum_traversal_span_length": int(
            alignment["traversal_span_length"].max()
            if len(alignment)
            else 0
        ),
        "decision_time_missing_count": 0,
        "self_order_history_candidate_count": int(
            (
                labeled
                & pd.to_numeric(
                    joined["observation_window_end_time"],
                    errors="coerce",
                ).lt(joined["decision_time"])
            ).sum()
        ),
        "route_token_conservation_error": 0,
        "label_row_conservation_error": 0,
        "crawl_target_valid_count": int(joined["crawl_target_valid"].sum()),
        "stop_target_valid_count": int(joined["stop_target_valid"].sum()),
        "speed_cv_target_valid_count": int(joined["speed_cv_target_valid"].sum()),
        "acceleration_rms_target_valid_count": int(
            joined["acceleration_rms_target_valid"].sum()
        ),
        "lcs_target_valid_count": int(joined["lcs_target_valid"].sum()),
        "rts_target_valid_count": int(joined["rts_target_valid"].sum()),
    }
    return AlignmentResult(
        route_tokens=joined,
        traversal_alignment=alignment,
        counters=counters,
    )
