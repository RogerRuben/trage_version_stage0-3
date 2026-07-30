"""Train-only directed-edge support and deterministic fallback routing."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import pandas as pd

from .schema import ContractError

if TYPE_CHECKING:
    from .config import Stage1V3Config


CATALOG_COLUMNS = (
    "observed_directed_edge_uid",
    "physical_edge_uid",
    "observed_from_node",
    "observed_to_node",
    "observed_direction",
    "canonical_length_m",
    "canonical_highway",
    "road_class",
    "bridge",
    "tunnel",
    "synthetic_reverse_edge",
    "osm_direction_disagreement",
    "upper_region_id",
)

COUNT_COLUMNS = ("scope", "key", "hour", "observation_count")


@dataclass(frozen=True)
class DirectedSupportModel:
    """Static directed graph plus dynamic counts fitted from Train only."""

    edge_catalog: pd.DataFrame
    counts: pd.DataFrame


def _physical_uid(uid: str) -> str:
    return uid[:-2] if uid.endswith((":F", ":R")) else uid


class DirectedCatalogBuilder:
    """Streaming conflict-checking catalog builder."""

    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}

    def update(self, route: pd.DataFrame) -> None:
        mapped = route.loc[
            route["canonical_mapping_available"].eq(True),
            [
                "observed_directed_edge_uid",
                "observed_from_node",
                "observed_to_node",
                "observed_direction",
                "canonical_length_m",
                "canonical_highway",
                "road_class",
                "bridge",
                "tunnel",
                "synthetic_reverse_edge",
                "osm_direction_disagreement",
            ],
        ].drop_duplicates()
        for row in mapped.itertuples(index=False):
            uid = str(row.observed_directed_edge_uid)
            candidate = {
                "observed_directed_edge_uid": uid,
                "physical_edge_uid": _physical_uid(uid),
                "observed_from_node": int(row.observed_from_node),
                "observed_to_node": int(row.observed_to_node),
                "observed_direction": str(row.observed_direction),
                "canonical_length_m": float(row.canonical_length_m),
                "canonical_highway": str(row.canonical_highway),
                "road_class": str(row.road_class),
                "bridge": bool(row.bridge),
                "tunnel": bool(row.tunnel),
                "synthetic_reverse_edge": bool(row.synthetic_reverse_edge),
                "osm_direction_disagreement": bool(
                    row.osm_direction_disagreement
                ),
            }
            previous = self._records.get(uid)
            if previous is None:
                self._records[uid] = candidate
                continue
            stable = (
                "physical_edge_uid",
                "observed_from_node",
                "observed_to_node",
                "observed_direction",
                "canonical_length_m",
                "canonical_highway",
                "road_class",
                "bridge",
                "tunnel",
            )
            for column in stable:
                left, right = previous[column], candidate[column]
                equal = (
                    np.isclose(left, right, atol=1e-6, rtol=1e-12)
                    if column == "canonical_length_m"
                    else left == right
                )
                if not equal:
                    raise ContractError(
                        f"directed edge catalog conflicts for {uid}.{column}"
                    )
            previous["synthetic_reverse_edge"] = bool(
                previous["synthetic_reverse_edge"]
                or candidate["synthetic_reverse_edge"]
            )
            previous["osm_direction_disagreement"] = bool(
                previous["osm_direction_disagreement"]
                or candidate["osm_direction_disagreement"]
            )

    def finalize(self) -> pd.DataFrame:
        records = {key: dict(value) for key, value in self._records.items()}
        parent: dict[int, int] = {}

        def find(node: int) -> int:
            parent.setdefault(node, node)
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[max(left_root, right_root)] = min(left_root, right_root)

        for record in records.values():
            union(record["observed_from_node"], record["observed_to_node"])
        for record in records.values():
            record["upper_region_id"] = str(
                find(record["observed_from_node"])
            )

        catalog = pd.DataFrame(records.values(), columns=CATALOG_COLUMNS)
        if len(catalog):
            catalog = catalog.sort_values(
                "observed_directed_edge_uid", kind="stable"
            ).reset_index(drop=True)
            catalog["observed_from_node"] = catalog[
                "observed_from_node"
            ].astype("Int64")
            catalog["observed_to_node"] = catalog["observed_to_node"].astype(
                "Int64"
            )
        return catalog


def build_directed_edge_catalog(
    route_batches: Iterable[pd.DataFrame],
) -> pd.DataFrame:
    """Create real graph edges for every observed direction, including synthetic R."""

    builder = DirectedCatalogBuilder()
    for route in route_batches:
        builder.update(route)
    return builder.finalize()


def catalog_region_statistics(catalog: pd.DataFrame) -> dict[str, Any]:
    if catalog.empty:
        return {
            "region_count": 0,
            "maximum_region_edge_share": 0.0,
            "region_size_p50": 0.0,
            "region_size_p90": 0.0,
            "region_size_max": 0,
        }
    sizes = catalog.groupby("upper_region_id", sort=False).size().to_numpy()
    return {
        "region_count": int(len(sizes)),
        "maximum_region_edge_share": float(sizes.max() / len(catalog)),
        "region_size_p50": float(np.quantile(sizes, 0.50)),
        "region_size_p90": float(np.quantile(sizes, 0.90)),
        "region_size_max": int(sizes.max()),
    }


def fit_directed_support(
    primitive_batches: Iterable[pd.DataFrame],
    edge_catalog: pd.DataFrame,
) -> DirectedSupportModel:
    """Count direct GPS intervals using Train batches only."""

    catalog = edge_catalog.set_index("observed_directed_edge_uid", drop=False)
    counters: dict[str, Counter[tuple[str, int]]] = {
        scope: Counter()
        for scope in (
            "edge",
            "edge_hour",
            "edge_time_bin_30m",
            "highway_hour",
            "node_hour",
            "global_hour",
        )
    }
    for frame in primitive_batches:
        if frame.empty:
            continue
        for row in frame.itertuples(index=False):
            uid = str(row.observed_directed_edge_uid)
            if uid not in catalog.index:
                raise ContractError(f"direct label edge absent from graph: {uid}")
            count = int(row.direct_interval_count)
            if count <= 0:
                raise ContractError("direct_interval_count must be positive")
            local_time = pd.to_datetime(
                float(row.observation_window_start_time),
                unit="s",
                utc=True,
            ).tz_convert("Asia/Shanghai")
            hour = int(local_time.hour)
            time_bin_30m = hour * 2 + int(local_time.minute >= 30)
            edge = catalog.loc[uid]
            counters["edge"][(uid, -1)] += count
            counters["edge_hour"][(uid, hour)] += count
            counters["edge_time_bin_30m"][(uid, time_bin_30m)] += count
            counters["highway_hour"][
                (str(edge.canonical_highway), hour)
            ] += count
            for node in {
                int(edge.observed_from_node),
                int(edge.observed_to_node),
            }:
                counters["node_hour"][(str(node), hour)] += count
            counters["global_hour"][("global", hour)] += count

    rows = [
        {
            "scope": scope,
            "key": key,
            "hour": hour,
            "observation_count": int(count),
        }
        for scope, counter in counters.items()
        for (key, hour), count in counter.items()
    ]
    counts = pd.DataFrame(rows, columns=COUNT_COLUMNS)
    if len(counts):
        counts = counts.sort_values(
            ["scope", "key", "hour"], kind="stable"
        ).reset_index(drop=True)
    return DirectedSupportModel(edge_catalog=edge_catalog, counts=counts)


def fit_directed_support_from_observations(
    observation_batches: Iterable[pd.DataFrame],
    edge_catalog: pd.DataFrame,
    config: "Stage1V3Config",
) -> DirectedSupportModel:
    """Fit the same counts without building expensive kinematic primitives."""

    catalog_columns = [
        "observed_directed_edge_uid",
        "observed_from_node",
        "observed_to_node",
        "canonical_highway",
    ]
    catalog = edge_catalog[catalog_columns]
    counters: dict[str, Counter[tuple[str, int]]] = {
        scope: Counter()
        for scope in (
            "edge",
            "edge_hour",
            "edge_time_bin_30m",
            "highway_hour",
            "node_hour",
            "global_hour",
        )
    }

    def update(
        scope: str,
        grouped: pd.DataFrame,
        key_column: str,
        hour_column: str,
    ) -> None:
        for row in grouped.itertuples(index=False):
            counters[scope][
                (str(getattr(row, key_column)), int(getattr(row, hour_column)))
            ] += int(row.observation_count)

    timezone = str(config.section("time")["timezone"])
    for observations in observation_batches:
        if observations.empty:
            continue
        samples = observations[
            [
                "order_id",
                "traversal_id",
                "gps_interval_id",
                "observed_directed_edge_uid",
                "interval_start_time",
            ]
        ].copy()
        samples["observation_count"] = 1
        local_time = pd.to_datetime(
            samples["interval_start_time"],
            unit="s",
            utc=True,
        ).dt.tz_convert(timezone)
        samples["hour"] = local_time.dt.hour.astype(int)
        samples["time_bin_30m"] = (
            samples["hour"] * 2 + (local_time.dt.minute >= 30).astype(int)
        )
        samples = samples.merge(
            catalog,
            on="observed_directed_edge_uid",
            how="left",
            validate="many_to_one",
            indicator=True,
        )
        if samples["_merge"].ne("both").any():
            raise ContractError("a Train direct label edge is absent from graph")
        edge_total = (
            samples.groupby("observed_directed_edge_uid", sort=False)[
                "observation_count"
            ]
            .sum()
            .reset_index()
            .assign(hour=-1)
        )
        update("edge", edge_total, "observed_directed_edge_uid", "hour")
        for scope, key in (
            ("edge_hour", "observed_directed_edge_uid"),
            ("highway_hour", "canonical_highway"),
        ):
            grouped = (
                samples.groupby([key, "hour"], sort=False)[
                    "observation_count"
                ]
                .sum()
                .reset_index()
            )
            update(scope, grouped, key, "hour")
        edge_time_bin = (
            samples.groupby(
                ["observed_directed_edge_uid", "time_bin_30m"], sort=False
            )["observation_count"]
            .sum()
            .reset_index()
        )
        update(
            "edge_time_bin_30m",
            edge_time_bin,
            "observed_directed_edge_uid",
            "time_bin_30m",
        )

        node_samples = pd.concat(
            [
                samples[
                    [
                        "order_id",
                        "traversal_id",
                        "gps_interval_id",
                        "hour",
                        "observation_count",
                        node_column,
                    ]
                ].rename(columns={node_column: "node"})
                for node_column in ("observed_from_node", "observed_to_node")
            ],
            ignore_index=True,
        ).drop_duplicates(
            ["order_id", "gps_interval_id", "node"]
        )
        node_grouped = (
            node_samples.groupby(["node", "hour"], sort=False)[
                "observation_count"
            ]
            .sum()
            .reset_index()
        )
        update("node_hour", node_grouped, "node", "hour")
        global_grouped = (
            samples.groupby("hour", sort=False)["observation_count"]
            .sum()
            .reset_index()
            .assign(key="global")
        )
        update("global_hour", global_grouped, "key", "hour")

    rows = [
        {
            "scope": scope,
            "key": key,
            "hour": hour,
            "observation_count": int(count),
        }
        for scope, counter in counters.items()
        for (key, hour), count in counter.items()
    ]
    counts = pd.DataFrame(rows, columns=COUNT_COLUMNS)
    if len(counts):
        counts = counts.sort_values(
            ["scope", "key", "hour"], kind="stable"
        ).reset_index(drop=True)
    return DirectedSupportModel(edge_catalog=edge_catalog, counts=counts)


def apply_directed_support(
    frame: pd.DataFrame,
    model: DirectedSupportModel,
    config: "Stage1V3Config",
) -> pd.DataFrame:
    """Attach Train-only support and the selected deterministic fallback level."""

    if frame.empty:
        result = frame.copy()
        result["edge_observation_count"] = pd.Series(dtype="int64")
        result["edge_hour_observation_count"] = pd.Series(dtype="int64")
        result["edge_time_bin_30m_observation_count"] = pd.Series(
            dtype="int64"
        )
        result["edge_support_level"] = pd.Series(dtype="object")
        result["edge_hour_support_level"] = pd.Series(dtype="object")
        result["directed_edge_model_scope"] = pd.Series(dtype="object")
        return result
    support = config.section("support")
    edge_minimum = int(support["minimum_edge_observations"])
    edge_hour_minimum = int(support["minimum_edge_hour_observations"])
    fallback_minimum = int(support["minimum_fallback_observations"])

    catalog = model.edge_catalog.set_index(
        "observed_directed_edge_uid", drop=False
    )
    lookup = {
        (str(row.scope), str(row.key), int(row.hour)): int(
            row.observation_count
        )
        for row in model.counts.itertuples(index=False)
    }
    result = frame.copy()
    edge_counts: list[int] = []
    edge_hour_counts: list[int] = []
    edge_time_bin_counts: list[int] = []
    edge_levels: list[str] = []
    hour_levels: list[str] = []
    model_scopes: list[str] = []
    for row in result.itertuples(index=False):
        uid = str(row.observed_directed_edge_uid)
        train_seen = uid in catalog.index
        if train_seen:
            edge = catalog.loc[uid]
            canonical_highway = str(edge.canonical_highway)
            from_node = int(edge.observed_from_node)
            to_node = int(edge.observed_to_node)
        else:
            canonical_highway = str(row.canonical_highway)
            from_node = int(row.observed_from_node)
            to_node = int(row.observed_to_node)
        model_scopes.append("train_seen" if train_seen else "evaluation_unseen")
        local_time = pd.to_datetime(
            float(row.observation_window_start_time), unit="s", utc=True
        ).tz_convert(str(config.section("time")["timezone"]))
        hour = int(local_time.hour)
        time_bin_30m = hour * 2 + int(local_time.minute >= 30)
        edge_count = lookup.get(("edge", uid, -1), 0)
        edge_hour_count = lookup.get(("edge_hour", uid, hour), 0)
        edge_time_bin_count = lookup.get(
            ("edge_time_bin_30m", uid, time_bin_30m), 0
        )
        edge_counts.append(edge_count)
        edge_hour_counts.append(edge_hour_count)
        edge_time_bin_counts.append(edge_time_bin_count)
        edge_levels.append(
            "edge" if edge_count >= edge_minimum else "fallback_required"
        )
        candidates = (
            ("edge_hour", edge_hour_count),
            (
                "highway_hour",
                lookup.get(
                    ("highway_hour", canonical_highway, hour), 0
                ),
            ),
            (
                "spatial_neighbor",
                max(
                    lookup.get(
                        ("node_hour", str(from_node), hour),
                        0,
                    ),
                    lookup.get(
                        ("node_hour", str(to_node), hour),
                        0,
                    ),
                ),
            ),
            ("global_hour", lookup.get(("global_hour", "global", hour), 0)),
        )
        chosen = "unavailable"
        for level, count in candidates:
            minimum = edge_hour_minimum if level == "edge_hour" else fallback_minimum
            if count >= minimum:
                chosen = level
                break
        hour_levels.append(chosen)

    result["edge_observation_count"] = edge_counts
    result["edge_hour_observation_count"] = edge_hour_counts
    result["edge_time_bin_30m_observation_count"] = edge_time_bin_counts
    result["edge_support_level"] = edge_levels
    result["edge_hour_support_level"] = hour_levels
    result["directed_edge_model_scope"] = model_scopes
    return result
