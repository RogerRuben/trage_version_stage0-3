"""Read-only Stage 0 v6 bucket discovery and validation for Stage 1 v3."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd

from stage1.v3.config import Stage1V3Config, validate_split_config
from stage1.v3.schema import (
    ALL_INPUT_PRODUCTS,
    DIRECT_MEASUREMENT_SOURCE,
    DYNAMIC_STATUSES,
    INPUT_BUCKET_SCHEMA_VERSION,
    INPUT_ROOT_NAME,
    KNOWN_MEASUREMENT_SOURCES,
    KNOWN_MOVEMENT_SOURCES,
    REQUIRED_COLUMNS,
    REQUIRED_PRODUCTS,
    Stage1V3InputError,
)


_DATE_DIRECTORY = re.compile(r"^date=(\d{8})$")
_BUCKET_DIRECTORY = re.compile(r"^bucket=(\d{5})$")
_KNOWN_SPLITS = ("train", "validation", "test")
_KNOWN_AUXILIARY_DIRECTORIES = frozenset({"manifests", "rejections"})
_ORDER_STATUS_VALUES = {
    "gps_status": frozenset({"clean", "local_outlier"}),
    "route_status": frozenset({"route_pass"}),
    "dynamic_status": DYNAMIC_STATUSES,
    "canonical_status": frozenset({"unique", "chain_resolved"}),
}


@dataclass(frozen=True)
class BucketRef:
    split: str
    date: str
    bucket: int
    path: Path
    manifest: dict[str, Any]


@dataclass(frozen=True)
class Stage0Bucket:
    order_base: pd.DataFrame
    route_parts: pd.DataFrame
    link_traversals: pd.DataFrame
    link_interval_observations: pd.DataFrame
    interval_measurements: pd.DataFrame
    turn_movements: pd.DataFrame
    gps_quality: pd.DataFrame
    route_quality: pd.DataFrame
    dynamic_quality: pd.DataFrame
    canonical_quality: pd.DataFrame


@dataclass(frozen=True)
class Stage0FitBucket:
    """The three preflight-validated products needed while fitting models."""

    route_parts: pd.DataFrame
    link_traversals: pd.DataFrame
    link_interval_observations: pd.DataFrame


def _split_dates(config: Stage1V3Config) -> dict[str, tuple[str, ...]]:
    return {
        "train": config.train_dates,
        "validation": config.validation_dates,
        "test": (config.test_date,),
    }


def _validated_input_root(root: str | Path) -> Path:
    path = Path(root).resolve()
    if not path.is_dir():
        raise Stage1V3InputError(f"Stage1 input root does not exist: {path}")
    if path.name != INPUT_ROOT_NAME:
        raise Stage1V3InputError(
            f"Stage1 v3 reads only a directory named {INPUT_ROOT_NAME!r}: {path}"
        )
    return path


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Stage1V3InputError(f"bucket manifest is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage1V3InputError(f"cannot read bucket manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Stage1V3InputError(f"bucket manifest must be a JSON object: {path}")
    return value


def _manifest_integer(manifest: dict[str, Any], name: str, context: Path) -> int:
    value = manifest.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise Stage1V3InputError(f"{context}: manifest {name!r} must be an integer")
    return value


def _validate_manifest(ref: BucketRef) -> None:
    manifest = ref.manifest
    context = ref.path / "manifest.json"
    if manifest.get("status") != "PASS":
        raise Stage1V3InputError(f"{context}: bucket status must be PASS")
    if manifest.get("schema_version") != INPUT_BUCKET_SCHEMA_VERSION:
        raise Stage1V3InputError(
            f"{context}: expected schema_version {INPUT_BUCKET_SCHEMA_VERSION!r}, "
            f"got {manifest.get('schema_version')!r}"
        )
    if manifest.get("split") != ref.split:
        raise Stage1V3InputError(f"{context}: manifest split does not match its path")
    if str(manifest.get("date")) != ref.date:
        raise Stage1V3InputError(f"{context}: manifest date does not match its path")
    if _manifest_integer(manifest, "bucket", context) != ref.bucket:
        raise Stage1V3InputError(f"{context}: manifest bucket does not match its path")
    accepted = _manifest_integer(manifest, "accepted_core_count", context)
    if accepted < 0:
        raise Stage1V3InputError(
            f"{context}: accepted_core_count cannot be negative"
        )
    exceptions = _manifest_integer(
        manifest, "processing_exception_count", context
    )
    if exceptions != 0:
        raise Stage1V3InputError(
            f"{context}: processing_exception_count must be zero"
        )
    counts = manifest.get("product_row_counts")
    if not isinstance(counts, dict):
        raise Stage1V3InputError(
            f"{context}: product_row_counts must be a JSON object"
        )
    for product in ALL_INPUT_PRODUCTS:
        count = counts.get(product)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise Stage1V3InputError(
                f"{context}: missing or invalid row count for {product!r}"
            )


def iter_stage0_buckets(
    root: str | Path,
    config: Stage1V3Config,
    splits: Sequence[str] | None = None,
) -> Iterator[BucketRef]:
    """Discover valid buckets and reject every unexpected split or date.

    Discovery is read-only.  Missing expected dates are allowed so callers can
    inspect an in-progress production tree; unexpected dates are never allowed.
    """

    validate_split_config(config)
    input_root = _validated_input_root(root)
    expected_dates = _split_dates(config)
    selected = tuple(_KNOWN_SPLITS if splits is None else splits)
    if len(set(selected)) != len(selected):
        raise Stage1V3InputError(f"duplicate split requested: {selected!r}")
    unknown_selected = sorted(set(selected) - set(_KNOWN_SPLITS))
    if unknown_selected:
        raise Stage1V3InputError(f"unknown requested splits: {unknown_selected}")

    split_directories: dict[str, Path] = {}
    for child in input_root.iterdir():
        if not child.is_dir():
            continue
        if child.name in _KNOWN_AUXILIARY_DIRECTORIES:
            continue
        if not child.name.startswith("split="):
            raise Stage1V3InputError(
                f"unexpected directory in Stage1 input root: {child}"
            )
        split = child.name.removeprefix("split=")
        if split not in _KNOWN_SPLITS:
            raise Stage1V3InputError(f"unexpected split directory: {child}")
        split_directories[split] = child

    discovered: list[BucketRef] = []
    seen_keys: set[tuple[str, str, int]] = set()
    for split in _KNOWN_SPLITS:
        split_path = split_directories.get(split)
        if split_path is None:
            continue
        allowed_dates = set(expected_dates[split])
        for date_path in sorted(split_path.iterdir(), key=lambda item: item.name):
            if not date_path.is_dir():
                continue
            match = _DATE_DIRECTORY.fullmatch(date_path.name)
            if match is None:
                raise Stage1V3InputError(
                    f"unexpected directory below split={split}: {date_path}"
                )
            date = match.group(1)
            if date not in allowed_dates:
                raise Stage1V3InputError(
                    f"unexpected date {date!r} in split {split!r}: {date_path}"
                )
            for bucket_path in sorted(
                date_path.iterdir(), key=lambda item: item.name
            ):
                if not bucket_path.is_dir():
                    continue
                bucket_match = _BUCKET_DIRECTORY.fullmatch(bucket_path.name)
                if bucket_match is None:
                    raise Stage1V3InputError(
                        f"unexpected directory below date={date}: {bucket_path}"
                    )
                bucket = int(bucket_match.group(1))
                key = (split, date, bucket)
                if key in seen_keys:
                    raise Stage1V3InputError(f"duplicate Stage0 bucket: {key!r}")
                seen_keys.add(key)
                manifest = _read_manifest(bucket_path / "manifest.json")
                ref = BucketRef(
                    split=split,
                    date=date,
                    bucket=bucket,
                    path=bucket_path,
                    manifest=copy.deepcopy(manifest),
                )
                _validate_manifest(ref)
                if split in selected:
                    discovered.append(ref)

    yield from sorted(
        discovered,
        key=lambda ref: (
            _KNOWN_SPLITS.index(ref.split),
            ref.date,
            ref.bucket,
        ),
    )


def _read_product(ref: BucketRef, product: str) -> pd.DataFrame:
    path = ref.path / f"{product}.parquet"
    if not path.is_file():
        raise Stage1V3InputError(f"required Stage0 product is missing: {path}")
    expected = ref.manifest["product_row_counts"][product]
    if expected == 0:
        try:
            empty = pd.read_parquet(path)
        except Exception as exc:
            raise Stage1V3InputError(
                f"cannot read empty required product {path}: {exc}"
            ) from exc
        if len(empty):
            raise Stage1V3InputError(
                f"{path}: manifest declares zero rows, found {len(empty)}"
            )
        # Stage 0 writes a zero-column parquet when a bucket accepts no
        # orders.  Restore the frozen logical schema without inventing rows.
        return pd.DataFrame(
            {
                column: pd.Series(dtype="object")
                for column in sorted(REQUIRED_COLUMNS[product])
            }
        )
    columns = sorted(REQUIRED_COLUMNS[product])
    try:
        frame = pd.read_parquet(path, columns=columns)
    except Exception as exc:
        raise Stage1V3InputError(
            f"cannot read required columns from {path}: {exc}"
        ) from exc
    missing = sorted(REQUIRED_COLUMNS[product] - set(frame.columns))
    if missing:
        raise Stage1V3InputError(f"{path}: missing required columns {missing}")
    if len(frame) != expected:
        raise Stage1V3InputError(
            f"{path}: expected {expected} rows from manifest, found {len(frame)}"
        )
    return frame


def _read_product_subset(
    ref: BucketRef,
    product: str,
    columns: Sequence[str],
) -> pd.DataFrame:
    requested = tuple(sorted(set(columns)))
    undeclared = sorted(set(requested) - REQUIRED_COLUMNS[product])
    if undeclared:
        raise Stage1V3InputError(
            f"{product}: requested undeclared columns {undeclared}"
        )
    path = ref.path / f"{product}.parquet"
    if not path.is_file():
        raise Stage1V3InputError(f"required Stage0 product is missing: {path}")
    expected = int(ref.manifest["product_row_counts"][product])
    if expected == 0:
        return pd.DataFrame(
            {column: pd.Series(dtype="object") for column in requested}
        )
    try:
        frame = pd.read_parquet(path, columns=list(requested))
    except Exception as exc:
        raise Stage1V3InputError(
            f"cannot read trusted columns from {path}: {exc}"
        ) from exc
    if len(frame) != expected:
        raise Stage1V3InputError(
            f"{path}: expected {expected} rows from manifest, found {len(frame)}"
        )
    return frame


def _require_non_null(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    for column in columns:
        count = int(frame[column].isna().sum())
        if count:
            raise Stage1V3InputError(
                f"{name}.{column} contains {count} null values"
            )


def _require_strict_true(series: pd.Series, name: str) -> None:
    strict_boolean = series.map(
        lambda value: isinstance(value, (bool, np.bool_))
    )
    if not strict_boolean.all() or not series.eq(True).all():
        raise Stage1V3InputError(f"{name} must contain only the boolean true")


def _require_unique(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    _require_non_null(frame, columns, name)
    duplicates = int(frame.duplicated(list(columns), keep=False).sum())
    if duplicates:
        raise Stage1V3InputError(
            f"{name} contains {duplicates} rows with duplicate key {list(columns)}"
        )


def _order_ids(frame: pd.DataFrame) -> set[Any]:
    return set(frame["order_id"].tolist())


def _require_order_set(
    frame: pd.DataFrame,
    expected: set[Any],
    name: str,
) -> None:
    actual = _order_ids(frame)
    missing = expected - actual
    unexpected = actual - expected
    if missing or unexpected:
        raise Stage1V3InputError(
            f"{name} order foreign key mismatch: "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )


def _require_order_subset(
    frame: pd.DataFrame,
    expected: set[Any],
    name: str,
) -> None:
    unexpected = _order_ids(frame) - expected
    if unexpected:
        raise Stage1V3InputError(
            f"{name} contains {len(unexpected)} orphan order foreign keys"
        )


def _validate_measurement_sources(frame: pd.DataFrame, name: str) -> None:
    _require_non_null(frame, ["measurement_source"], name)
    observed = set(frame["measurement_source"].astype(str).unique())
    unknown = sorted(observed - KNOWN_MEASUREMENT_SOURCES)
    if unknown:
        raise Stage1V3InputError(
            f"{name} contains unknown measurement_source values: {unknown}"
        )


def _series_matches(
    left: pd.Series,
    right: pd.Series,
) -> np.ndarray:
    both_null = left.isna().to_numpy() & right.isna().to_numpy()
    both_present = left.notna().to_numpy() & right.notna().to_numpy()
    equal_present = (
        left.astype("string").fillna("").to_numpy()
        == right.astype("string").fillna("").to_numpy()
    )
    return both_null | (both_present & equal_present)


def _tolerance(
    config: Stage1V3Config,
    name: str,
    default: float,
) -> float:
    direct_aliases = {
        "interval_time_abs_s": "duration_tolerance_s",
        "interval_speed_abs_mps": "speed_tolerance_mps",
        "interval_overlap_abs_s": "duration_tolerance_s",
        "dynamic_time_abs_s": "duration_tolerance_s",
        "dynamic_distance_abs_m": "distance_identity_tolerance_m",
        "interval_time_rel": "interval_time_rel",
        "interval_speed_rel": "interval_speed_rel",
        "dynamic_time_rel": "dynamic_time_rel",
        "dynamic_distance_rel": "dynamic_distance_rel",
    }
    direct = config.section("direct")
    value: Any = default
    direct_name = direct_aliases.get(name)
    if direct_name is not None and direct_name in direct:
        value = direct[direct_name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Stage1V3InputError(f"tolerance {name!r} must be numeric")
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise Stage1V3InputError(
            f"tolerance {name!r} must be finite and non-negative"
        )
    return result


def _numeric(frame: pd.DataFrame, column: str, name: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    invalid = ~np.isfinite(values)
    if invalid.any():
        raise Stage1V3InputError(
            f"{name}.{column} contains {int(invalid.sum())} non-finite values"
        )
    return values


def _optional_numeric(
    frame: pd.DataFrame,
    column: str,
    name: str,
) -> pd.Series:
    raw = frame[column]
    values = pd.to_numeric(raw, errors="coerce")
    invalid = raw.notna() & (values.isna() | ~np.isfinite(values))
    if invalid.any():
        raise Stage1V3InputError(
            f"{name}.{column} contains {int(invalid.sum())} invalid values"
        )
    return values.astype(float)


_NULLABLE_NODE_COLUMNS = {
    "order_base": ("start_node", "end_node"),
    "route_parts": (
        "canonical_from_node",
        "canonical_to_node",
        "begin_osm_node_id",
        "end_osm_node_id",
    ),
    "turn_movements": ("via_node",),
}


def _nullable_int64(
    series: pd.Series,
    *,
    name: str,
) -> pd.Series:
    """Normalize logically integral identifiers independently of Parquet dtype."""

    numeric = pd.to_numeric(series, errors="coerce")
    invalid = series.notna() & numeric.isna()
    non_integral = numeric.notna() & numeric.ne(np.floor(numeric))
    if invalid.any() or non_integral.any():
        raise Stage1V3InputError(
            f"{name} contains "
            f"{int((invalid | non_integral).sum())} non-integral identifiers"
        )
    return numeric.astype("Int64")


def _normalize_nullable_dtypes(products: dict[str, pd.DataFrame]) -> None:
    for product, columns in _NULLABLE_NODE_COLUMNS.items():
        frame = products[product]
        for column in columns:
            frame[column] = _nullable_int64(
                frame[column],
                name=f"{product}.{column}",
            )


def _directed_uid(uid: Any, direction: Any) -> Any:
    if pd.isna(uid) or pd.isna(direction):
        return pd.NA
    direction_text = str(direction).strip().upper()
    if direction_text not in {"F", "R"}:
        return pd.NA
    uid_text = str(uid)
    if re.search(r":[FR]$", uid_text):
        return re.sub(r":[FR]$", f":{direction_text}", uid_text)
    raise Stage1V3InputError(
        f"canonical_edge_uid has no terminal direction suffix: {uid_text!r}"
    )


def _enrich_route_direction_lineage(route: pd.DataFrame) -> None:
    """Attach actual directed identity to route rows in place."""

    direction = route["canonical_traversal_direction"].astype("string").str.upper()
    mapped = (
        route["canonical_edge_uid"].notna()
        & direction.isin(["F", "R"])
        & route["canonical_from_node"].notna()
        & route["canonical_to_node"].notna()
        & route["mapping_status"].astype("string").ne("unmapped")
    )
    invalid_direction = (
        route["canonical_edge_uid"].notna()
        & ~direction.isin(["F", "R"])
    )
    if invalid_direction.any():
        raise Stage1V3InputError(
            "mapped route_parts contain "
            f"{int(invalid_direction.sum())} missing/invalid actual directions"
        )

    route["canonical_mapping_available"] = mapped.astype(bool)
    route["observed_direction"] = direction.where(mapped, pd.NA)
    route["observed_directed_edge_uid"] = [
        _directed_uid(uid, actual_direction) if available else pd.NA
        for uid, actual_direction, available in zip(
            route["canonical_edge_uid"],
            route["observed_direction"],
            mapped,
        )
    ]
    route["observed_from_node"] = route["canonical_from_node"].where(mapped).astype(
        "Int64"
    )
    route["observed_to_node"] = route["canonical_to_node"].where(mapped).astype(
        "Int64"
    )
    uid_direction = route["canonical_edge_uid"].astype("string").str.extract(
        r":([FR])$", expand=False
    )
    route["synthetic_reverse_edge"] = (
        mapped & uid_direction.eq("F") & direction.eq("R")
    ).astype(bool)
    route["osm_direction_disagreement"] = (
        route["traversed_against_osm_oneway"].fillna(False).astype(bool)
    )
    route["route_lineage_status"] = np.where(
        mapped,
        "mapped",
        "unmapped_lineage_gap",
    )
    route["sequence_feature_mask"] = mapped.astype(bool)


def _enrich_direction_lineage(products: dict[str, pd.DataFrame]) -> None:
    """Attach actual traversal identity while retaining physical Stage 0 lineage."""

    route = products["route_parts"]
    _enrich_route_direction_lineage(route)
    traversal = products["link_traversals"]
    context_columns = [
        "order_id",
        "route_sequence",
        "observed_directed_edge_uid",
        "observed_from_node",
        "observed_to_node",
        "observed_direction",
        "synthetic_reverse_edge",
        "osm_direction_disagreement",
        "canonical_mapping_available",
        "mapping_status",
        "osm_oneway",
        "canonical_highway",
        "road_class",
        "bridge",
        "tunnel",
    ]
    traversal_context = route[context_columns]
    enriched = traversal.merge(
        traversal_context,
        on=["order_id", "route_sequence"],
        how="left",
        validate="many_to_one",
        indicator="_direction_join",
    )
    if enriched["_direction_join"].ne("both").any():
        raise Stage1V3InputError(
            "link_traversals contains route_sequence values without route_parts"
        )
    products["link_traversals"] = enriched.drop(columns="_direction_join")

    observations = products["link_interval_observations"]
    observation_context = products["link_traversals"][
        [
            "order_id",
            "traversal_id",
            "observed_directed_edge_uid",
            "observed_from_node",
            "observed_to_node",
            "observed_direction",
            "synthetic_reverse_edge",
            "osm_direction_disagreement",
            "canonical_mapping_available",
            "mapping_status",
            "osm_oneway",
        ]
    ]
    enriched_observations = observations.merge(
        observation_context,
        on=["order_id", "traversal_id"],
        how="left",
        validate="many_to_one",
        indicator="_direction_join",
    )
    if enriched_observations["_direction_join"].ne("both").any():
        raise Stage1V3InputError(
            "link_interval_observations contains traversal foreign-key failures"
        )
    products["link_interval_observations"] = enriched_observations.drop(
        columns="_direction_join"
    )


def _fill_nullable_order_endpoints(products: dict[str, pd.DataFrame]) -> None:
    order_base = products["order_base"]
    route = products["route_parts"].sort_values(
        ["order_id", "route_sequence"], kind="stable"
    )
    first = route.drop_duplicates("order_id", keep="first").set_index("order_id")
    last = route.drop_duplicates("order_id", keep="last").set_index("order_id")
    order_index = order_base["order_id"]

    start_canonical = order_index.map(first["canonical_from_node"])
    start_osm = order_index.map(first["begin_osm_node_id"])
    end_canonical = order_index.map(last["canonical_to_node"])
    end_osm = order_index.map(last["end_osm_node_id"])
    order_base["start_node"] = (
        order_base["start_node"]
        .combine_first(start_canonical)
        .combine_first(start_osm)
        .astype("Int64")
    )
    order_base["end_node"] = (
        order_base["end_node"]
        .combine_first(end_canonical)
        .combine_first(end_osm)
        .astype("Int64")
    )


def _validate_order_base(ref: BucketRef, frame: pd.DataFrame) -> set[Any]:
    _require_unique(frame, ["order_id"], "order_base")
    _require_non_null(
        frame,
        ["selection_hash"],
        "order_base",
    )
    accepted = ref.manifest["accepted_core_count"]
    if len(frame) != accepted:
        raise Stage1V3InputError(
            f"{ref.path}: accepted_core_count={accepted}, "
            f"order_base rows={len(frame)}"
        )
    dates = set(frame["date"].astype(str).unique())
    splits = set(frame["split"].astype(str).unique())
    if dates != ({ref.date} if len(frame) else set()):
        raise Stage1V3InputError(
            f"{ref.path}: order_base date values do not match {ref.date!r}"
        )
    if splits != ({ref.split} if len(frame) else set()):
        raise Stage1V3InputError(
            f"{ref.path}: order_base split values do not match {ref.split!r}"
        )
    departure = _numeric(frame, "departure_time", "order_base")
    arrival = _numeric(frame, "arrival_time", "order_base")
    if (arrival <= departure).any():
        raise Stage1V3InputError(
            f"{ref.path}: order_base arrival_time must follow departure_time"
        )
    _require_strict_true(
        frame["stage1_core_eligible"],
        f"{ref.path}: order_base.stage1_core_eligible",
    )
    for column, allowed in _ORDER_STATUS_VALUES.items():
        values = frame[column]
        observed = set(values.dropna().astype(str).unique())
        unexpected = sorted(observed - allowed)
        if values.isna().any() or unexpected:
            raise Stage1V3InputError(
                f"{ref.path}: order_base.{column} must be one of "
                f"{sorted(allowed)}, found unexpected values {unexpected}"
            )
    return _order_ids(frame)


def _validate_primary_and_foreign_keys(
    bucket: Stage0Bucket,
    order_ids: set[Any],
) -> None:
    route = bucket.route_parts
    traversals = bucket.link_traversals
    observations = bucket.link_interval_observations
    intervals = bucket.interval_measurements
    movements = bucket.turn_movements
    gps_quality = bucket.gps_quality
    route_quality = bucket.route_quality
    dynamic = bucket.dynamic_quality
    canonical_quality = bucket.canonical_quality

    _require_unique(route, ["order_id", "route_sequence"], "route_parts")
    _require_unique(
        traversals, ["order_id", "traversal_id"], "link_traversals"
    )
    _require_unique(
        observations,
        ["order_id", "gps_interval_id"],
        "link_interval_observations",
    )
    _require_unique(
        intervals,
        ["order_id", "gps_interval_id"],
        "interval_measurements",
    )
    _require_unique(
        movements,
        ["order_id", "movement_sequence"],
        "turn_movements",
    )
    _require_unique(gps_quality, ["order_id"], "gps_quality")
    _require_unique(route_quality, ["order_id"], "route_quality")
    _require_unique(dynamic, ["order_id"], "dynamic_quality")
    _require_unique(canonical_quality, ["order_id"], "canonical_quality")
    _require_non_null(route, ["length_m"], "route_parts")
    _require_non_null(
        traversals,
        ["route_sequence", "route_sequence_end"],
        "link_traversals",
    )
    route_sequence = _numeric(route, "route_sequence", "route_parts")
    traversal_start = _numeric(
        traversals, "route_sequence", "link_traversals"
    )
    traversal_end = _numeric(
        traversals, "route_sequence_end", "link_traversals"
    )
    if (
        not np.equal(route_sequence, np.floor(route_sequence)).all()
        or not np.equal(traversal_start, np.floor(traversal_start)).all()
        or not np.equal(traversal_end, np.floor(traversal_end)).all()
        or (traversal_end < traversal_start).any()
    ):
        raise Stage1V3InputError(
            "route_sequence values must be integers and traversal ranges "
            "must be monotone"
        )
    traversal_ranges = traversals[
        ["order_id", "traversal_id", "route_sequence", "route_sequence_end"]
    ].copy()
    traversal_ranges["_start"] = traversal_start
    traversal_ranges["_end"] = traversal_end
    traversal_ranges = traversal_ranges.sort_values(
        ["order_id", "_start", "_end", "traversal_id"],
        kind="stable",
    )
    traversal_ranges["_maximum_end_so_far"] = traversal_ranges.groupby(
        "order_id", sort=False
    )["_end"].cummax()
    previous_maximum_end = traversal_ranges.groupby(
        "order_id", sort=False
    )["_maximum_end_so_far"].shift()
    overlapping_ranges = previous_maximum_end.notna() & (
        traversal_ranges["_start"] <= previous_maximum_end
    )
    if overlapping_ranges.any():
        raise Stage1V3InputError(
            "link_traversals contains overlapping route_sequence ranges "
            "that would duplicate physical route distance"
        )

    for name, frame in (
        ("route_parts", route),
        ("link_traversals", traversals),
        ("link_interval_observations", observations),
        ("interval_measurements", intervals),
        ("gps_quality", gps_quality),
        ("route_quality", route_quality),
        ("dynamic_quality", dynamic),
        ("canonical_quality", canonical_quality),
    ):
        _require_order_set(frame, order_ids, name)
    _require_order_subset(movements, order_ids, "turn_movements")

    order_dynamic = bucket.order_base[
        ["order_id", "dynamic_status"]
    ].merge(
        dynamic[["order_id", "dynamic_status"]],
        on="order_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_order", "_quality"),
    )
    dynamic_status_match = _series_matches(
        order_dynamic["dynamic_status_order"],
        order_dynamic["dynamic_status_quality"],
    )
    if not dynamic_status_match.all():
        raise Stage1V3InputError(
            "order_base and dynamic_quality disagree on dynamic_status for "
            f"{int((~dynamic_status_match).sum())} orders"
        )

    for product, quality_frame, status_column in (
        ("gps_quality", gps_quality, "gps_status"),
        ("route_quality", route_quality, "route_status"),
        ("canonical_quality", canonical_quality, "canonical_status"),
    ):
        compared = bucket.order_base[["order_id", status_column]].merge(
            quality_frame[["order_id", status_column]],
            on="order_id",
            how="inner",
            validate="one_to_one",
            suffixes=("_order", "_quality"),
        )
        status_match = _series_matches(
            compared[f"{status_column}_order"],
            compared[f"{status_column}_quality"],
        )
        if not status_match.all():
            raise Stage1V3InputError(
                f"order_base and {product} disagree on {status_column} for "
                f"{int((~status_match).sum())} orders"
            )

    traversal_route = traversals.merge(
        route[["order_id", "route_sequence", "canonical_edge_uid"]],
        on=["order_id", "route_sequence"],
        how="left",
        validate="many_to_one",
        indicator=True,
        suffixes=("_traversal", "_route"),
    )
    missing_route = traversal_route["_merge"].ne("both")
    if missing_route.any():
        raise Stage1V3InputError(
            "link_traversals contains "
            f"{int(missing_route.sum())} route_sequence foreign-key failures"
        )
    canonical_match = _series_matches(
        traversal_route["canonical_edge_uid_traversal"],
        traversal_route["canonical_edge_uid_route"],
    )
    if not canonical_match.all():
        raise Stage1V3InputError(
            "link_traversals and route_parts disagree on canonical_edge_uid "
            f"for {int((~canonical_match).sum())} rows"
        )
    traversal_route_end = traversals.merge(
        route[["order_id", "route_sequence", "canonical_edge_uid"]].rename(
            columns={"route_sequence": "route_sequence_end"}
        ),
        on=["order_id", "route_sequence_end"],
        how="left",
        validate="many_to_one",
        indicator=True,
        suffixes=("_traversal", "_route"),
    )
    missing_route_end = traversal_route_end["_merge"].ne("both")
    if missing_route_end.any():
        raise Stage1V3InputError(
            "link_traversals contains "
            f"{int(missing_route_end.sum())} route_sequence_end "
            "foreign-key failures"
        )
    end_canonical_match = _series_matches(
        traversal_route_end["canonical_edge_uid_traversal"],
        traversal_route_end["canonical_edge_uid_route"],
    )
    if not end_canonical_match.all():
        raise Stage1V3InputError(
            "link_traversals route_sequence_end disagrees with route_parts "
            "canonical_edge_uid"
        )
    traversal_direction_end = traversals[
        [
            "order_id",
            "traversal_id",
            "route_sequence_end",
            "observed_directed_edge_uid",
            "observed_direction",
            "canonical_mapping_available",
        ]
    ].merge(
        route[
            [
                "order_id",
                "route_sequence",
                "observed_directed_edge_uid",
                "observed_direction",
                "canonical_mapping_available",
            ]
        ].rename(columns={"route_sequence": "route_sequence_end"}),
        on=["order_id", "route_sequence_end"],
        how="left",
        validate="many_to_one",
        suffixes=("_traversal", "_route"),
    )
    for column in (
        "observed_directed_edge_uid",
        "observed_direction",
        "canonical_mapping_available",
    ):
        matches = _series_matches(
            traversal_direction_end[f"{column}_traversal"],
            traversal_direction_end[f"{column}_route"],
        )
        if not matches.all():
            raise Stage1V3InputError(
                "link_traversal route range crosses actual directed edge "
                f"identity on {column}"
            )

    observation_traversal = observations.merge(
        traversals[
            [
                "order_id",
                "traversal_id",
                "canonical_edge_uid",
                "measurement_source",
            ]
        ],
        on=["order_id", "traversal_id"],
        how="left",
        validate="many_to_one",
        indicator=True,
        suffixes=("_observation", "_traversal"),
    )
    missing_traversal = observation_traversal["_merge"].ne("both")
    if missing_traversal.any():
        raise Stage1V3InputError(
            "link_interval_observations contains "
            f"{int(missing_traversal.sum())} traversal foreign-key failures"
        )
    canonical_match = _series_matches(
        observation_traversal["canonical_edge_uid_observation"],
        observation_traversal["canonical_edge_uid_traversal"],
    )
    if not canonical_match.all():
        raise Stage1V3InputError(
            "link_interval_observations and link_traversals disagree on "
            f"canonical_edge_uid for {int((~canonical_match).sum())} rows"
        )
    traversal_source = observation_traversal[
        "measurement_source_traversal"
    ].astype(str)
    if not traversal_source.eq(DIRECT_MEASUREMENT_SOURCE).all():
        raise Stage1V3InputError(
            "a direct interval references a traversal that is not "
            f"{DIRECT_MEASUREMENT_SOURCE!r}"
        )


def _validate_interval_classification(
    intervals: pd.DataFrame,
    observations: pd.DataFrame,
    dynamic: pd.DataFrame,
    config: Stage1V3Config,
) -> None:
    """Validate one exclusive class per raw GPS interval and its direct label."""

    _validate_measurement_sources(intervals, "interval_measurements")
    start = _numeric(
        intervals, "interval_start_time", "interval_measurements"
    )
    end = _numeric(intervals, "interval_end_time", "interval_measurements")
    duration = _numeric(
        intervals, "interval_duration_s", "interval_measurements"
    )
    gps_distance = _numeric(
        intervals, "gps_interval_distance_m", "interval_measurements"
    )
    if (end < start).any() or (duration < 0).any():
        raise Stage1V3InputError(
            "interval_measurements timestamps and durations must be non-negative"
        )
    if (gps_distance < 0).any():
        raise Stage1V3InputError(
            "interval_measurements.gps_interval_distance_m must be non-negative"
        )
    time_abs = _tolerance(config, "interval_time_abs_s", 1e-6)
    time_rel = _tolerance(config, "interval_time_rel", 1e-12)
    duration_match = np.isclose(
        end - start,
        duration,
        atol=time_abs,
        rtol=time_rel,
    )
    if not duration_match.all():
        raise Stage1V3InputError(
            "interval_measurements timestamp duration mismatch for "
            f"{int((~duration_match).sum())} rows"
        )

    source = intervals["measurement_source"].astype(str)
    category_columns = {
        "direct_observed": "direct_observed_travel_time_s",
        "interval_supported": "interval_supported_time_s",
        "engine_interpolated": "engine_allocated_only_time_s",
        "unresolved": "unresolved_time_s",
    }
    category_values = {
        category: _optional_numeric(
            intervals, column, "interval_measurements"
        )
        for category, column in category_columns.items()
    }
    for category, column in category_columns.items():
        allocated = category_values[category]
        selected = source.eq(category)
        invalid_selected = selected & (
            allocated.isna()
            | ~np.isclose(
                allocated.fillna(0.0).to_numpy(dtype=float),
                duration,
                atol=time_abs,
                rtol=time_rel,
            )
        )
        invalid_other = ~selected & allocated.notna()
        if invalid_selected.any() or invalid_other.any():
            raise Stage1V3InputError(
                f"interval_measurements.{column} violates exclusive "
                f"{category!r} allocation: selected_failures="
                f"{int(invalid_selected.sum())}, "
                f"nonselected_values={int(invalid_other.sum())}"
            )

    direct_distance = _optional_numeric(
        intervals,
        "direct_observed_distance_m",
        "interval_measurements",
    )
    direct_mask = source.eq(DIRECT_MEASUREMENT_SOURCE)
    invalid_direct_distance = direct_mask & (
        direct_distance.isna() | direct_distance.lt(0)
    )
    invalid_non_direct_distance = ~direct_mask & direct_distance.notna()
    if invalid_direct_distance.any() or invalid_non_direct_distance.any():
        raise Stage1V3InputError(
            "direct_observed_distance_m must exist only for direct intervals"
        )

    direct_columns = [
        "order_id",
        "gps_interval_id",
        "interval_start_time",
        "interval_end_time",
        "direct_observed_travel_time_s",
        "direct_observed_distance_m",
    ]
    direct_intervals = intervals.loc[direct_mask, direct_columns]
    direct_observations = observations[
        [
            "order_id",
            "gps_interval_id",
            "interval_start_time",
            "interval_end_time",
            "observed_travel_time_s",
            "observed_distance_m",
        ]
    ]
    interval_keys = set(
        direct_intervals[["order_id", "gps_interval_id"]].itertuples(
            index=False, name=None
        )
    )
    observation_keys = set(
        direct_observations[["order_id", "gps_interval_id"]].itertuples(
            index=False, name=None
        )
    )
    if interval_keys != observation_keys:
        raise Stage1V3InputError(
            "direct interval classifications and link_interval_observations "
            "must have exactly the same (order_id, gps_interval_id) keys"
        )
    paired = direct_intervals.merge(
        direct_observations,
        on=["order_id", "gps_interval_id"],
        how="inner",
        validate="one_to_one",
        suffixes=("_interval", "_observation"),
    )
    comparisons = (
        (
            "interval_start_time",
            _tolerance(config, "interval_time_abs_s", 1e-6),
        ),
        (
            "interval_end_time",
            _tolerance(config, "interval_time_abs_s", 1e-6),
        ),
        (
            "travel_time_s",
            _tolerance(config, "interval_time_abs_s", 1e-6),
        ),
        (
            "distance_m",
            _tolerance(config, "dynamic_distance_abs_m", 1e-6),
        ),
    )
    paired_values = {
        "interval_start_time": (
            "interval_start_time_interval",
            "interval_start_time_observation",
        ),
        "interval_end_time": (
            "interval_end_time_interval",
            "interval_end_time_observation",
        ),
        "travel_time_s": (
            "direct_observed_travel_time_s",
            "observed_travel_time_s",
        ),
        "distance_m": (
            "direct_observed_distance_m",
            "observed_distance_m",
        ),
    }
    for name, absolute_tolerance in comparisons:
        left_column, right_column = paired_values[name]
        left = _numeric(paired, left_column, "direct interval comparison")
        right = _numeric(paired, right_column, "direct interval comparison")
        matches = np.isclose(
            left,
            right,
            atol=absolute_tolerance,
            rtol=time_rel,
        )
        if not matches.all():
            raise Stage1V3InputError(
                f"direct interval classification disagrees with observation "
                f"on {name} for {int((~matches).sum())} rows"
            )

    grouped = intervals.assign(
        _direct=category_values["direct_observed"].fillna(0.0),
        _supported=category_values["interval_supported"].fillna(0.0),
        _engine=category_values["engine_interpolated"].fillna(0.0),
        _unresolved=category_values["unresolved"].fillna(0.0),
        _total=duration,
    ).groupby("order_id", sort=False).agg(
        direct_observed_time_s_actual=("_direct", "sum"),
        interval_supported_time_s_actual=("_supported", "sum"),
        engine_allocated_only_time_s_actual=("_engine", "sum"),
        unresolved_interval_time_s_actual=("_unresolved", "sum"),
        total_interval_time_s_actual=("_total", "sum"),
    )
    totals = dynamic.merge(
        grouped,
        left_on="order_id",
        right_index=True,
        how="left",
        validate="one_to_one",
    )
    for expected_column in (
        "direct_observed_time_s",
        "interval_supported_time_s",
        "engine_allocated_only_time_s",
        "unresolved_interval_time_s",
        "total_interval_time_s",
    ):
        actual_column = f"{expected_column}_actual"
        actual = _numeric(totals, actual_column, "interval totals")
        expected = _numeric(totals, expected_column, "dynamic_quality")
        matches = np.isclose(
            actual,
            expected,
            atol=_tolerance(config, "dynamic_time_abs_s", 1e-6),
            rtol=_tolerance(config, "dynamic_time_rel", 1e-12),
        )
        if not matches.all():
            raise Stage1V3InputError(
                f"interval classification totals disagree with "
                f"dynamic_quality.{expected_column} for "
                f"{int((~matches).sum())} orders"
            )

    time_error = (
        _numeric(
            totals,
            "direct_observed_time_s_actual",
            "interval totals",
        )
        + _numeric(
            totals,
            "interval_supported_time_s_actual",
            "interval totals",
        )
        + _numeric(
            totals,
            "engine_allocated_only_time_s_actual",
            "interval totals",
        )
        + _numeric(
            totals,
            "unresolved_interval_time_s_actual",
            "interval totals",
        )
        - _numeric(totals, "total_interval_time_s_actual", "interval totals")
    )
    reported_time_error = _numeric(
        totals, "time_conservation_error_s", "dynamic_quality"
    )
    conservation_tolerance = _tolerance(
        config, "dynamic_time_abs_s", 1e-6
    )
    error_match = np.isclose(
        time_error,
        reported_time_error,
        atol=conservation_tolerance,
        rtol=_tolerance(config, "dynamic_time_rel", 1e-12),
    )
    if (
        (np.abs(time_error) > conservation_tolerance).any()
        or not error_match.all()
    ):
        raise Stage1V3InputError(
            "interval time classification is not conserved or disagrees "
            "with dynamic_quality.time_conservation_error_s"
        )


def _validate_turn_movements(
    movements: pd.DataFrame,
    route: pd.DataFrame,
) -> None:
    _require_non_null(
        movements,
        [
            "movement_sequence",
            "from_edge_uid",
            "to_edge_uid",
            "movement_source",
            "movement_quality",
        ],
        "turn_movements",
    )
    sources = set(movements["movement_source"].astype(str).unique())
    unknown = sorted(sources - KNOWN_MOVEMENT_SOURCES)
    if unknown:
        raise Stage1V3InputError(
            f"turn_movements contains unknown movement_source values: {unknown}"
        )
    route_edge_keys = set(
        route[["order_id", "canonical_edge_uid"]].itertuples(
            index=False, name=None
        )
    )
    for column in ("from_edge_uid", "to_edge_uid"):
        mapped_movements = movements.loc[
            ~movements["movement_source"].eq("unmapped_transition")
        ]
        movement_edge_keys = set(
            mapped_movements[["order_id", column]].itertuples(
                index=False, name=None
            )
        )
        unknown_edges = movement_edge_keys - route_edge_keys
        if unknown_edges:
            raise Stage1V3InputError(
                f"turn_movements.{column} contains {len(unknown_edges)} "
                "edge foreign-key failures"
            )
    forbidden_dynamic_fields = (
        "movement_travel_time_s",
        "movement_delay_s",
        "observed_interval_time_s",
        "dynamic_time_source",
    )
    for column in forbidden_dynamic_fields:
        count = int(movements[column].notna().sum())
        if count:
            raise Stage1V3InputError(
                f"turn_movements.{column} must remain null; found "
                f"{count} unsupported dynamic values"
            )


def _validate_route_distance_conservation(
    route: pd.DataFrame,
    traversals: pd.DataFrame,
    dynamic: pd.DataFrame,
    config: Stage1V3Config,
) -> None:
    route_length = _numeric(route, "length_m", "route_parts")
    traversal_length = _numeric(
        traversals, "allocated_distance_m", "link_traversals"
    )
    if (route_length < 0).any() or (traversal_length < 0).any():
        raise Stage1V3InputError(
            "route and traversal physical distances must be non-negative"
        )

    indexed_route = route[
        [
            "order_id",
            "route_sequence",
            "canonical_edge_uid",
            "measurement_source",
        ]
    ].copy()
    indexed_route["_route_length"] = route_length
    indexed_route = indexed_route.sort_values(
        ["order_id", "route_sequence"], kind="stable"
    )
    previous_sequence = indexed_route.groupby(
        "order_id", sort=False
    )["route_sequence"].shift()
    sequence_gap = previous_sequence.notna() & (
        pd.to_numeric(
            indexed_route["route_sequence"], errors="coerce"
        )
        != pd.to_numeric(previous_sequence, errors="coerce") + 1
    )
    if sequence_gap.any():
        raise Stage1V3InputError(
            "route_parts route_sequence must be contiguous within each order"
        )
    indexed_route["_prefix_end"] = indexed_route.groupby(
        "order_id", sort=False
    )["_route_length"].cumsum()
    indexed_route["_prefix_before"] = (
        indexed_route["_prefix_end"] - indexed_route["_route_length"]
    )
    previous_edge = indexed_route.groupby(
        "order_id", sort=False
    )["canonical_edge_uid"].shift()
    first_in_order = indexed_route.groupby(
        "order_id", sort=False
    ).cumcount().eq(0)
    edge_changed = first_in_order | ~_series_matches(
        indexed_route["canonical_edge_uid"],
        previous_edge,
    )
    indexed_route["_canonical_run"] = edge_changed.astype(int).groupby(
        indexed_route["order_id"], sort=False
    ).cumsum()
    previous_source = indexed_route.groupby(
        "order_id", sort=False
    )["measurement_source"].shift()
    source_changed = first_in_order | ~_series_matches(
        indexed_route["measurement_source"],
        previous_source,
    )
    indexed_route["_source_run"] = source_changed.astype(int).groupby(
        indexed_route["order_id"], sort=False
    ).cumsum()

    start_lookup = indexed_route[
        [
            "order_id",
            "route_sequence",
            "_prefix_before",
            "_canonical_run",
            "_source_run",
            "measurement_source",
        ]
    ].rename(
        columns={
            "_canonical_run": "_start_canonical_run",
            "_source_run": "_start_source_run",
            "measurement_source": "_route_measurement_source",
        }
    )
    end_lookup = indexed_route[
        [
            "order_id",
            "route_sequence",
            "_prefix_end",
            "_canonical_run",
            "_source_run",
        ]
    ].rename(
        columns={
            "route_sequence": "route_sequence_end",
            "_canonical_run": "_end_canonical_run",
            "_source_run": "_end_source_run",
        }
    )
    traversal_ranges = traversals[
        [
            "order_id",
            "traversal_id",
            "route_sequence",
            "route_sequence_end",
            "allocated_distance_m",
            "measurement_source",
        ]
    ].merge(
        start_lookup,
        on=["order_id", "route_sequence"],
        how="left",
        validate="many_to_one",
    ).merge(
        end_lookup,
        on=["order_id", "route_sequence_end"],
        how="left",
        validate="many_to_one",
    )
    if traversal_ranges[
        [
            "_prefix_before",
            "_prefix_end",
            "_start_canonical_run",
            "_end_canonical_run",
            "_start_source_run",
            "_end_source_run",
        ]
    ].isna().any().any():
        raise Stage1V3InputError(
            "a traversal range references a missing route_sequence"
        )
    if not traversal_ranges["_start_canonical_run"].eq(
        traversal_ranges["_end_canonical_run"]
    ).all():
        raise Stage1V3InputError(
            "a traversal range crosses canonical edges"
        )
    source_range_match = traversal_ranges["_start_source_run"].eq(
        traversal_ranges["_end_source_run"]
    )
    traversal_source_match = _series_matches(
        traversal_ranges["measurement_source"],
        traversal_ranges["_route_measurement_source"],
    )
    if not source_range_match.all() or not traversal_source_match.all():
        raise Stage1V3InputError(
            "a traversal range disagrees with route_parts measurement_source"
        )
    expected_traversal_distance = (
        pd.to_numeric(traversal_ranges["_prefix_end"], errors="coerce")
        - pd.to_numeric(
            traversal_ranges["_prefix_before"], errors="coerce"
        )
    ).to_numpy(dtype=float)
    reported_traversal_distance = _numeric(
        traversal_ranges,
        "allocated_distance_m",
        "link_traversals",
    )
    tolerance = _tolerance(config, "dynamic_distance_abs_m", 1e-6)
    allocated_match = np.isclose(
        expected_traversal_distance,
        reported_traversal_distance,
        atol=tolerance,
        rtol=_tolerance(config, "dynamic_distance_rel", 1e-12),
    )
    if not allocated_match.all():
        raise Stage1V3InputError(
            "link_traversals.allocated_distance_m differs from its exact "
            f"route range for {int((~allocated_match).sum())} traversals"
        )

    ordered_ranges = traversal_ranges.sort_values(
        ["order_id", "route_sequence", "route_sequence_end", "traversal_id"],
        kind="stable",
    )
    previous_end = ordered_ranges.groupby(
        "order_id", sort=False
    )["route_sequence_end"].shift()
    internal_gap = previous_end.notna() & (
        pd.to_numeric(
            ordered_ranges["route_sequence"], errors="coerce"
        )
        != pd.to_numeric(previous_end, errors="coerce") + 1
    )
    route_bounds = indexed_route.groupby("order_id", sort=False).agg(
        first_route_sequence=("route_sequence", "min"),
        last_route_sequence=("route_sequence", "max"),
    )
    traversal_bounds = ordered_ranges.groupby(
        "order_id", sort=False
    ).agg(
        first_traversal_sequence=("route_sequence", "min"),
        last_traversal_sequence=("route_sequence_end", "max"),
    ).join(route_bounds, how="outer")
    boundary_gap = (
        traversal_bounds["first_traversal_sequence"].ne(
            traversal_bounds["first_route_sequence"]
        )
        | traversal_bounds["last_traversal_sequence"].ne(
            traversal_bounds["last_route_sequence"]
        )
    )
    if internal_gap.any() or boundary_gap.any():
        raise Stage1V3InputError(
            "link_traversals must cover every route_sequence exactly once"
        )

    route_totals = indexed_route.groupby(
        "order_id", sort=False
    )["_route_length"].sum()
    traversal_totals = traversal_ranges.assign(
        _traversal_length=reported_traversal_distance
    ).groupby("order_id", sort=False)["_traversal_length"].sum()
    comparison = dynamic[[
        "order_id",
        "traversal_distance_conservation_error_m",
    ]].merge(
        route_totals.rename("route_distance_m"),
        left_on="order_id",
        right_index=True,
        how="left",
        validate="one_to_one",
    ).merge(
        traversal_totals.rename("traversal_distance_m"),
        left_on="order_id",
        right_index=True,
        how="left",
        validate="one_to_one",
    )
    route_distance = _numeric(
        comparison, "route_distance_m", "route distance totals"
    )
    traversal_distance = _numeric(
        comparison, "traversal_distance_m", "traversal distance totals"
    )
    actual_error = traversal_distance - route_distance
    reported_error = _numeric(
        comparison,
        "traversal_distance_conservation_error_m",
        "dynamic_quality",
    )
    error_match = np.isclose(
        actual_error,
        reported_error,
        atol=tolerance,
        rtol=_tolerance(config, "dynamic_distance_rel", 1e-12),
    )
    if (np.abs(actual_error) > tolerance).any() or not error_match.all():
        raise Stage1V3InputError(
            "traversal distance is not conserved or disagrees with "
            "dynamic_quality.traversal_distance_conservation_error_m"
        )


def _validate_traversal_observation_totals(
    traversals: pd.DataFrame,
    observations: pd.DataFrame,
    config: Stage1V3Config,
) -> None:
    """Ensure traversal summaries are derived only from their direct intervals."""

    traversal_time = _optional_numeric(
        traversals, "observed_travel_time_s", "link_traversals"
    )
    traversal_distance = _optional_numeric(
        traversals, "observed_distance_m", "link_traversals"
    )
    direct_mask = traversals["measurement_source"].astype(str).eq(
        DIRECT_MEASUREMENT_SOURCE
    )
    invalid_direct = direct_mask & (
        traversal_time.isna()
        | traversal_time.le(0)
        | traversal_distance.isna()
        | traversal_distance.lt(0)
    )
    invalid_non_direct = ~direct_mask & (
        traversal_time.notna() | traversal_distance.notna()
    )
    if invalid_direct.any() or invalid_non_direct.any():
        raise Stage1V3InputError(
            "link_traversals observed time/distance must exist only for "
            "direct_observed traversals"
        )

    grouped = observations.groupby(
        ["order_id", "traversal_id"], sort=False
    ).agg(
        interval_time_s=("observed_travel_time_s", "sum"),
        interval_distance_m=("observed_distance_m", "sum"),
    ).reset_index()
    direct = traversals.loc[
        direct_mask,
        [
            "order_id",
            "traversal_id",
            "observed_travel_time_s",
            "observed_distance_m",
        ],
    ]
    direct_keys = set(
        direct[["order_id", "traversal_id"]].itertuples(
            index=False, name=None
        )
    )
    grouped_keys = set(
        grouped[["order_id", "traversal_id"]].itertuples(
            index=False, name=None
        )
    )
    if direct_keys != grouped_keys:
        raise Stage1V3InputError(
            "direct traversal keys do not exactly match grouped direct "
            "interval observations"
        )
    comparison = direct.merge(
        grouped,
        on=["order_id", "traversal_id"],
        how="inner",
        validate="one_to_one",
    )
    time_match = np.isclose(
        _numeric(
            comparison, "observed_travel_time_s", "traversal comparison"
        ),
        _numeric(comparison, "interval_time_s", "traversal comparison"),
        atol=_tolerance(config, "dynamic_time_abs_s", 1e-6),
        rtol=_tolerance(config, "dynamic_time_rel", 1e-12),
    )
    distance_match = np.isclose(
        _numeric(
            comparison, "observed_distance_m", "traversal comparison"
        ),
        _numeric(
            comparison, "interval_distance_m", "traversal comparison"
        ),
        atol=_tolerance(config, "dynamic_distance_abs_m", 1e-6),
        rtol=_tolerance(config, "dynamic_distance_rel", 1e-12),
    )
    if not time_match.all() or not distance_match.all():
        raise Stage1V3InputError(
            "link_traversals direct totals disagree with their interval "
            f"observations: time_failures={int((~time_match).sum())}, "
            f"distance_failures={int((~distance_match).sum())}"
        )


def _validate_direct_intervals(
    observations: pd.DataFrame,
    config: Stage1V3Config,
) -> None:
    _validate_measurement_sources(
        observations, "link_interval_observations"
    )
    source = observations["measurement_source"].astype(str)
    if not source.eq(DIRECT_MEASUREMENT_SOURCE).all():
        invalid = sorted(source[source.ne(DIRECT_MEASUREMENT_SOURCE)].unique())
        raise Stage1V3InputError(
            "link_interval_observations must contain direct labels only; "
            f"found {invalid}"
        )
    _require_strict_true(
        observations["label_valid"],
        "link_interval_observations.label_valid",
    )
    _require_non_null(
        observations,
        [
            "traversal_id",
            "canonical_edge_uid",
            "observed_directed_edge_uid",
            "observed_from_node",
            "observed_to_node",
            "observed_direction",
        ],
        "link_interval_observations",
    )
    if not observations["observed_direction"].astype(str).isin({"F", "R"}).all():
        raise Stage1V3InputError(
            "link_interval_observations actual direction must be F or R"
        )
    if not observations["canonical_mapping_available"].eq(True).all():
        raise Stage1V3InputError(
            "a direct observation is attached to an unmapped traversal"
        )

    start = _numeric(
        observations,
        "interval_start_time",
        "link_interval_observations",
    )
    end = _numeric(
        observations,
        "interval_end_time",
        "link_interval_observations",
    )
    duration = _numeric(
        observations,
        "observed_travel_time_s",
        "link_interval_observations",
    )
    distance = _numeric(
        observations,
        "observed_distance_m",
        "link_interval_observations",
    )
    speed = _numeric(
        observations,
        "observed_speed_mps",
        "link_interval_observations",
    )
    if (end <= start).any() or (duration <= 0).any():
        raise Stage1V3InputError(
            "direct intervals must have positive timestamp and observed duration"
        )
    if (distance < 0).any() or (speed < 0).any():
        raise Stage1V3InputError(
            "direct interval distance and speed must be non-negative"
        )

    time_abs = _tolerance(config, "interval_time_abs_s", 1e-6)
    time_rel = _tolerance(config, "interval_time_rel", 1e-12)
    if not np.isclose(
        end - start,
        duration,
        atol=time_abs,
        rtol=time_rel,
    ).all():
        failures = ~np.isclose(
            end - start,
            duration,
            atol=time_abs,
            rtol=time_rel,
        )
        raise Stage1V3InputError(
            "interval timestamp duration differs from observed_travel_time_s "
            f"for {int(failures.sum())} rows"
        )

    expected_speed = distance / duration
    speed_abs = _tolerance(config, "interval_speed_abs_mps", 1e-6)
    speed_rel = _tolerance(config, "interval_speed_rel", 1e-9)
    speed_match = np.isclose(
        speed,
        expected_speed,
        atol=speed_abs,
        rtol=speed_rel,
    )
    if not speed_match.all():
        raise Stage1V3InputError(
            "observed_speed_mps differs from distance/time for "
            f"{int((~speed_match).sum())} direct intervals"
        )

    overlap_abs = _tolerance(config, "interval_overlap_abs_s", 1e-6)
    ordered = observations[
        [
            "order_id",
            "traversal_id",
            "gps_interval_id",
            "interval_start_time",
            "interval_end_time",
        ]
    ].copy()
    ordered["_start"] = pd.to_numeric(
        ordered["interval_start_time"], errors="coerce"
    )
    ordered["_end"] = pd.to_numeric(
        ordered["interval_end_time"], errors="coerce"
    )
    ordered = ordered.sort_values(
        [
            "order_id",
            "traversal_id",
            "_start",
            "_end",
            "gps_interval_id",
        ],
        kind="stable",
    )
    ordered["_maximum_end_so_far"] = ordered.groupby(
        ["order_id", "traversal_id"], sort=False
    )["_end"].cummax()
    previous_maximum_end = ordered.groupby(
        ["order_id", "traversal_id"], sort=False
    )["_maximum_end_so_far"].shift()
    overlaps = previous_maximum_end.notna() & (
        ordered["_start"] < previous_maximum_end - overlap_abs
    )
    if overlaps.any():
        raise Stage1V3InputError(
            "direct intervals overlap within the same traversal for "
            f"{int(overlaps.sum())} rows"
        )


def _require_zero(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        values = _numeric(frame, column, "dynamic_quality")
        failures = ~np.isclose(values, 0.0, atol=0.0, rtol=0.0)
        if failures.any():
            raise Stage1V3InputError(
                f"dynamic_quality.{column} has {int(failures.sum())} violations"
            )


def _require_true(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        _require_strict_true(
            frame[column],
            f"dynamic_quality.{column}",
        )


def _validate_dynamic_accounting(
    observations: pd.DataFrame,
    dynamic: pd.DataFrame,
    config: Stage1V3Config,
) -> None:
    _require_zero(
        dynamic,
        (
            "duplicate_interval_allocation_count",
            "non_direct_observed_time_violation_count",
            "unresolved_duplicate_allocation_count",
            "traversal_duplicate_distance_count",
        ),
    )
    _require_true(
        dynamic,
        ("time_conservation_valid", "distance_conservation_valid"),
    )
    statuses = set(dynamic["dynamic_status"].dropna().astype(str).unique())
    unknown_statuses = sorted(statuses - DYNAMIC_STATUSES)
    if dynamic["dynamic_status"].isna().any() or unknown_statuses:
        raise Stage1V3InputError(
            f"dynamic_quality contains invalid dynamic_status values: "
            f"{unknown_statuses}"
        )

    grouped = observations.groupby("order_id", sort=False).agg(
        observed_time_sum=("observed_travel_time_s", "sum"),
        observed_distance_sum=("observed_distance_m", "sum"),
        observed_interval_count=("gps_interval_id", "size"),
        observed_edge_count=("canonical_edge_uid", "nunique"),
    )
    comparison = dynamic.merge(
        grouped,
        left_on="order_id",
        right_index=True,
        how="left",
        validate="one_to_one",
    )
    if comparison[
        [
            "observed_time_sum",
            "observed_distance_sum",
            "observed_interval_count",
            "observed_edge_count",
        ]
    ].isna().any().any():
        raise Stage1V3InputError(
            "an accepted order has no direct interval accounting rows"
        )

    time_actual = _numeric(
        comparison, "observed_time_sum", "dynamic comparison"
    )
    time_expected = _numeric(
        comparison, "direct_observed_time_s", "dynamic_quality"
    )
    distance_actual = _numeric(
        comparison, "observed_distance_sum", "dynamic comparison"
    )
    distance_expected = _numeric(
        comparison, "direct_observed_distance_m", "dynamic_quality"
    )
    time_match = np.isclose(
        time_actual,
        time_expected,
        atol=_tolerance(config, "dynamic_time_abs_s", 1e-6),
        rtol=_tolerance(config, "dynamic_time_rel", 1e-12),
    )
    distance_match = np.isclose(
        distance_actual,
        distance_expected,
        atol=_tolerance(config, "dynamic_distance_abs_m", 1e-6),
        rtol=_tolerance(config, "dynamic_distance_rel", 1e-12),
    )
    if not time_match.all() or not distance_match.all():
        raise Stage1V3InputError(
            "direct interval totals disagree with dynamic_quality: "
            f"time_failures={int((~time_match).sum())}, "
            f"distance_failures={int((~distance_match).sum())}"
        )

    interval_count = pd.to_numeric(
        comparison["observed_interval_count"], errors="coerce"
    ).to_numpy(dtype=float)
    expected_interval_count = pd.to_numeric(
        comparison["valid_direct_interval_count"], errors="coerce"
    ).to_numpy(dtype=float)
    edge_count = pd.to_numeric(
        comparison["observed_edge_count"], errors="coerce"
    ).to_numpy(dtype=float)
    expected_edge_count = pd.to_numeric(
        comparison["unique_timed_edge_count"], errors="coerce"
    ).to_numpy(dtype=float)
    count_arrays = (
        interval_count,
        expected_interval_count,
        edge_count,
        expected_edge_count,
    )
    if any(
        (~np.isfinite(values) | (values < 0) | (values != np.floor(values))).any()
        for values in count_arrays
    ):
        raise Stage1V3InputError(
            "direct interval and unique timed-edge counts must be "
            "non-negative integers"
        )
    if (
        not np.array_equal(interval_count, expected_interval_count)
        or not np.array_equal(edge_count, expected_edge_count)
    ):
        raise Stage1V3InputError(
            "direct interval or unique timed-edge counts disagree with "
            "dynamic_quality"
        )
    coverage = config.section("coverage")
    minimum_intervals = coverage.get("minimum_direct_interval_count", 8)
    minimum_edges = coverage.get("minimum_unique_timed_edge_count", 5)
    for name, value in (
        ("coverage.minimum_direct_interval_count", minimum_intervals),
        ("coverage.minimum_unique_timed_edge_count", minimum_edges),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise Stage1V3InputError(
                f"{name} must be a non-negative integer"
            )
    if (
        (expected_interval_count < minimum_intervals).any()
        or (expected_edge_count < minimum_edges).any()
    ):
        raise Stage1V3InputError(
            "accepted order violates frozen direct-label minima: "
            f"minimum_intervals={minimum_intervals}, "
            f"minimum_edges={minimum_edges}"
        )


def load_stage0_bucket(
    ref: BucketRef,
    config: Stage1V3Config,
) -> Stage0Bucket:
    """Load and fully validate the ten Stage 0 products used by Stage 1 v3."""

    validate_split_config(config)
    _validate_manifest(ref)
    if ref.split not in _KNOWN_SPLITS:
        raise Stage1V3InputError(f"unknown BucketRef split: {ref.split!r}")
    if ref.date not in set(_split_dates(config)[ref.split]):
        raise Stage1V3InputError(
            f"BucketRef date {ref.date!r} is not allowed for split {ref.split!r}"
        )
    resolved_bucket = ref.path.resolve()
    if len(resolved_bucket.parents) < 3:
        raise Stage1V3InputError(f"BucketRef path is too shallow: {ref.path}")
    input_root = resolved_bucket.parents[2]
    if input_root.name != INPUT_ROOT_NAME:
        raise Stage1V3InputError(
            f"BucketRef must be below a directory named {INPUT_ROOT_NAME!r}: "
            f"{ref.path}"
        )
    expected_path = (
        input_root
        / f"split={ref.split}"
        / f"date={ref.date}"
        / f"bucket={ref.bucket:05d}"
    )
    if resolved_bucket != expected_path.resolve():
        raise Stage1V3InputError(
            f"BucketRef path does not match its split/date/bucket: {ref.path}"
        )
    disk_manifest = _read_manifest(resolved_bucket / "manifest.json")
    if disk_manifest != ref.manifest:
        raise Stage1V3InputError(
            f"{resolved_bucket}: BucketRef manifest differs from the on-disk manifest"
        )

    products = {
        product: _read_product(ref, product)
        for product in REQUIRED_PRODUCTS
    }
    _normalize_nullable_dtypes(products)
    _enrich_direction_lineage(products)
    _fill_nullable_order_endpoints(products)
    bucket = Stage0Bucket(
        order_base=products["order_base"],
        route_parts=products["route_parts"],
        link_traversals=products["link_traversals"],
        link_interval_observations=products[
            "link_interval_observations"
        ],
        interval_measurements=products["interval_measurements"],
        turn_movements=products["turn_movements"],
        gps_quality=products["gps_quality"],
        route_quality=products["route_quality"],
        dynamic_quality=products["dynamic_quality"],
        canonical_quality=products["canonical_quality"],
    )

    order_ids = _validate_order_base(ref, bucket.order_base)
    _validate_measurement_sources(bucket.route_parts, "route_parts")
    _validate_measurement_sources(
        bucket.link_traversals, "link_traversals"
    )
    _validate_direct_intervals(
        bucket.link_interval_observations,
        config,
    )
    _validate_primary_and_foreign_keys(bucket, order_ids)
    _validate_traversal_observation_totals(
        bucket.link_traversals,
        bucket.link_interval_observations,
        config,
    )
    _validate_interval_classification(
        bucket.interval_measurements,
        bucket.link_interval_observations,
        bucket.dynamic_quality,
        config,
    )
    _validate_turn_movements(
        bucket.turn_movements,
        bucket.route_parts,
    )
    _validate_route_distance_conservation(
        bucket.route_parts,
        bucket.link_traversals,
        bucket.dynamic_quality,
        config,
    )
    _validate_dynamic_accounting(
        bucket.link_interval_observations,
        bucket.dynamic_quality,
        config,
    )
    return bucket


def load_stage0_fit_bucket(
    ref: BucketRef,
    config: Stage1V3Config,
) -> Stage0FitBucket:
    """Load only fit products after a bound global preflight has passed.

    The caller must validate the preflight report and its product hashes once
    before iterating. This loader still checks the bucket manifest, required
    columns, row counts, direction joins, and direct-label arithmetic.
    """

    validate_split_config(config)
    _validate_manifest(ref)
    disk_manifest = _read_manifest(ref.path / "manifest.json")
    if disk_manifest != ref.manifest:
        raise Stage1V3InputError(
            f"{ref.path}: BucketRef manifest differs from the on-disk manifest"
        )
    products = {
        product: _read_product(ref, product)
        for product in (
            "route_parts",
            "link_traversals",
            "link_interval_observations",
        )
    }
    for column in _NULLABLE_NODE_COLUMNS["route_parts"]:
        products["route_parts"][column] = _nullable_int64(
            products["route_parts"][column],
            name=f"route_parts.{column}",
        )
    _enrich_direction_lineage(products)
    _validate_measurement_sources(products["route_parts"], "route_parts")
    _validate_measurement_sources(
        products["link_traversals"], "link_traversals"
    )
    _validate_direct_intervals(
        products["link_interval_observations"],
        config,
    )
    return Stage0FitBucket(
        route_parts=products["route_parts"],
        link_traversals=products["link_traversals"],
        link_interval_observations=products[
            "link_interval_observations"
        ],
    )


def load_stage0_fit_route_parts(ref: BucketRef) -> pd.DataFrame:
    """Read only the route columns required for a directed fit catalog."""

    _validate_manifest(ref)
    columns = (
        "order_id",
        "route_sequence",
        "canonical_edge_uid",
        "canonical_from_node",
        "canonical_to_node",
        "canonical_traversal_direction",
        "mapping_status",
        "traversed_against_osm_oneway",
        "osm_oneway",
        "canonical_highway",
        "canonical_length_m",
        "road_class",
        "bridge",
        "tunnel",
    )
    route = _read_product_subset(ref, "route_parts", columns)
    for column in ("canonical_from_node", "canonical_to_node"):
        route[column] = _nullable_int64(
            route[column],
            name=f"route_parts.{column}",
        )
    _enrich_route_direction_lineage(route)
    return route


def derive_movement_direction_context(
    movements: pd.DataFrame,
    route_parts: pd.DataFrame,
) -> pd.DataFrame:
    """Map movement sequence boundaries to actual directed route identities."""

    result = movements.copy()
    route = route_parts[
        [
            "order_id",
            "route_sequence",
            "observed_directed_edge_uid",
        ]
    ]
    from_context = route.rename(
        columns={
            "route_sequence": "movement_sequence",
            "observed_directed_edge_uid": (
                "observed_from_directed_edge_uid"
            ),
        }
    )
    to_context = route.assign(
        movement_sequence=pd.to_numeric(
            route["route_sequence"], errors="raise"
        ).astype(int)
        - 1
    )[
        ["order_id", "movement_sequence", "observed_directed_edge_uid"]
    ].rename(
        columns={
            "observed_directed_edge_uid": "observed_to_directed_edge_uid"
        }
    )
    result = result.merge(
        from_context,
        on=["order_id", "movement_sequence"],
        how="left",
        validate="one_to_one",
    ).merge(
        to_context,
        on=["order_id", "movement_sequence"],
        how="left",
        validate="one_to_one",
    )
    result["movement_direction_mapping_available"] = (
        result["observed_from_directed_edge_uid"].notna()
        & result["observed_to_directed_edge_uid"].notna()
    )
    result["movement_lineage_only"] = (
        ~result["movement_direction_mapping_available"]
    )
    return result
