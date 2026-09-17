"""Frozen configuration contract for the Stage 1 v3 input adapter.

Stage 1 v3 deliberately has one temporal split.  A configuration may tune
numeric validation tolerances, but it cannot redefine the train, validation,
test, or reference-fit dates without changing this module and the schema
version.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stage1.v3.schema import (
    ContractError,
    OUTPUT_PRIMARY_KEYS,
    OUTPUT_REQUIRED_COLUMNS,
)


STAGE1_V3_SCHEMA_VERSION = "stage1_label_schema_v3"

FROZEN_TRAIN_DATES = tuple(f"201610{day:02d}" for day in range(9, 25))
FROZEN_VALIDATION_DATES = ("20161025", "20161026", "20161027")
FROZEN_TEST_DATE = "20161031"
FROZEN_REFERENCE_FIT_DATES = FROZEN_TRAIN_DATES
FROZEN_STAGE0_RELEASE = {
    "stage0_tag": "stage0-v6-final",
    "stage0_tag_commit": "729275d81ec5dc224ac0967a6e600457764607b8",
    "stage0_source_content_hash": "a5e482f4a0d2b607",
}


class Stage1V3ConfigError(ContractError):
    """Raised when a Stage 1 v3 configuration violates the frozen contract."""


class _FrozenDict(dict):
    """A JSON-compatible dictionary that cannot be mutated in place."""

    def _immutable(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("Stage1V3Config.data is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[str, Any]:
        return {
            copy.deepcopy(key, memo): copy.deepcopy(value, memo)
            for key, value in self.items()
        }


class _FrozenList(list):
    """A JSON-compatible list that cannot be mutated in place."""

    def _immutable(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("Stage1V3Config.data is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable
    __iadd__ = _immutable
    __imul__ = _immutable

    def __deepcopy__(self, memo: dict[int, Any]) -> list[Any]:
        return [copy.deepcopy(value, memo) for value in self]


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return _FrozenDict(
            {
                copy.deepcopy(key): _freeze(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, list):
        return _FrozenList(_freeze(item) for item in value)
    return copy.deepcopy(value)


def _canonical_digest(data: dict[str, Any]) -> str:
    payload = json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class Stage1V3Config:
    """Immutable Stage 1 v3 configuration.

    ``data`` is defensively copied and recursively frozen on construction.
    :meth:`section` returns a mutable deep copy for consumers, while ``digest``
    always describes the immutable accepted configuration.
    """

    data: dict[str, Any]
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.data, dict):
            raise Stage1V3ConfigError("Stage1 v3 config root must be a mapping")
        copied = _freeze(self.data)
        object.__setattr__(self, "data", copied)
        object.__setattr__(self, "digest", _canonical_digest(copied))
        if self.schema_version != STAGE1_V3_SCHEMA_VERSION:
            raise Stage1V3ConfigError(
                "schema_version must be "
                f"{STAGE1_V3_SCHEMA_VERSION!r}, got {self.schema_version!r}"
            )

    @property
    def schema_version(self) -> str:
        value = self.data.get("schema_version")
        return str(value) if value is not None else ""

    def section(self, name: str) -> dict[str, Any]:
        value = self.data.get(name, {})
        if not isinstance(value, dict):
            raise Stage1V3ConfigError(f"config section {name!r} must be a mapping")
        return copy.deepcopy(value)

    @property
    def train_dates(self) -> tuple[str, ...]:
        return _date_tuple(self.section("split"), "train_dates")

    @property
    def validation_dates(self) -> tuple[str, ...]:
        return _date_tuple(self.section("split"), "validation_dates")

    @property
    def test_date(self) -> str:
        value = self.section("split").get("test_date")
        if not isinstance(value, str):
            raise Stage1V3ConfigError("split.test_date must be a YYYYMMDD string")
        return value

    @property
    def reference_fit_dates(self) -> tuple[str, ...]:
        return _date_tuple(self.section("split"), "reference_fit_dates")


def _date_tuple(section: dict[str, Any], key: str) -> tuple[str, ...]:
    value = section.get(key)
    if not isinstance(value, list):
        raise Stage1V3ConfigError(f"split.{key} must be a list of YYYYMMDD strings")
    if not all(isinstance(item, str) for item in value):
        raise Stage1V3ConfigError(f"split.{key} must contain only strings")
    return tuple(value)


def validate_split_config(config: Stage1V3Config) -> None:
    """Reject any temporal split other than the frozen final Stage 0 split."""

    expected = {
        "train_dates": FROZEN_TRAIN_DATES,
        "validation_dates": FROZEN_VALIDATION_DATES,
        "test_date": FROZEN_TEST_DATE,
        "reference_fit_dates": FROZEN_REFERENCE_FIT_DATES,
    }
    actual = {
        "train_dates": config.train_dates,
        "validation_dates": config.validation_dates,
        "test_date": config.test_date,
        "reference_fit_dates": config.reference_fit_dates,
    }
    failures = [
        f"split.{name}: expected {expected[name]!r}, got {actual[name]!r}"
        for name in expected
        if actual[name] != expected[name]
    ]
    if failures:
        raise Stage1V3ConfigError(
            "Stage1 v3 temporal split is frozen; " + "; ".join(failures)
        )

    all_dates = (
        set(config.train_dates)
        | set(config.validation_dates)
        | {config.test_date}
    )
    expected_count = (
        len(config.train_dates) + len(config.validation_dates) + 1
    )
    if len(all_dates) != expected_count:
        raise Stage1V3ConfigError("Stage1 v3 split dates must be disjoint")


def _number(
    section: dict[str, Any],
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    key = name.rsplit(".", 1)[-1]
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Stage1V3ConfigError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise Stage1V3ConfigError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise Stage1V3ConfigError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise Stage1V3ConfigError(f"{name} must be <= {maximum}")
    return result


def _integer(
    section: dict[str, Any],
    name: str,
    *,
    minimum: int,
) -> int:
    key = name.rsplit(".", 1)[-1]
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Stage1V3ConfigError(f"{name} must be an integer >= {minimum}")
    return value


def validate_config(config: Stage1V3Config) -> None:
    """Validate every executable threshold in addition to the frozen split."""

    validate_split_config(config)
    status = config.data.get("status")
    if status not in {"review_candidate", "frozen_for_execution"}:
        raise Stage1V3ConfigError(
            "status must be review_candidate or frozen_for_execution"
        )
    if config.data.get("core_composite_status") != "disabled":
        raise Stage1V3ConfigError("Stage1 v3 core composite must remain disabled")
    if config.section("time").get("timezone") != "Asia/Shanghai":
        raise Stage1V3ConfigError("time.timezone must be Asia/Shanghai")
    if (
        config.data.get("label_temporality")
        != "retrospective_post_trip_realized_label"
    ):
        raise Stage1V3ConfigError(
            "Stage1 v3 labels must be declared retrospective post-trip"
        )
    if config.section("stage0_release") != FROZEN_STAGE0_RELEASE:
        raise Stage1V3ConfigError(
            "stage0_release must identify the frozen final Stage0 tag/content"
        )

    direct = config.section("direct")
    for legacy_section in ("tolerances", "validation"):
        if config.data.get(legacy_section) not in (None, {}):
            raise Stage1V3ConfigError(
                f"legacy {legacy_section!r} overrides are forbidden in v3"
            )
    for name in (
        "duration_tolerance_s",
        "distance_identity_tolerance_m",
        "speed_tolerance_mps",
        "interval_time_rel",
        "interval_speed_rel",
        "dynamic_time_rel",
        "dynamic_distance_rel",
    ):
        _number(direct, f"direct.{name}", minimum=0.0)

    lcs = config.section("lcs")
    _integer(
        lcs,
        "lcs.minimum_direct_intervals_per_traversal",
        minimum=2,
    )
    _integer(lcs, "lcs.minimum_acceleration_pairs", minimum=1)
    for name in (
        "minimum_observed_time_s",
        "minimum_direct_observed_distance_m",
        "maximum_adjacent_gap_s",
        "stop_speed_mps",
    ):
        _number(lcs, f"lcs.{name}", minimum=0.0)
    for name in (
        "low_speed_mps",
        "speed_cv_scale",
        "acceleration_rms_scale_mps2",
        "maximum_physical_speed_mps",
        "maximum_absolute_acceleration_mps2",
    ):
        _number(lcs, f"lcs.{name}", minimum=1e-12)
    if float(lcs["stop_speed_mps"]) >= float(lcs["low_speed_mps"]):
        raise Stage1V3ConfigError("LCS requires stop_speed_mps < low_speed_mps")
    if float(lcs["low_speed_mps"]) > float(lcs["maximum_physical_speed_mps"]):
        raise Stage1V3ConfigError(
            "LCS low_speed_mps cannot exceed maximum_physical_speed_mps"
        )
    component_names = (
        "crawl_time_share",
        "stop_time_share",
        "speed_cv_bounded",
        "acceleration_rms_bounded",
    )
    components = lcs.get("components")
    if not isinstance(components, dict) or set(components) != set(component_names):
        raise Stage1V3ConfigError(
            "lcs.components must define exactly the four frozen components"
        )
    weights: list[float] = []
    for component in component_names:
        details = components.get(component)
        if not isinstance(details, dict):
            raise Stage1V3ConfigError(
                f"lcs.components.{component} must be a mapping"
            )
        weights.append(
            _number(
                details,
                f"lcs.components.{component}.weight",
                minimum=0.0,
                maximum=1.0,
            )
        )
    if abs(sum(weights) - 1.0) > 1e-12:
        raise Stage1V3ConfigError("LCS component weights must sum to one")

    rts = config.section("rts")
    _number(rts, "rts.minimum_direct_observed_time_s", minimum=1e-12)
    rts_distance = _number(
        rts,
        "rts.minimum_direct_observed_distance_m",
        minimum=1e-12,
    )
    rts_maximum_speed = _number(
        rts,
        "rts.maximum_direct_speed_mps",
        minimum=1e-12,
    )
    _number(
        rts,
        "rts.tail_event_percentile_threshold",
        minimum=0.0,
        maximum=1.0,
    )
    rts_reference_support = _integer(
        rts,
        "rts.minimum_reference_sample_size",
        minimum=1,
    )

    reference = config.section("reference")
    reference_distance = _number(
        reference,
        "reference.minimum_observed_distance_m",
        minimum=1e-12,
    )
    reference_maximum_speed = _number(
        reference,
        "reference.maximum_direct_speed_mps",
        minimum=1e-12,
    )
    if abs(reference_distance - rts_distance) > 1e-12:
        raise Stage1V3ConfigError(
            "reference and RTS minimum observed distances must be identical"
        )
    lcs_maximum_speed = float(
        config.section("lcs")["maximum_physical_speed_mps"]
    )
    if (
        abs(reference_maximum_speed - rts_maximum_speed) > 1e-12
        or abs(rts_maximum_speed - lcs_maximum_speed) > 1e-12
    ):
        raise Stage1V3ConfigError(
            "LCS, RTS, and reference maximum direct speeds must be identical"
        )
    lower = _number(
        reference,
        "reference.histogram_min_sec_per_m",
        minimum=1e-12,
    )
    upper = _number(
        reference,
        "reference.histogram_max_sec_per_m",
        minimum=1e-12,
    )
    if lower >= upper:
        raise Stage1V3ConfigError("reference histogram minimum must be below maximum")
    _integer(reference, "reference.histogram_bins", minimum=2)
    reference_support = _integer(
        reference,
        "reference.minimum_cohort_support",
        minimum=1,
    )
    if reference_support != rts_reference_support:
        raise Stage1V3ConfigError(
            "rts.minimum_reference_sample_size must equal "
            "reference.minimum_cohort_support"
        )
    _number(reference, "reference.quantile", minimum=0.0, maximum=1.0)
    clip = rts.get("sec_per_m_clip")
    if (
        not isinstance(clip, list)
        or len(clip) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in clip
        )
    ):
        raise Stage1V3ConfigError(
            "rts.sec_per_m_clip must contain two finite numeric bounds"
        )
    if (
        abs(float(clip[0]) - lower) > 1e-12
        or abs(float(clip[1]) - upper) > 1e-12
    ):
        raise Stage1V3ConfigError(
            "rts.sec_per_m_clip must equal the reference histogram bounds"
        )

    normalization = config.section("normalization")
    _integer(normalization, "normalization.raw_bins", minimum=2)
    normalization_support = _integer(
        normalization,
        "normalization.minimum_cohort_support",
        minimum=1,
    )
    _number(
        config.section("aggregation"),
        "aggregation.tail_percentile_threshold",
        minimum=0.0,
        maximum=1.0,
    )
    cohort = config.section("cohort_reference")
    if _integer(
        cohort,
        "cohort_reference.time_bin_minutes",
        minimum=1,
    ) != 30:
        raise Stage1V3ConfigError(
            "cohort_reference.time_bin_minutes is frozen at 30"
        )
    cohort_support = _integer(
        cohort,
        "cohort_reference.minimum_sample_size",
        minimum=1,
    )
    if (
        cohort_support != reference_support
        or cohort_support != normalization_support
    ):
        raise Stage1V3ConfigError(
            "cohort/reference/normalization minimum supports must agree"
        )
    fallback = cohort.get("fallback")
    expected_fallback = (
        "edge_time_weekday",
        "edge_peak",
        "edge",
        "highway_time_weekday",
        "highway",
        "global",
    )
    if (
        not isinstance(fallback, list)
        or tuple(
            item.get("name") if isinstance(item, dict) else None
            for item in fallback
        )
        != expected_fallback
    ):
        raise Stage1V3ConfigError("cohort_reference fallback order is frozen")
    expected_fallback_keys = (
        ("observed_directed_edge_uid", "time_bin_30m", "weekday_type"),
        ("observed_directed_edge_uid", "peak_offpeak"),
        ("observed_directed_edge_uid",),
        ("canonical_highway", "time_bin_30m", "weekday_type"),
        ("canonical_highway",),
        ("global",),
    )
    if tuple(
        tuple(item.get("key", ())) if isinstance(item, dict) else ()
        for item in fallback
    ) != expected_fallback_keys:
        raise Stage1V3ConfigError(
            "cohort_reference edge keys must use actual directed identity"
        )
    if (
        cohort.get("fallback_policy")
        != "first_level_meeting_minimum_sample_size_else_global_if_nonempty_else_na"
    ):
        raise Stage1V3ConfigError("cohort_reference fallback policy is frozen")
    if cohort.get("peak_windows_local") != [
        ["07:00", "09:30"],
        ["17:00", "19:30"],
    ]:
        raise Stage1V3ConfigError(
            "cohort_reference peak windows are frozen for v3"
        )
    if cohort.get("train_reference_application") != "leave_one_out":
        raise Stage1V3ConfigError(
            "train reference application must be leave-one-out"
        )
    if (
        cohort.get("validation_test_reference_application")
        != "full_train_frozen"
    ):
        raise Stage1V3ConfigError(
            "validation/test reference application must use full frozen train"
        )
    if (
        cohort.get("raw_cdf_application")
        != "full_train_empirical_self_rank_for_train"
    ):
        raise Stage1V3ConfigError("raw CDF application policy is frozen")

    coverage = config.section("coverage")
    _integer(coverage, "coverage.minimum_direct_interval_count", minimum=1)
    _integer(coverage, "coverage.minimum_unique_timed_edge_count", minimum=1)
    _number(
        coverage,
        "coverage.minimum_observed_time_share",
        minimum=0.0,
        maximum=1.0,
    )
    support = config.section("support")
    for name in (
        "minimum_edge_observations",
        "minimum_edge_hour_observations",
        "minimum_fallback_observations",
    ):
        _integer(support, f"support.{name}", minimum=1)
    if support.get("fit_scope") != "train_only":
        raise Stage1V3ConfigError("support.fit_scope must be train_only")
    if support.get("fallback_order") != [
        "edge_hour",
        "highway_hour",
        "spatial_neighbor",
        "global_hour",
        "unavailable",
    ]:
        raise Stage1V3ConfigError("support fallback order is frozen")
    if (
        support.get("upper_region_usage")
        != "audit_only_not_a_model_fallback"
    ):
        raise Stage1V3ConfigError(
            "connected-component upper regions are audit-only"
        )
    if (
        support.get("validation_test_policy")
        != "apply_frozen_train_support_only"
    ):
        raise Stage1V3ConfigError(
            "validation/test support must use frozen Train counts"
        )
    preflight = config.section("preflight")
    if preflight.get("expected_order_count") != 220000:
        raise Stage1V3ConfigError("preflight expected order count must be 220000")
    if preflight.get("expected_split_counts") != {
        "train": 160000,
        "validation": 30000,
        "test": 30000,
    }:
        raise Stage1V3ConfigError("preflight split counts differ from the freeze")
    _number(
        coverage,
        "coverage.minimum_observed_distance_share",
        minimum=0.0,
        maximum=1.0,
    )
    for dimension in ("iis", "pmis"):
        section = config.section(dimension)
        if section.get("status") != "unavailable" or section.get("available") is not False:
            raise Stage1V3ConfigError(
                f"{dimension} must remain explicitly unavailable in v3"
            )
    gns = config.section("gns")
    if gns.get("status") != "external_static_extension":
        raise Stage1V3ConfigError(
            "GNS must remain a separately frozen static extension"
        )
    if gns.get("core_label_role") != "excluded":
        raise Stage1V3ConfigError("GNS must remain excluded from the v3 core label")
    outputs = config.section("outputs")
    for product, required_columns in OUTPUT_REQUIRED_COLUMNS.items():
        contract = outputs.get(product)
        if not isinstance(contract, dict):
            raise Stage1V3ConfigError(
                f"outputs.{product} must be a mapping"
            )
        if tuple(contract.get("primary_key", [])) != tuple(
            OUTPUT_PRIMARY_KEYS[product]
        ):
            raise Stage1V3ConfigError(
                f"outputs.{product}.primary_key differs from the code contract"
            )
        if set(contract.get("required_fields", [])) != set(required_columns):
            raise Stage1V3ConfigError(
                f"outputs.{product}.required_fields differs from the code contract"
            )
    if status == "frozen_for_execution":
        candidate_sections = (
            "direct",
            "lcs",
            "rts",
            "reference",
            "normalization",
            "aggregation",
            "cohort_reference",
            "coverage",
            "support",
        )
        unfrozen = [
            name
            for name in candidate_sections
            if config.section(name).get("threshold_status") != "frozen"
        ]
        if unfrozen:
            raise Stage1V3ConfigError(
                f"frozen execution config has unfrozen thresholds: {unfrozen}"
            )


def load_config(path: str | Path) -> Stage1V3Config:
    """Load a JSON or YAML Stage 1 v3 configuration without modifying it."""

    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        parsed = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - environment diagnostic
            raise Stage1V3ConfigError(
                "PyYAML is required to load a non-JSON Stage1 v3 config"
            ) from exc
        parsed = yaml.safe_load(text)
    if not isinstance(parsed, dict):
        raise Stage1V3ConfigError(f"{source}: config root must be a mapping")
    return Stage1V3Config(parsed)
