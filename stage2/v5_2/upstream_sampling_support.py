"""Pure support functions for the Stage 0/1 upstream sampling audit."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from stage0.v5.config import stable_hash

from .contracts import Stage2V52ContractError


TARGETS = ("crawl", "stop", "speed_cv", "acceleration_rms")
TARGET_VALUE_COLUMNS = {
    "crawl": "crawl_time_share",
    "stop": "stop_time_share",
    "speed_cv": "speed_cv_bounded",
    "acceleration_rms": "acceleration_rms_bounded",
}
TARGET_VALID_COLUMNS = {
    target: f"{target}_target_valid" for target in TARGETS
}
SUPPORT_GROUPS = ("unseen", "low", "medium", "high")

REJECTION_TOKEN_GROUPS = {
    "TOO_FEW_VALID_POINTS": "GPS/PREFILTER",
    "DURATION_TOO_SHORT": "GPS/PREFILTER",
    "INVALID_COORDINATE_SHARE": "GPS/PREFILTER",
    "TIMESTAMP_SEVERELY_REVERSED": "GPS/PREFILTER",
    "DUPLICATE_POINT_SHARE": "GPS/PREFILTER",
    "OD_DISTANCE_TOO_SHORT": "GPS/PREFILTER",
    "IMPOSSIBLE_SPEED_DOMINATES": "GPS/PREFILTER",
    "GPS_NOT_CORE_ELIGIBLE": "GPS/PREFILTER",
    "LOCAL_OUTLIER_AFFECTS_CORRIDOR": "GPS/PREFILTER",
    "MISSING_MATERIALIZED_POINTS": "GPS/PREFILTER",
    "ROUTE_NOT_PASS": "ROUTE/MAP-MATCH",
    "CANONICAL_NOT_RESOLVED": "CANONICAL/NETWORK",
    "DYNAMIC_UNUSABLE": "DYNAMIC/SUPERVISION-RELATED",
    "TIME_CONSERVATION_FAILURE": "DYNAMIC/SUPERVISION-RELATED",
    "DISTANCE_CONSERVATION_FAILURE": "DYNAMIC/SUPERVISION-RELATED",
    "INSUFFICIENT_DIRECT_INTERVALS": "DYNAMIC/SUPERVISION-RELATED",
    "INSUFFICIENT_TIMED_EDGES": "DYNAMIC/SUPERVISION-RELATED",
}
REJECTION_GROUPS = (
    "GPS/PREFILTER",
    "ROUTE/MAP-MATCH",
    "CANONICAL/NETWORK",
    "DYNAMIC/SUPERVISION-RELATED",
)


def selection_hex(date: str, order_id: str, seed: int) -> str:
    """Return the exact Stage 0 production priority hash."""

    return f"{stable_hash(str(date), str(order_id), seed=int(seed)):016x}"


def assert_selection_hash_contract(
    frame: pd.DataFrame, *, date: str, seed: int
) -> None:
    required = {"order_id", "selection_hash"}
    missing = required - set(frame.columns)
    if missing:
        raise Stage2V52ContractError(
            f"candidate manifest lacks selection fields: {sorted(missing)}"
        )
    expected = frame["order_id"].astype(str).map(
        lambda order_id: selection_hex(date, order_id, seed)
    )
    actual = frame["selection_hash"].astype(str)
    mismatch = int(actual.ne(expected).sum())
    if mismatch:
        raise Stage2V52ContractError(
            f"{date} has {mismatch} selection hashes not derived from date/order/seed"
        )


def normalized_selection_rank(frame: pd.DataFrame) -> pd.Series:
    """Stable daily rank using the frozen hash then order-id tie-break."""

    if frame.empty:
        return pd.Series(dtype=float, index=frame.index)
    order = frame.sort_values(
        ["selection_hash", "order_id"], kind="stable"
    ).index
    values = pd.Series(np.empty(len(frame), dtype=float), index=frame.index)
    values.loc[order] = (np.arange(len(frame), dtype=float) + 0.5) / len(frame)
    return values


def local_time_bin(
    epoch_seconds: pd.Series, *, timezone: str = "Asia/Shanghai"
) -> pd.Series:
    local = pd.to_datetime(
        pd.to_numeric(epoch_seconds, errors="coerce"), unit="s", utc=True,
        errors="coerce",
    ).dt.tz_convert(timezone)
    result = local.dt.hour * 2 + local.dt.minute.ge(30).astype("Int64")
    return result.astype("Int64")


def positive_support_quantiles(
    counts: Mapping[Any, int] | pd.Series,
    quantiles: Sequence[float] = (0.25, 0.50, 0.75, 0.90),
) -> dict[str, int]:
    values = np.asarray(
        list(counts.values()) if isinstance(counts, Mapping) else counts.to_numpy(),
        dtype=float,
    )
    values = values[np.isfinite(values) & (values > 0)]
    if not len(values):
        raise Stage2V52ContractError("support fit has no positive cells")
    return {
        f"p{int(round(q * 100))}": int(np.quantile(values, q, method="nearest"))
        for q in quantiles
    }


def assign_support_group(count: int | float, *, p25: int, p75: int) -> str:
    value = int(count) if pd.notna(count) else 0
    if value <= 0:
        return "unseen"
    if value <= int(p25):
        return "low"
    if value <= int(p75):
        return "medium"
    return "high"


def assign_support_groups(
    keys: pd.Series, counts: Mapping[Any, int], quantiles: Mapping[str, int]
) -> pd.Series:
    support = keys.map(counts).fillna(0).astype("int64")
    p25, p75 = int(quantiles["p25"]), int(quantiles["p75"])
    values = np.select(
        [support.eq(0), support.le(p25), support.le(p75)],
        ["unseen", "low", "medium"], default="high",
    )
    return pd.Series(values, index=keys.index, dtype="string")


def _finite_unit_interval(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    array = values.to_numpy(dtype=float, na_value=np.nan)
    return pd.Series(
        np.isfinite(array)
        & values.between(0.0, 1.0, inclusive="both").fillna(False).to_numpy(),
        index=series.index,
        dtype=bool,
    )


def add_frozen_stage1_target_masks(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the exact frozen Stage 1 v3 component availability rules."""

    required = {
        "direct_interval_count", "direct_observed_time_s",
        "crawl_time_share", "stop_time_share", "speed_cv_bounded",
        "acceleration_rms_bounded", "acceleration_pair_count",
        "acceleration_weight_s",
    }
    missing = required - set(frame.columns)
    if missing:
        raise Stage2V52ContractError(
            f"traversal labels lack frozen mask inputs: {sorted(missing)}"
        )
    result = frame.copy()
    direct_count = pd.to_numeric(
        result["direct_interval_count"], errors="coerce"
    ).fillna(0)
    direct_time = pd.to_numeric(
        result["direct_observed_time_s"], errors="coerce"
    ).fillna(0)
    has_direct = direct_count.gt(0) & direct_time.gt(0)
    result["crawl_target_valid"] = (
        _finite_unit_interval(result["crawl_time_share"]) & has_direct
    )
    result["stop_target_valid"] = (
        _finite_unit_interval(result["stop_time_share"]) & has_direct
    )
    result["speed_cv_target_valid"] = (
        _finite_unit_interval(result["speed_cv_bounded"]) & direct_count.ge(2)
    )
    result["acceleration_rms_target_valid"] = (
        _finite_unit_interval(result["acceleration_rms_bounded"])
        & pd.to_numeric(
            result["acceleration_pair_count"], errors="coerce"
        ).fillna(0).gt(0)
        & pd.to_numeric(
            result["acceleration_weight_s"], errors="coerce"
        ).fillna(0).gt(0)
    )
    return result


def split_rejection_tokens(reason: Any) -> tuple[str, ...]:
    if reason is None or (isinstance(reason, float) and np.isnan(reason)):
        raise Stage2V52ContractError("rejection reason is missing")
    tokens = tuple(token.strip() for token in str(reason).split("|") if token.strip())
    if not tokens:
        raise Stage2V52ContractError("rejection reason is empty")
    return tokens


def rejection_mechanisms(reason: Any) -> tuple[str, ...]:
    mechanisms: list[str] = []
    unknown: list[str] = []
    for token in split_rejection_tokens(reason):
        if token.startswith("MATCH_REJECTION:"):
            group = "ROUTE/MAP-MATCH"
        elif token.startswith("PROCESSING_EXCEPTION:"):
            group = "ROUTE/MAP-MATCH"
        else:
            group = REJECTION_TOKEN_GROUPS.get(token)
        if group is None:
            unknown.append(token)
        elif group not in mechanisms:
            mechanisms.append(group)
    if unknown:
        raise Stage2V52ContractError(
            f"unmapped rejection tokens: {sorted(unknown)}"
        )
    return tuple(mechanisms)


def probability_distribution(counts: Mapping[Any, int], keys: Sequence[Any]) -> np.ndarray:
    values = np.asarray([float(counts.get(key, 0)) for key in keys], dtype=float)
    total = float(values.sum())
    if total <= 0:
        raise Stage2V52ContractError("distribution has zero total")
    return values / total


def total_variation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape:
        raise Stage2V52ContractError("distribution shapes disagree")
    return float(0.5 * np.abs(left - right).sum())


def jensen_shannon_divergence(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape:
        raise Stage2V52ContractError("distribution shapes disagree")
    midpoint = 0.5 * (left + right)

    def kl(values: np.ndarray, reference: np.ndarray) -> float:
        mask = values > 0
        return float(np.sum(values[mask] * np.log2(values[mask] / reference[mask])))

    return 0.5 * kl(left, midpoint) + 0.5 * kl(right, midpoint)


@dataclass(frozen=True)
class RateEffect:
    rare_count: int
    rare_denominator: int
    common_count: int
    common_denominator: int
    rare_rate: float
    common_rate: float
    percentage_point_difference: float
    rate_ratio: float | None
    difference_ci_low: float
    difference_ci_high: float
    ratio_ci_low: float | None
    ratio_ci_high: float | None


def material_negative_rate_effect(
    effect: RateEffect | Mapping[str, Any], *, maximum_rate_ratio: float,
    minimum_absolute_gap: float,
) -> bool:
    value = effect.__dict__ if isinstance(effect, RateEffect) else effect
    ratio = value.get("rate_ratio")
    difference = float(value["percentage_point_difference"])
    return bool(
        ratio is not None
        and (
            float(ratio) < float(maximum_rate_ratio)
            or difference < -float(minimum_absolute_gap)
        )
        and float(value["difference_ci_high"]) < 0
    )


def aggregate_rank_decision(
    frame: pd.DataFrame, *, minimum_count: int = 30
) -> tuple[pd.DataFrame, float]:
    required = {"dimension", "stratum", "count", "mean_rank"}
    missing = required - set(frame.columns)
    if missing:
        raise Stage2V52ContractError(
            f"rank summary lacks columns: {sorted(missing)}"
        )
    working = frame.copy()
    working["weighted_rank"] = working["count"] * working["mean_rank"]
    aggregate = working.groupby(
        ["dimension", "stratum"], as_index=False
    ).agg(count=("count", "sum"), weighted_rank=("weighted_rank", "sum"))
    aggregate = aggregate.loc[aggregate["count"].ge(int(minimum_count))].copy()
    if aggregate.empty:
        raise Stage2V52ContractError("no rank strata meet the minimum count")
    aggregate["mean_rank"] = aggregate["weighted_rank"] / aggregate["count"]
    maximum_gap = float((aggregate["mean_rank"] - 0.5).abs().max())
    return aggregate, maximum_gap


def cluster_bootstrap_rate_effect(
    daily: pd.DataFrame,
    *,
    numerator: str,
    denominator: str,
    group_column: str = "comparison_group",
    rare_label: str = "rare",
    common_label: str = "common",
    replicates: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 20261009,
) -> RateEffect:
    required = {"date", group_column, numerator, denominator}
    missing = required - set(daily.columns)
    if missing:
        raise Stage2V52ContractError(
            f"bootstrap input lacks columns: {sorted(missing)}"
        )
    dates = np.asarray(sorted(daily["date"].astype(str).unique()), dtype=object)
    if not len(dates):
        raise Stage2V52ContractError("bootstrap input has no dates")
    lookup = daily.copy()
    lookup["date"] = lookup["date"].astype(str)

    def daily_arrays(label: str) -> tuple[np.ndarray, np.ndarray]:
        selected = lookup.loc[
            lookup[group_column].eq(label), ["date", numerator, denominator]
        ].groupby("date", as_index=True)[[numerator, denominator]].sum()
        selected = selected.reindex(dates, fill_value=0)
        return (
            selected[numerator].to_numpy(dtype=np.int64),
            selected[denominator].to_numpy(dtype=np.int64),
        )

    rare_numerator, rare_denominator = daily_arrays(rare_label)
    common_numerator, common_denominator = daily_arrays(common_label)
    rare_num, rare_den = int(rare_numerator.sum()), int(rare_denominator.sum())
    common_num, common_den = int(common_numerator.sum()), int(common_denominator.sum())
    if rare_den <= 0 or common_den <= 0:
        raise Stage2V52ContractError("bootstrap comparison has zero denominator")
    rare_rate, common_rate = rare_num / rare_den, common_num / common_den
    rng = np.random.default_rng(int(seed))
    differences: list[float] = []
    ratios: list[float] = []
    for _ in range(int(replicates)):
        indices = rng.integers(0, len(dates), size=len(dates))
        rn = int(rare_numerator[indices].sum())
        rd = int(rare_denominator[indices].sum())
        cn = int(common_numerator[indices].sum())
        cd = int(common_denominator[indices].sum())
        if rd <= 0 or cd <= 0:
            continue
        rr, cr = rn / rd, cn / cd
        differences.append(rr - cr)
        if cr > 0:
            ratios.append(rr / cr)
    if not differences:
        raise Stage2V52ContractError("all bootstrap samples were invalid")
    alpha = (1.0 - float(confidence_level)) / 2.0
    difference_ci = np.quantile(differences, [alpha, 1.0 - alpha])
    ratio_ci = (
        np.quantile(ratios, [alpha, 1.0 - alpha]) if ratios else (np.nan, np.nan)
    )
    return RateEffect(
        rare_count=rare_num,
        rare_denominator=rare_den,
        common_count=common_num,
        common_denominator=common_den,
        rare_rate=float(rare_rate),
        common_rate=float(common_rate),
        percentage_point_difference=float(rare_rate - common_rate),
        rate_ratio=float(rare_rate / common_rate) if common_rate > 0 else None,
        difference_ci_low=float(difference_ci[0]),
        difference_ci_high=float(difference_ci[1]),
        ratio_ci_low=float(ratio_ci[0]) if np.isfinite(ratio_ci[0]) else None,
        ratio_ci_high=float(ratio_ci[1]) if np.isfinite(ratio_ci[1]) else None,
    )


def classify_upstream(
    *, demand_concentrated: bool, stage0_quality_effect: bool,
    stage1_supervision_effect: bool,
) -> tuple[str, str]:
    if stage0_quality_effect and stage1_supervision_effect:
        return "UP-D", "MULTI_STAGE_SELECTION_COMPOUNDS_SPARSE_CONTEXT_ATTRITION"
    if stage0_quality_effect:
        return "UP-B", "STAGE0_QUALITY_SELECTION_REDUCES_RARE_CONTEXT_REPRESENTATION"
    if stage1_supervision_effect:
        return "UP-C", "STAGE1_SUPERVISION_ATTRITION_AMPLIFIES_TARGET_SPECIFIC_SPARSITY"
    return "UP-A", "LOW_SPARSE_SHARE_PRIMARILY_REFLECTS_RAW_RIDEHAILING_DEMAND"


def validate_funnel_identity(
    *, raw: int, processed: int, accepted: int, rejected: int,
    unprocessed_quota: int,
) -> None:
    if processed != accepted + rejected:
        raise Stage2V52ContractError(
            "processed must equal accepted plus rejected"
        )
    if raw != processed + unprocessed_quota:
        raise Stage2V52ContractError(
            "raw must equal processed plus quota-unprocessed"
        )


def assert_disjoint_identity_sets(
    raw: set[tuple[str, str]], accepted: set[tuple[str, str]],
    rejected: set[tuple[str, str]],
) -> None:
    if accepted & rejected:
        raise Stage2V52ContractError("accepted and rejected identities overlap")
    missing = (accepted | rejected) - raw
    if missing:
        raise Stage2V52ContractError(
            f"{len(missing)} processed identities are absent from raw candidates"
        )


def assert_distribution_sums_to_one(values: Sequence[float], *, atol: float = 1e-10) -> None:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all() or not np.isclose(array.sum(), 1.0, atol=atol):
        raise Stage2V52ContractError(
            f"distribution shares sum to {array.sum()}, expected 1"
        )


def snapshot_paths(paths: Sequence[str]) -> str:
    """Cheap write guard over relative path, size and mtime, not content."""

    digest = hashlib.sha256()
    for value in sorted(str(item) for item in paths):
        digest.update(value.encode("utf-8"))
    return digest.hexdigest()
