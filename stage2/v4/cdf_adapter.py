"""Apply the frozen Stage 1 Train CDF to Stage 2 raw predictions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stage1.v3.config import load_config as load_stage1_config
from stage1.v3.models import load_model_bundle
from stage1.v3.references import COHORT_LEVELS, add_cohort_keys

from .config import Stage2V4Config
from .contracts import Stage2V4ContractError


def _normalized_text(values: pd.Series, missing: str) -> pd.Series:
    """Return real Python strings, including for Arrow-backed missing values."""

    return values.astype("string").fillna(missing).astype(str)


def _peak_offpeak(
    estimated_entry_time: pd.Series,
    timezone: str,
) -> np.ndarray:
    local = pd.to_datetime(
        estimated_entry_time,
        unit="s",
        utc=True,
    ).dt.tz_convert(timezone)
    minute = local.dt.hour.to_numpy(dtype=int) * 60 + local.dt.minute.to_numpy(dtype=int)
    peak = ((minute >= 7 * 60) & (minute < 9 * 60 + 30)) | (
        (minute >= 17 * 60) & (minute < 19 * 60 + 30)
    )
    return np.where(peak, "peak", "offpeak")


def _choose_cdf_indexed(
    model: object,
    frame: pd.DataFrame,
    value_column: str,
    *,
    minimum_support: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stage 1 ``choose_cdf`` semantics without its quadratic key scans.

    The frozen Stage 1 implementation finds every cohort with ``keys == key``.
    That is exact but becomes O(rows * unique_edges) on the 2.1M-token Test
    day. Factorizing the unresolved cohort keys once gives identical groups in
    O(rows), while all support, fallback and sparse-CDF rules remain frozen.
    """

    keyed = add_cohort_keys(frame)
    raw = pd.to_numeric(keyed[value_column], errors="coerce").to_numpy(
        dtype=np.float64
    )
    values = np.full(len(keyed), np.nan, dtype=np.float64)
    levels = np.full(len(keyed), "unresolved", dtype=object)
    supports = np.zeros(len(keyed), dtype=np.int64)
    unresolved = np.isfinite(raw)
    counts_by_level = getattr(model, "counts")
    sparse_cdf = getattr(model, "_sparse_cdf")
    for level in COHORT_LEVELS:
        if not unresolved.any():
            break
        positions = np.flatnonzero(unresolved)
        keys = keyed.iloc[positions][level].astype(str).to_numpy()
        codes, unique_keys = pd.factorize(keys, sort=False)
        order = np.argsort(codes, kind="stable")
        group_sizes = np.bincount(codes, minlength=len(unique_keys))
        offset = 0
        level_counts = counts_by_level.get(level, {})
        required_support = 1 if level == "global" else minimum_support
        for code, key_value in enumerate(unique_keys):
            size = int(group_sizes[code])
            local_positions = order[offset : offset + size]
            offset += size
            key = str(key_value)
            bins = level_counts.get(key, {})
            support = int(sum(bins.values()))
            if support < required_support:
                continue
            key_positions = positions[local_positions]
            values[key_positions] = sparse_cdf(
                level,
                key,
                raw[key_positions],
            )
            levels[key_positions] = level
            supports[key_positions] = support
            unresolved[key_positions] = False
    return values, levels, supports


def apply_frozen_stage1_cdf(
    frame: pd.DataFrame,
    config: Stage2V4Config,
    *,
    stage1_model_root: str | Path = "stage1/models/stage1_v3_final",
    stage1_config_path: str | Path = "stage1/config/stage1_label_schema_v3.json",
) -> pd.DataFrame:
    required = {
        "observed_directed_edge_uid",
        "canonical_highway",
        "estimated_time_bin",
        "estimated_weekday_type",
        "estimated_entry_time",
        "pred_lcs_raw",
        "pred_rts_raw",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise Stage2V4ContractError(f"CDF adapter is missing fields: {missing}")
    stage1_config = load_stage1_config(stage1_config_path)
    models = load_model_bundle(stage1_model_root, stage1_config)
    expected_model_id = config.section("stage1_release")["model_id"]
    if models.model_id != expected_model_id:
        raise Stage2V4ContractError("frozen Stage 1 CDF model ID mismatch")
    working = pd.DataFrame(
        {
            "observed_directed_edge_uid": frame[
                "observed_directed_edge_uid"
            ].pipe(_normalized_text, "unmapped"),
            "canonical_highway": frame["canonical_highway"].pipe(
                _normalized_text,
                "unknown",
            ),
            "time_bin_30m": pd.to_numeric(
                frame["estimated_time_bin"],
                errors="raise",
            ).astype(int),
            "weekday_type": frame["estimated_weekday_type"].pipe(
                _normalized_text,
                "unknown",
            ),
            "peak_offpeak": _peak_offpeak(
                frame["estimated_entry_time"],
                str(config.section("causality")["timezone"]),
            ),
            "lcs_raw": pd.to_numeric(frame["pred_lcs_raw"], errors="coerce"),
            "rts_raw": pd.to_numeric(frame["pred_rts_raw"], errors="coerce"),
        }
    )
    for dimension in ("lcs", "rts"):
        available = working[f"{dimension}_raw"].notna()
        working[f"{dimension}_available"] = available
        working[f"{dimension}_unavailable_reason"] = np.where(
            available,
            "",
            "PREDICTION_UNAVAILABLE",
        )
    result = frame.copy()
    for dimension in ("lcs", "rts"):
        model = models.lcs if dimension == "lcs" else models.rts
        percentile, level, support = _choose_cdf_indexed(
            model,
            working,
            f"{dimension}_raw",
            minimum_support=int(
                stage1_config.section("normalization")[
                    "minimum_cohort_support"
                ]
            ),
        )
        source_available = working[f"{dimension}_available"].eq(True).to_numpy(
            dtype=bool
        )
        available = source_available & np.isfinite(percentile)
        percentile[~available] = np.nan
        level[~available] = "unresolved"
        support[~available] = 0
        result[f"pred_{dimension}_pct"] = percentile
        result[f"pred_{dimension}_cdf_level_used"] = level
        result[f"pred_{dimension}_cdf_sample_size"] = support
    result["cdf_model_id"] = models.model_id
    return result
