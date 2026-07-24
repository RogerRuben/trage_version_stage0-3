"""Validation-calibrated release-time uncertainty for preassignment.

The residual convention is deliberately explicit::

    residual_sec = predicted_service_time_sec - realized_service_time_sec
    safe_release = expected_release - Q(residual_sec)

This follows the Simulator v3 preassignment contract.  The resolver uses a
validation-only hierarchy (time-zone-stress, time-zone, time, global) and
never substitutes a fixed number of seconds when a narrow cell is sparse.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


RESIDUAL_DEFINITION = "predicted_service_time_sec_minus_realized_service_time_sec"
LEVEL_COLUMNS = {
    "time-zone-stress": ["time_bin", "zone", "stress_bucket"],
    "time-zone": ["time_bin", "zone"],
    "time": ["time_bin"],
    "global": [],
}


@dataclass(frozen=True)
class SafeReleaseResolution:
    expected_release_time: pd.Timestamp
    safe_release_time: pd.Timestamp
    residual_quantile_sec: float
    buffer_source: str
    buffer_sample_count: int
    buffer_quantile: float
    residual_definition: str = RESIDUAL_DEFINITION

    def to_metadata(self) -> dict[str, Any]:
        return {
            "expected_release_time": str(self.expected_release_time),
            "safe_release_time": str(self.safe_release_time),
            "release_residual_quantile_sec": self.residual_quantile_sec,
            "release_buffer_sec": float(
                (self.safe_release_time - self.expected_release_time).total_seconds()
            ),
            "buffer_source": self.buffer_source,
            "buffer_sample_count": self.buffer_sample_count,
            "buffer_quantile": self.buffer_quantile,
            "residual_definition": self.residual_definition,
        }


class SafeReleaseBufferResolver:
    """Resolve Q0.9 service residuals from an auditable validation table."""

    REQUIRED_TABLE_COLUMNS = {
        "level",
        "time_bin",
        "zone",
        "stress_bucket",
        "residual_quantile_sec",
        "sample_count",
        "quantile",
        "validation_date",
        "residual_definition",
        "source_dataset",
    }

    def __init__(self, table: pd.DataFrame, minimum_samples: int = 30):
        missing = self.REQUIRED_TABLE_COLUMNS - set(table.columns)
        if missing:
            raise ValueError(f"Release residual table is missing columns: {sorted(missing)}")
        if table.empty:
            raise ValueError("Release residual table is empty")
        definitions = set(table["residual_definition"].dropna().astype(str))
        if definitions != {RESIDUAL_DEFINITION}:
            raise ValueError(f"Unsupported residual definition(s): {sorted(definitions)}")
        quantiles = set(pd.to_numeric(table["quantile"], errors="coerce").dropna().round(8))
        if len(quantiles) != 1:
            raise ValueError("Release residual table must contain exactly one quantile")
        self.table = table.copy()
        self.minimum_samples = int(minimum_samples)
        self.quantile = float(next(iter(quantiles)))
        self.validation_dates = sorted(set(self.table["validation_date"].astype(str)))
        self.source_datasets = sorted(set(self.table["source_dataset"].astype(str)))

    @classmethod
    def from_parquet(cls, path: str | Path, minimum_samples: int = 30) -> "SafeReleaseBufferResolver":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Validation residual table does not exist: {path}")
        return cls(pd.read_parquet(path), minimum_samples=minimum_samples)

    @staticmethod
    def build_table(
        validation_rows: pd.DataFrame,
        quantile: float = 0.9,
        validation_date: str = "",
        source_dataset: str = "validation_service_time_predictions",
    ) -> pd.DataFrame:
        required = {
            "time_bin",
            "zone",
            "stress_bucket",
            "predicted_service_time_sec",
            "realized_service_time_sec",
        }
        missing = required - set(validation_rows.columns)
        if missing:
            raise ValueError(f"Validation residual rows are missing columns: {sorted(missing)}")
        if not 0 < float(quantile) < 1:
            raise ValueError("quantile must be strictly between zero and one")

        work = validation_rows.copy()
        predicted = pd.to_numeric(work["predicted_service_time_sec"], errors="coerce")
        realized = pd.to_numeric(work["realized_service_time_sec"], errors="coerce")
        work["residual_sec"] = predicted - realized
        work = work.loc[
            predicted.notna()
            & realized.notna()
            & predicted.gt(0)
            & realized.gt(0)
            & work["residual_sec"].notna()
        ].copy()
        if work.empty:
            raise ValueError("No finite positive validation service-time rows remain")

        records: list[dict[str, Any]] = []
        for level, columns in LEVEL_COLUMNS.items():
            if columns:
                grouper = columns[0] if len(columns) == 1 else columns
                groups = work.groupby(grouper, dropna=False, sort=True)
                iterator = groups
            else:
                iterator = [((), work)]
            for key, group in iterator:
                keys = key if isinstance(key, tuple) else (key,)
                identity = dict(zip(columns, keys))
                records.append({
                    "level": level,
                    "time_bin": identity.get("time_bin", pd.NA),
                    "zone": identity.get("zone", ""),
                    "stress_bucket": identity.get("stress_bucket", ""),
                    "residual_quantile_sec": float(group["residual_sec"].quantile(quantile)),
                    "sample_count": int(len(group)),
                    "quantile": float(quantile),
                    "validation_date": str(validation_date),
                    "residual_definition": RESIDUAL_DEFINITION,
                    "source_dataset": str(source_dataset),
                })
        return pd.DataFrame.from_records(records)

    def resolve(
        self,
        expected_release_time: pd.Timestamp,
        time_bin: int,
        zone: str,
        stress_bucket: str,
    ) -> SafeReleaseResolution:
        expected_release_time = pd.Timestamp(expected_release_time)
        selected: pd.Series | None = None
        selected_level = ""
        for level, columns in LEVEL_COLUMNS.items():
            candidates = self.table.loc[self.table["level"].eq(level)]
            if "time_bin" in columns:
                candidates = candidates.loc[
                    pd.to_numeric(candidates["time_bin"], errors="coerce").eq(int(time_bin))
                ]
            if "zone" in columns:
                candidates = candidates.loc[candidates["zone"].astype(str).eq(str(zone))]
            if "stress_bucket" in columns:
                candidates = candidates.loc[
                    candidates["stress_bucket"].astype(str).eq(str(stress_bucket))
                ]
            if level != "global":
                candidates = candidates.loc[
                    pd.to_numeric(candidates["sample_count"], errors="coerce").ge(self.minimum_samples)
                ]
            if not candidates.empty:
                selected = candidates.sort_values("sample_count", ascending=False).iloc[0]
                selected_level = level
                break
        if selected is None:
            raise ValueError("Validation residual table has no usable global fallback")

        residual_q = float(selected["residual_quantile_sec"])
        safe_release = expected_release_time - pd.Timedelta(seconds=residual_q)
        return SafeReleaseResolution(
            expected_release_time=expected_release_time,
            safe_release_time=safe_release,
            residual_quantile_sec=residual_q,
            buffer_source=f"validation_q{self.quantile:.2f}:{selected_level}",
            buffer_sample_count=int(selected["sample_count"]),
            buffer_quantile=self.quantile,
        )

    def audit(self) -> dict[str, Any]:
        levels = self.table.groupby("level").size().to_dict()
        global_rows = self.table.loc[self.table["level"].eq("global")]
        return {
            "status": "PASS" if len(global_rows) == 1 else "FAIL",
            "quantile": self.quantile,
            "minimum_samples": self.minimum_samples,
            "validation_dates": self.validation_dates,
            "source_datasets": self.source_datasets,
            "residual_definition": RESIDUAL_DEFINITION,
            "level_row_counts": {str(k): int(v) for k, v in levels.items()},
            "global_fallback_rows": int(len(global_rows)),
        }
