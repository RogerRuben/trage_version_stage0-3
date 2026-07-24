"""Reproducible review-pack selection; never creates human labels."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .archive import sampling_run_id
from .config import Stage0Config, stable_hash
from .gates import sample_order_sha256
from .manifest import write_manifest


def export_review_pack(config: Stage0Config, repo: Path, split: str, count: int) -> dict[str, Any]:
    """Export an unlabeled, probability-aware review index.

    ``development`` deliberately pools train and validation products.  Test has
    a separate blind pack and cannot be requested through the development role.
    """
    output = config.path("output", repo)
    if split == "development":
        dates = [
            str(value)
            for role in ("train", "validation")
            for value in config.section("split")[role]
        ]
    elif split == "test":
        dates = [str(value) for value in config.section("split")["test"]]
    else:
        raise ValueError("review split must be 'development' or 'test'")
    files = [path for date in dates for path in (output / "route_quality" / f"day={date}").glob("*.parquet")]
    if not files:
        raise RuntimeError(f"no route-quality products for {split}")
    quality = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    quality["selection_stratum"] = "random_representative"
    quality.loc[quality.route_quality.ne("rejected"), "selection_stratum"] = "ordinary_road"
    quality.loc[quality.parallel_ambiguity_share.gt(0), "selection_stratum"] = "parallel_highway_ground"
    quality.loc[quality.get("suspicious_level_transition_count", pd.Series(0, index=quality.index)).gt(0), "selection_stratum"] = "ramp_interchange"
    quality.loc[quality.get("fallback_share", pd.Series(0, index=quality.index)).gt(0), "selection_stratum"] = "fallback"
    quality.loc[quality.get("topology_gap_count", pd.Series(0, index=quality.index)).gt(0), "selection_stratum"] = "topology_gap"
    strata = sorted(quality.selection_stratum.unique())
    quota = max(1, count // len(strata))
    chosen: list[pd.DataFrame] = []
    for stratum, frame in quality.groupby("selection_stratum"):
        frame = frame.copy()
        frame["selection_hash"] = frame.apply(lambda row: stable_hash(split, row.date, row.order_id, seed=20261009), axis=1)
        selected = frame.sort_values("selection_hash").head(quota)
        selected["stratum_population"] = len(frame)
        selected["stratum_selected"] = len(selected)
        selected["selection_probability"] = len(selected) / len(frame)
        chosen.append(selected)
    index = pd.concat(chosen).sort_values("selection_hash").head(count).copy()
    for column in ("review_label", "major_error_reason", "reviewer_notes"):
        index[column] = ""
    name = "blind_test_review_pack" if split == "test" else "development_review_pack"
    target = output / "manual_review" / name
    target.mkdir(parents=True, exist_ok=True)
    index.to_csv(target / "review_index.csv", index=False, encoding="utf-8-sig")
    (target / "review_instructions.md").write_text(
        "# Stage 0 v5 manual route review\n\nReview GPS and route geometry independently. "
        "Do not infer correctness from matching mode. Fill only the empty human-label columns. "
        "Complex cases are oversampled; use `selection_probability` for population estimates.\n",
        encoding="utf-8",
    )
    return {"split": split, "requested": count, "exported": len(index), "path": str(target)}


def audit_development_review(config: Stage0Config, repo: Path, review_csv: Path) -> dict[str, Any]:
    """Validate completed human labels and create the evidence consumed by Gate 1."""
    frame = pd.read_csv(review_csv, dtype={"date": str, "order_id": str})
    required = {"date", "order_id", "selection_stratum", "route_quality", "review_label"}
    errors = [f"missing_column:{name}" for name in sorted(required - set(frame.columns))]
    allowed = {"correct", "minor_error", "major_error", "data_limitation"}
    completed = frame.loc[frame.review_label.astype(str).str.strip().isin(allowed)].copy() if not errors else frame.iloc[0:0]
    duplicate_count = int(completed.duplicated(["date", "order_id"]).sum())
    if duplicate_count:
        errors.append(f"duplicate_completed_reviews:{duplicate_count}")
    completed = completed.drop_duplicates(["date", "order_id"], keep=False)
    strict = completed.loc[completed.route_quality.eq("strict_core")]
    strict_precision = (
        float(strict.review_label.isin({"correct", "minor_error"}).mean())
        if len(strict) else None
    )
    dates = ["20161010", "20161014", "20161016"]
    run_id = sampling_run_id(dates, 2000, int(config.section("sampling")["seed"]))
    sampling_path = (
        config.path("output", repo) / "manifests" / "sampling_runs"
        / run_id / "sampling_manifest.parquet"
    )
    sample = pd.read_parquet(sampling_path, columns=["date", "order_id"])
    payload = {
        "status": "PASS" if not errors else "FAIL",
        "schema_errors": errors,
        "completed_unique_reviews": int(len(completed)),
        "strict_core_completed_reviews": int(len(strict)),
        "strict_core_precision": strict_precision,
        "major_error_count": int(completed.review_label.eq("major_error").sum()),
        "stratum_completed_counts": {
            str(key): int(value)
            for key, value in completed.selection_stratum.value_counts().to_dict().items()
        },
        "stratum_population_counts": {
            str(key): int(value)
            for key, value in frame.selection_stratum.value_counts().to_dict().items()
        },
        "sampling_run_id": run_id,
        "sample_order_sha256": sample_order_sha256(sample),
        "config_hash": config.digest,
        "source_review_csv": str(review_csv.resolve()),
    }
    write_manifest(
        config.path("output", repo) / "manual_review" / "development_review_audit.json",
        payload,
    )
    return payload
