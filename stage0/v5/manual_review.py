"""Reproducible review-pack selection; never creates human labels."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import Stage0Config, stable_hash


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
    quality["selection_stratum"] = "representative"
    quality.loc[quality.parallel_ambiguity_share.gt(0), "selection_stratum"] = "parallel_or_grade_complex"
    quality.loc[quality.route_quality.eq("rejected"), "selection_stratum"] = "rejected"
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
