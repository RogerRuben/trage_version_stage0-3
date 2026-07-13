"""Internal validity analysis for the Stage3 condition vector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DIMENSIONS = {
    "LCS": ("lcs_tail_probability", "order_lcs_raw", "order_lcs_tail"),
    "PMIS": ("pmis_tail_probability", "order_pmis_raw", "order_pmis_tail"),
    "RTS": ("rts_tail_probability", "order_rts_raw", "order_rts_tail"),
    "IIS": ("intersection_tail_probability", "order_iis_severity_q90", "order_iis_tail"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage4-input-root", type=Path, default=Path("stage3/output/stage4_inputs_core_v2"))
    parser.add_argument("--target-root", type=Path, default=Path("stage3/output/rolling_order_targets"))
    parser.add_argument("--output-root", type=Path, default=Path("stage3/output/condition_vector_validity"))
    return parser.parse_args()


def decile_rows(data: pd.DataFrame, fold: int, dim: str, pred_col: str, raw_col: str, tail_col: str) -> list[dict]:
    valid = data[pred_col].notna() & data[raw_col].notna()
    part = data.loc[valid].copy()
    if part.empty:
        return []
    part["prediction_decile"] = pd.qcut(part[pred_col], 10, labels=False, duplicates="drop") + 1
    rows = []
    for decile, group in part.groupby("prediction_decile"):
        rows.append({
            "fold": fold,
            "dimension": dim,
            "decile": int(decile),
            "rows": len(group),
            "mean_prediction": float(group[pred_col].mean()),
            "mean_realized_raw": float(group[raw_col].mean()),
            "true_tail_rate": float(group[tail_col].mean()),
        })
    return rows


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    deciles = []
    corr_rows = []
    top_rows = []
    for path in sorted(args.stage4_input_root.glob("fold=*/stage4_inputs.parquet")):
        fold = int(path.parent.name.split("=", 1)[-1])
        inputs = pd.read_parquet(path)
        targets = pd.read_parquet(args.target_root / f"fold={fold}" / "split=test" / "order_targets.parquet")
        data = inputs.merge(targets, on="order_id", suffixes=("", "_target"), validate="one_to_one")
        for dim, (pred_col, raw_col, tail_col) in DIMENSIONS.items():
            if pred_col not in data or raw_col not in data:
                continue
            deciles.extend(decile_rows(data, fold, dim, pred_col, raw_col, tail_col))
            valid = data[pred_col].notna() & data[raw_col].notna()
            corr_rows.append({
                "fold": fold,
                "dimension": dim,
                "spearman_pred_realized": float(data.loc[valid, pred_col].corr(data.loc[valid, raw_col], method="spearman")),
                "pearson_pred_realized": float(data.loc[valid, pred_col].corr(data.loc[valid, raw_col], method="pearson")),
                "top10_lift": float(data.loc[data[pred_col].nlargest(max(1, int(valid.sum() * 0.10))).index, tail_col].mean() / max(data.loc[valid, tail_col].mean(), 1e-9)),
            })
        core_cols = ["lcs_tail_probability", "pmis_tail_probability", "rts_tail_probability"]
        corr = data[core_cols].corr(method="spearman")
        for left in core_cols:
            for right in core_cols:
                if left < right:
                    top_rows.append({"fold": fold, "left": left, "right": right, "spearman": float(corr.loc[left, right])})
    decile_frame = pd.DataFrame(deciles)
    corr_frame = pd.DataFrame(corr_rows)
    redundancy = pd.DataFrame(top_rows)
    decile_frame.to_csv(args.output_root / "condition_vector_decile_validity.csv", index=False)
    corr_frame.to_csv(args.output_root / "condition_vector_correlations.csv", index=False)
    redundancy.to_csv(args.output_root / "condition_vector_dimension_redundancy.csv", index=False)
    summary = corr_frame.groupby("dimension", as_index=False)[["spearman_pred_realized", "pearson_pred_realized", "top10_lift"]].mean()
    report = [
        "# Stage3 condition vector validity report",
        "",
        "This report validates internal consistency against realized Stage1 measurements. It does not claim AV safety outcomes.",
        "",
        "## Dimension summary",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Core dimension redundancy",
        "",
        redundancy.groupby(["left", "right"], as_index=False)["spearman"].mean().to_markdown(index=False, floatfmt=".4f"),
    ]
    (args.output_root / "stage3_condition_vector_validity_report.md").write_text("\n".join(report), encoding="utf-8")
    (args.output_root / "manifest.json").write_text(json.dumps({"status": "PASS"}, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
