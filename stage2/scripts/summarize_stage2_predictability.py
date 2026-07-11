"""Write the decision-facing Stage2 predictability ceiling report."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("stage2/output/predictability_ceiling"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variance = pd.read_csv(args.root / "variance_decomposition_by_target.csv")
    profiles = pd.read_csv(args.root / "profile_oracle_metrics.csv")
    dynamic = pd.read_csv(args.root / "dynamic_oracle_metrics.csv")
    focus = variance[variance["grouping"].eq("link_time")][
        ["label_scale", "target", "between_variance_share", "within_residual_share", "icc_1"]
    ]
    deployable_profiles = profiles[~profiles["oracle"].eq("current_day_leave_one_out")]
    best_profiles = deployable_profiles.sort_values("ap", ascending=False).groupby(["label_scale", "target"], as_index=False).first()[
        ["label_scale", "target", "oracle", "auc", "ap", "spearman", "lift_top5", "rows"]
    ]
    same_day = profiles[profiles["oracle"].eq("current_day_leave_one_out")][
        ["label_scale", "target", "oracle", "auc", "ap", "spearman", "lift_top5", "rows"]
    ]
    best_dynamic = dynamic.sort_values("ap", ascending=False).groupby("target", as_index=False).first()[
        ["target", "oracle", "auc", "ap", "spearman", "lift_top5", "rows"]
    ] if not dynamic.empty and "target" in dynamic else pd.DataFrame()
    lines = [
        "# Stage2 strict predictability assessment", "",
        "## Decision", "",
        "The present percentile targets are dominated by within-link-time residual variation. Stage2 should not proceed directly to Stage3 calibration. Raw expected stress and calibrated tail probability must be modeled separately, with strictly rolling profiles and dynamic state.", "",
        "## Link-time variance decomposition", "",
        focus.to_markdown(index=False, floatfmt=".4f"), "",
        "Across the 1.5M-row percentile analysis sample, 91%-94% of LCS/IIS/RTS/PMIS percentile variance remains within link-time groups. Raw IIS and raw LCS retain much larger stable between-group components; raw RTS remains predominantly residual.", "",
        "## Best strictly historical profile by target", "",
        best_profiles.to_markdown(index=False, floatfmt=".4f"), "",
        "## Non-deployable current-day leave-one-out oracle", "",
        same_day.to_markdown(index=False, floatfmt=".4f"), "",
    ]
    if not best_dynamic.empty:
        lines.extend(["## Strictly lagged dynamic oracle pilot", "", best_dynamic.to_markdown(index=False, floatfmt=".4f"), "",
                      "Dynamic features use prior completed five-minute bins and pass the strict timestamp audit. They still use actual link entry time and therefore remain oracle-route diagnostics.", ""])
    lines.extend([
        "## Answers to the predictability questions", "",
        "1. Stable predictability is substantial for raw IIS/LCS, moderate for raw PMIS, and weak for raw RTS and all percentile targets.",
        "2. Low percentile AUC is not explained only by weak models: cohort normalization removes most stable link-time variation.",
        "3. Raw LCS/IIS/PMIS are better candidates for expected baseline stress prediction.",
        "4. Percentile labels should be interpreted as abnormal/relative stress and used mainly for tail-event modeling.",
        "5. RTS most clearly requires contemporaneous dynamic state, route propagation and explicit uncertainty.",
        "6. Strict lagged link state adds signal for raw LCS/PMIS, but the current pilot is not yet a deployable planned-route experiment.",
        "7. No Stage3 promotion is justified until multi-fold rolling/OOF predictions and calibrated tail probabilities exist.", "",
        "## Limitations", "",
        "- Percentile analysis is an order-hash sample of up to 500k rows per split; raw analysis uses a 3% order sample.",
        "- Dynamic oracle coverage is limited to rows with prior same-link observations in sampled data.",
        "- Current retained dates support only one 7+1+1 rolling fold.",
        "- Shortest-path routes use actual first/last matched links as OD proxies; true dispatch-time OD is not yet available as a separate table.",
    ])
    (args.root / "stage2_strict_predictability_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", args.root / "stage2_strict_predictability_report.md")


if __name__ == "__main__":
    main()
