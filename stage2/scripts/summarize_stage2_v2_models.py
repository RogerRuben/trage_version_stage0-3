"""Create a compact Stage2 Deep Modeling v2 report."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2-output", type=Path, default=Path("stage2/output"))
    return parser.parse_args()


def read_metrics(root: Path) -> pd.DataFrame:
    specs = [
        ("full_tabular_lgbm_3m_tail", root / "deep_baselines" / "full_tabular" / "full_tabular_metrics_by_target.csv"),
        ("route_local_transformer_v2", root / "deep_baselines_v2" / "route_local_transformer" / "route_local_transformer_metrics_by_target.csv"),
        ("dual_graph_route_transformer_v2", root / "deep_baselines_v2" / "dual_graph_route_transformer" / "dual_graph_route_transformer_metrics_by_target.csv"),
    ]
    frames = []
    for model, path in specs:
        if path.exists():
            frame = pd.read_csv(path)
            frame["model"] = model
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    args = parse_args()
    root = args.stage2_output
    metrics = read_metrics(root)
    slice_path = root / "deep_baselines_v2" / "v2_slice_metrics.csv"
    order_path = root / "deep_baselines_v2" / "v2_order_tail_separation.csv"
    fair_path = root / "deep_baselines_v2" / "fair_30k_same_pred_rows_comparison.csv"
    slices = pd.read_csv(slice_path) if slice_path.exists() else pd.DataFrame()
    orders = pd.read_csv(order_path) if order_path.exists() else pd.DataFrame()
    fair = pd.read_csv(fair_path) if fair_path.exists() else pd.DataFrame()
    report = root / "deep_model_v2_report.md"

    lines = [
        "# Stage2 Deep Modeling v2 report",
        "",
        "## Scope and comparability",
        "",
        "This report is a structural probe summary, not a final model ranking. The v2 deep models were trained with a small budget "
        "(about 30k train orders, 15k eval orders, two supervised epochs, hidden dimension 96), while `full_tabular_lgbm_3m_tail` "
        "uses a much larger training budget and stronger engineered / historical profile features.",
        "",
        "Therefore the tables below should be read as diagnostic evidence only. They can support the statement that the current "
        "small-scale deep probe has not yet shown a stable advantage, but they cannot support the stronger conclusion that route/graph "
        "deep models are inferior to the full tabular model.",
        "",
        "## Models implemented",
        "",
        "- `RouteLocalTransformer`: local route convolution, local-window Transformer attention, route-level auxiliary high-stress head, and contrastive route pretraining.",
        "- `DualGraphRouteTransformer`: physical consecutive-link graph, route co-occurrence / stress-propagation graph, local-window Transformer, and intersection-gated IIS channel.",
        "- `full_tabular_lgbm_3m_tail` remains the strong tabular reference with train-only historical profiles and tail weighting.",
        "",
        "All models respect the Stage2 data contract: IIS missing labels are masked, and post-trip realized trajectory primitives are excluded from model inputs.",
        "",
        "## Test all-link comparison",
        "",
        "| model | target | AUC | Spearman | top10 lift | top5 lift |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if not metrics.empty:
        test = metrics[metrics.split.eq("test")]
        for row in test.sort_values(["target", "auc"], ascending=[True, False]).itertuples(index=False):
            lines.append(f"| {row.model} | {row.target} | {row.auc:.3f} | {row.spearman:.3f} | {row.top10_lift:.2f} | {row.top5_lift:.2f} |")

    if not fair.empty:
        fair_test = fair[fair.split.eq("test")]
        lines += [
            "",
            "## Fair 30k comparison on common prediction rows",
            "",
            "This comparison restricts evaluation to rows where both v2 deep probes produced non-null predictions. It is therefore a stricter fair-probe comparison than the all-link table above.",
            "",
            "| model | target | rows | AUC | Spearman | top10 lift | top5 lift |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in fair_test.sort_values(["target", "auc"], ascending=[True, False]).itertuples(index=False):
            lines.append(f"| {row.model} | {row.target} | {int(row.rows)} | {row.auc:.3f} | {row.spearman:.3f} | {row.top10_lift:.2f} | {row.top5_lift:.2f} |")

    if not slices.empty:
        key = slices[(slices.split == "test") & (slices.slice.isin(["rare_links", "high_endpoint_degree", "peak", "long_route"]))]
        lines += [
            "",
            "## Key layered test slices",
            "",
            "| model | target | slice | AUC | Spearman | top10 lift |",
            "|---|---:|---|---:|---:|---:|",
        ]
        for row in key.sort_values(["slice", "target", "auc"], ascending=[True, True, False]).itertuples(index=False):
            lines.append(f"| {row.model} | {row.target} | {row.slice} | {row.auc:.3f} | {row.spearman:.3f} | {row.top10_lift:.2f} |")

    if not orders.empty:
        order_test = orders[(orders.split == "test") & (orders.score == "pred_max")]
        lines += [
            "",
            "## Test order-level tail separation",
            "",
            "| model | target | base tail | lift@top10 | lift@top5 |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in order_test.sort_values(["target", "lift_at_top10"], ascending=[True, False]).itertuples(index=False):
            lines.append(f"| {row.model} | {row.target} | {row.base_tail_rate:.3f} | {row.lift_at_top10:.2f} | {row.lift_at_top5:.2f} |")

    lines += [
        "",
        "## Interpretation",
        "",
        "- The v2 architectures are more faithful to ITS/TTE/ETA-style modeling ideas than the previous plain BiGRU/GNN probes.",
        "- Under the current 30k-order / 15k-eval / 2-epoch probe, RouteLocalTransformer and DualGraphRouteTransformer have not shown a stable overall advantage. This is a limited-budget probe result, not a definitive architecture verdict.",
        "- The comparison against `full_tabular_lgbm_3m_tail` is intentionally retained as context, but it is not budget-matched: the tabular model uses more data and richer engineered inputs.",
        "- In the fair 30k common-row comparison, tabular remains stronger for LCS/IIS/PMIS, while DualGraphRouteTransformer shows the clearest RTS signal. This supports an RTS-specific route/graph branch rather than a blanket deep-model replacement.",
        "- DualGraphRouteTransformer also shows a potential RTS signal in some peak slices. This should be treated as a hypothesis that needs larger-sample, rolling-split, and bootstrap-CI confirmation.",
        "- Temporal shift is a plausible bottleneck, but it is not proven by this probe alone. It should be tested with random split vs time split vs rolling split, day-wise AUC drift, calibration drift, train-day expansion curves, and peak/off-peak stability.",
        "- RTS remains drift-sensitive and should continue to be modeled and reported separately.",
        "",
        "## Recommendation",
        "",
        "Do not replace the Stage3 input with the current v2 deep probe predictions. The next step is a controlled Stage2 v2 comparison program: "
        "fair 30k comparisons, scaling curves, hybrid fusion, RTS-specific modeling, rolling/OOF evaluation, and slice-level confidence intervals. "
        "The most promising thesis path is likely not `deep replaces tabular`, but `route/graph representation adds incremental value to a strong tabular and calibrated prediction stack`.",
        "",
        "## Next controlled experiments",
        "",
        "1. Fair small-sample comparison: train `full_tabular_lgbm_30k`, BiGRU, RouteLocalTransformer, DualGraphRouteTransformer, and plain GNN on the same order budget.",
        "2. Scaling curve: repeat at 30k, 100k, 300k, 1M, and 3M train orders/rows where feasible.",
        "3. Hybrid fusion: feed route/graph embeddings or deep stress scores into a tabular/calibrated model and test incremental value.",
        "4. RTS-specific branch: evaluate route-aware / graph-aware / peak-aware models for RTS separately from LCS/IIS/PMIS.",
        "5. Robustness: run rolling/OOF evaluation, bootstrap confidence intervals, seen-vs-rare link slices, intersection-heavy slices, and peak/off-peak slices.",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
