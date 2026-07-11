"""Metric-rich evaluation for Stage2 deep/tabular predictions.

The evaluator intentionally goes beyond AUC. It reports ranking, tail
detection, continuous ranking, decile behavior, order-level decision metrics,
slice robustness, block-bootstrap confidence intervals, and simple validation
selected score fusion.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, roc_auc_score


TARGETS = ["lcs", "iis", "rts", "pmis"]
ID_COLUMNS = ["order_id", "driver_id", "date", "link_id", "link_seq"]
CONTEXT_COLUMNS = [
    "order_id", "driver_id", "date", "link_id", "link_seq",
    "peak_offpeak", "route_link_count", "endpoint_degree", "time_bin", "hour",
]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    root: Path
    validation_file: str
    test_file: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("stage2/output/link_dataset"))
    parser.add_argument("--full-prediction-root", type=Path, default=Path("stage2/output/deep_predictions"))
    parser.add_argument("--tabular30-prediction-root", type=Path, default=Path("stage2/output/deep_predictions_30k"))
    parser.add_argument("--v2-prediction-root", type=Path, default=Path("stage2/output/deep_predictions_v2"))
    parser.add_argument("--v2-baseline-root", type=Path, default=Path("stage2/output/deep_baselines_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/deep_eval_metrics"))
    parser.add_argument("--report-path", type=Path, default=Path("stage2/output/deep_model_v2_report.md"))
    parser.add_argument("--bootstrap-rounds", type=int, default=200)
    parser.add_argument("--bootstrap-max-rows", type=int, default=200_000)
    parser.add_argument("--kendall-max-rows", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def available_columns(path: Path) -> set[str]:
    return set(pq.ParquetFile(path).schema_arrow.names)


def read_context(dataset_root: Path, split: str) -> pd.DataFrame:
    path = dataset_root / f"{split}.parquet"
    cols = [c for c in CONTEXT_COLUMNS if c in available_columns(path)]
    context = pd.read_parquet(path, columns=cols)
    return context.drop_duplicates(subset=[c for c in ID_COLUMNS if c in context.columns])


def read_link_counts(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame({"link_id": [], "train_link_count": []})


def model_specs(args: argparse.Namespace) -> list[ModelSpec]:
    specs = [
        ModelSpec("full_tabular_3m", args.full_prediction_root, "full_tabular_validation.parquet", "full_tabular_test.parquet"),
        ModelSpec("full_tabular_30k", args.tabular30_prediction_root, "full_tabular_validation.parquet", "full_tabular_test.parquet"),
        ModelSpec("sequence_bigru", args.full_prediction_root, "sequence_validation.parquet", "sequence_test.parquet"),
        ModelSpec("gnn_sequence", args.full_prediction_root, "gnn_sequence_validation.parquet", "gnn_sequence_test.parquet"),
        ModelSpec("route_local_transformer", args.v2_prediction_root, "route_local_transformer_validation.parquet", "route_local_transformer_test.parquet"),
        ModelSpec("dual_graph_route_transformer", args.v2_prediction_root, "dual_graph_route_transformer_validation.parquet", "dual_graph_route_transformer_test.parquet"),
    ]
    return [s for s in specs if (s.root / s.validation_file).exists() and (s.root / s.test_file).exists()]


def read_prediction(spec: ModelSpec, split: str) -> pd.DataFrame:
    path = spec.root / (spec.validation_file if split == "validation" else spec.test_file)
    return pd.read_parquet(path)


def add_context(frame: pd.DataFrame, context: pd.DataFrame, link_counts: pd.DataFrame) -> pd.DataFrame:
    join_cols = [c for c in ID_COLUMNS if c in frame.columns and c in context.columns]
    result = frame.merge(context, on=join_cols, how="left", suffixes=("", "_ctx"))
    if "train_link_count" not in result.columns and not link_counts.empty:
        result = result.merge(link_counts, on="link_id", how="left")
    if "train_link_count" in result.columns:
        result["train_link_count"] = result["train_link_count"].fillna(0)
    return result


def valid_target_frame(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    y_col = f"target_{target}"
    p_col = f"pred_{target}"
    v_col = f"{target}_valid"
    if y_col not in frame.columns or p_col not in frame.columns:
        return frame.iloc[0:0].copy()
    valid = frame[y_col].notna() & frame[p_col].notna()
    if v_col in frame.columns:
        valid = valid & frame[v_col].fillna(False)
    return frame.loc[valid].copy()


def ndcg_at_fraction(y_high: np.ndarray, pred: np.ndarray, fraction: float) -> float:
    if len(y_high) == 0:
        return float("nan")
    k = max(1, int(len(y_high) * fraction))
    order = np.argsort(-pred)[:k]
    gains = y_high[order].astype(float)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gains * discounts))
    ideal = np.sort(y_high.astype(float))[::-1][:k]
    idcg = float(np.sum(ideal * discounts))
    return dcg / idcg if idcg > 0 else float("nan")


def ranking_metrics(frame: pd.DataFrame, target: str, kendall_max_rows: int = 100_000, seed: int = 2026) -> dict[str, float]:
    y = frame[f"target_{target}"].astype(float).to_numpy()
    pred = np.clip(frame[f"pred_{target}"].astype(float).to_numpy(), 0, 1)
    high = y >= 0.90
    if kendall_max_rows and len(frame) > kendall_max_rows:
        rng = np.random.default_rng(seed)
        kendall_idx = rng.choice(len(frame), size=kendall_max_rows, replace=False)
        kendall_y = y[kendall_idx]
        kendall_pred = pred[kendall_idx]
    else:
        kendall_y = y
        kendall_pred = pred
    result: dict[str, float] = {
        "rows": int(len(frame)),
        "high_rate": float(high.mean()) if len(high) else float("nan"),
        "pearson": float(pd.Series(y).corr(pd.Series(pred), method="pearson")) if len(frame) > 1 else float("nan"),
        "spearman": float(pd.Series(y).corr(pd.Series(pred), method="spearman")) if len(frame) > 1 else float("nan"),
        "kendall_tau": float(pd.Series(kendall_y).corr(pd.Series(kendall_pred), method="kendall")) if len(frame) > 1 else float("nan"),
        "mae": float(mean_absolute_error(y, pred)) if len(frame) else float("nan"),
        "rmse": float(mean_squared_error(y, pred, squared=False)) if len(frame) else float("nan"),
        "ndcg_top10pct": ndcg_at_fraction(high, pred, 0.10),
        "ndcg_top5pct": ndcg_at_fraction(high, pred, 0.05),
    }
    if high.any() and (~high).any():
        result["auc"] = float(roc_auc_score(high, pred))
        result["ap"] = float(average_precision_score(high, pred))
    else:
        result["auc"] = float("nan")
        result["ap"] = float("nan")
    base = result["high_rate"]
    for frac, name in [(0.10, "top10"), (0.05, "top5")]:
        k = max(1, int(len(frame) * frac))
        idx = np.argsort(-pred)[:k]
        precision = float(high[idx].mean()) if len(idx) else float("nan")
        recall = float(high[idx].sum() / max(high.sum(), 1)) if len(idx) else float("nan")
        result[f"precision_at_{name}pct"] = precision
        result[f"recall_at_{name}pct"] = recall
        result[f"lift_at_{name}pct"] = precision / base if base and base > 0 else float("nan")
    return result


def decile_stats(frame: pd.DataFrame, model: str, split: str, target: str, scope: str) -> tuple[list[dict], dict]:
    y_col = f"target_{target}"
    p_col = f"pred_{target}"
    data = frame[[y_col, p_col]].dropna().copy()
    if len(data) < 20:
        return [], {}
    ranked = data[p_col].rank(method="first")
    data["pred_decile"] = pd.qcut(ranked, 10, labels=False, duplicates="drop") + 1
    rows = []
    for decile, part in data.groupby("pred_decile"):
        high = part[y_col].ge(0.90)
        rows.append({
            "model": model, "split": split, "target": target.upper(), "scope": scope,
            "pred_decile": int(decile), "rows": int(len(part)),
            "observed_target_mean": float(part[y_col].mean()),
            "observed_high_rate": float(high.mean()),
            "prediction_mean": float(part[p_col].mean()),
        })
    table = pd.DataFrame(rows).sort_values("pred_decile")
    low = table.iloc[0]["observed_high_rate"]
    high = table.iloc[-1]["observed_high_rate"]
    summary = {
        "top_bottom_high_rate_ratio": float(high / low) if low > 0 else float("nan"),
        "decile_monotonicity_spearman_target_mean": float(table["pred_decile"].corr(table["observed_target_mean"], method="spearman")),
        "decile_monotonicity_spearman_high_rate": float(table["pred_decile"].corr(table["observed_high_rate"], method="spearman")),
    }
    return rows, summary


def slice_definitions(frame: pd.DataFrame, target: str) -> dict[str, pd.Series]:
    masks: dict[str, pd.Series] = {"all": pd.Series(True, index=frame.index)}
    if "date" in frame.columns:
        for value in sorted(frame["date"].dropna().astype(str).unique()):
            masks[f"day={value}"] = frame["date"].astype(str).eq(value)
    if "peak_offpeak" in frame.columns:
        peak = frame["peak_offpeak"].astype(str).str.lower().eq("peak")
        masks["peak"] = peak
        masks["offpeak"] = ~peak
    if "route_link_count" in frame.columns:
        route = pd.to_numeric(frame["route_link_count"], errors="coerce")
        masks["short_route"] = route.le(20)
        masks["medium_route"] = route.gt(20) & route.le(60)
        masks["long_route"] = route.gt(60)
    if "train_link_count" in frame.columns:
        freq = pd.to_numeric(frame["train_link_count"], errors="coerce").fillna(0)
        masks["rare_link"] = freq.le(5)
        masks["midfreq_link"] = freq.gt(5) & freq.le(50)
        masks["common_link"] = freq.gt(50)
    if "endpoint_degree" in frame.columns:
        deg = pd.to_numeric(frame["endpoint_degree"], errors="coerce")
        masks["endpoint_degree_low"] = deg.le(2)
        masks["endpoint_degree_mid"] = deg.eq(3)
        masks["endpoint_degree_high"] = deg.ge(4)
        masks["endpoint_degree_ge3"] = deg.ge(3)
    if "iis_valid" in frame.columns:
        masks["iis_valid_subset"] = frame["iis_valid"].fillna(False)
    y_col = f"target_{target}"
    if y_col in frame.columns:
        masks["high_stress_tail"] = frame[y_col].ge(0.90)
    return masks


def bootstrap_ci(frame: pd.DataFrame, target: str, rounds: int, max_rows: int, seed: int, kendall_max_rows: int) -> dict[str, float]:
    data = valid_target_frame(frame, target)
    if len(data) < 200:
        return {}
    if max_rows and len(data) > max_rows:
        data = data.sample(n=max_rows, random_state=seed)
    groups = [idx.to_numpy() for _, idx in pd.Series(np.arange(len(data)), index=data["order_id"]).groupby(level=0)]
    if len(groups) < 20:
        return {}
    rng = np.random.default_rng(seed)
    values = {"auc": [], "ap": [], "lift_at_top5pct": [], "lift_at_top10pct": []}
    for _ in range(rounds):
        sampled = rng.integers(0, len(groups), len(groups))
        indices = np.concatenate([groups[i] for i in sampled])
        metrics = ranking_metrics(data.iloc[indices], target, kendall_max_rows, seed)
        for key in values:
            values[key].append(metrics.get(key, float("nan")))
    result = {}
    for key, vals in values.items():
        arr = np.array(vals, dtype=float)
        result[f"{key}_ci_low"] = float(np.nanquantile(arr, 0.025))
        result[f"{key}_ci_high"] = float(np.nanquantile(arr, 0.975))
    return result


def order_metrics(frame: pd.DataFrame, model: str, split: str, target: str, scope: str) -> list[dict]:
    data = valid_target_frame(frame, target)
    if data.empty:
        return []
    y_col = f"target_{target}"
    p_col = f"pred_{target}"
    grouped = data.groupby("order_id").agg(
        true_mean=(y_col, "mean"),
        true_max=(y_col, "max"),
        true_q90=(y_col, lambda x: float(np.nanquantile(x, 0.90))),
        pred_mean=(p_col, "mean"),
        pred_max=(p_col, "max"),
        pred_q90=(p_col, lambda x: float(np.nanquantile(x, 0.90))),
        link_rows=(p_col, "size"),
    ).reset_index()
    true_tail = grouped["true_q90"].ge(0.90)
    rows = []
    for pred_score in ["pred_mean", "pred_q90", "pred_max"]:
        for frac, label in [(0.10, "top10"), (0.05, "top5")]:
            k = max(1, int(len(grouped) * frac))
            top = grouped.nlargest(k, pred_score)
            precision = float(top["true_q90"].ge(0.90).mean())
            recall = float(top["true_q90"].ge(0.90).sum() / max(true_tail.sum(), 1))
            base = float(true_tail.mean())
            stress_capture = float(top["true_q90"].sum() / max(grouped["true_q90"].sum(), 1e-12))
            rows.append({
                "model": model, "split": split, "target": target.upper(), "scope": scope,
                "pred_score": pred_score, "gate": label, "orders": int(len(grouped)),
                "true_tail_rate": base,
                "precision": precision,
                "recall": recall,
                "lift": precision / base if base > 0 else float("nan"),
                "mean_true_stress_exposure": float(top["true_mean"].mean()),
                "q90_true_stress_exposure": float(top["true_q90"].mean()),
                "max_true_stress_exposure": float(top["true_max"].mean()),
                "odd_gate_coverage": frac,
                "true_stress_capture_share": stress_capture,
                "true_stress_reduction_proxy": stress_capture - frac,
            })
    return rows


def common_support(frames: dict[str, pd.DataFrame], target: str, models: list[str]) -> pd.DataFrame:
    p_col = f"pred_{target}"
    support: pd.DataFrame | None = None
    for model in models:
        frame = frames[model]
        ids = frame.loc[frame[p_col].notna(), ID_COLUMNS].drop_duplicates()
        support = ids if support is None else support.merge(ids, on=ID_COLUMNS, how="inner")
    return support.drop_duplicates() if support is not None else pd.DataFrame(columns=ID_COLUMNS)


def apply_support(frame: pd.DataFrame, support: pd.DataFrame) -> pd.DataFrame:
    if support.empty:
        return frame.iloc[0:0].copy()
    return frame.merge(support, on=ID_COLUMNS, how="inner")


def choose_hybrid_alphas(frames_by_split: dict[str, dict[str, pd.DataFrame]], output_root: Path) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    pairs = [
        ("hybrid_30k_dualgraph", "full_tabular_30k", "dual_graph_route_transformer"),
        ("hybrid_full_dualgraph", "full_tabular_3m", "dual_graph_route_transformer"),
        ("hybrid_30k_sequence", "full_tabular_30k", "sequence_bigru"),
    ]
    rows = []
    selected: dict[str, dict[str, float]] = {}
    for hybrid_name, base_name, deep_name in pairs:
        if base_name not in frames_by_split["validation"] or deep_name not in frames_by_split["validation"]:
            continue
        selected[hybrid_name] = {}
        for target in TARGETS:
            support = common_support(frames_by_split["validation"], target, [base_name, deep_name])
            base = apply_support(frames_by_split["validation"][base_name], support)
            deep = apply_support(frames_by_split["validation"][deep_name], support)
            merged = base[ID_COLUMNS + [f"target_{target}", f"{target}_valid", f"pred_{target}"]].merge(
                deep[ID_COLUMNS + [f"pred_{target}"]], on=ID_COLUMNS, suffixes=("_base", "_deep")
            )
            valid = merged[f"{target}_valid"].fillna(False) & merged[f"target_{target}"].notna()
            merged = merged.loc[valid]
            if len(merged) < 200:
                continue
            best_alpha = 1.0
            best_ap = -math.inf
            for alpha in np.linspace(0, 1, 21):
                pred = alpha * merged[f"pred_{target}_base"].to_numpy() + (1 - alpha) * merged[f"pred_{target}_deep"].to_numpy()
                high = merged[f"target_{target}"].ge(0.90).to_numpy()
                ap = average_precision_score(high, pred) if high.any() and (~high).any() else float("nan")
                rows.append({"hybrid_model": hybrid_name, "target": target.upper(), "alpha_tabular": float(alpha), "validation_ap": float(ap), "rows": int(len(merged))})
                if np.isfinite(ap) and ap > best_ap:
                    best_ap = float(ap)
                    best_alpha = float(alpha)
            selected[hybrid_name][target] = best_alpha
    table = pd.DataFrame(rows)
    if not table.empty:
        table.to_csv(output_root / "hybrid_alpha_grid.csv", index=False)
    return table, selected


def build_hybrid_frames(frames_by_split: dict[str, dict[str, pd.DataFrame]], selected: dict[str, dict[str, float]]) -> dict[str, dict[str, pd.DataFrame]]:
    pair_lookup = {
        "hybrid_30k_dualgraph": ("full_tabular_30k", "dual_graph_route_transformer"),
        "hybrid_full_dualgraph": ("full_tabular_3m", "dual_graph_route_transformer"),
        "hybrid_30k_sequence": ("full_tabular_30k", "sequence_bigru"),
    }
    result: dict[str, dict[str, pd.DataFrame]] = {"validation": {}, "test": {}}
    for hybrid_name, target_alphas in selected.items():
        base_name, deep_name = pair_lookup[hybrid_name]
        for split in ["validation", "test"]:
            if base_name not in frames_by_split[split] or deep_name not in frames_by_split[split]:
                continue
            support_models = [base_name, deep_name]
            base_all = frames_by_split[split][base_name]
            hybrid = None
            for target, alpha in target_alphas.items():
                support = common_support(frames_by_split[split], target, support_models)
                base = apply_support(base_all, support)
                deep = apply_support(frames_by_split[split][deep_name], support)
                columns = ID_COLUMNS + [f"target_{target}", f"{target}_valid", f"pred_{target}"]
                merged = base[columns].merge(deep[ID_COLUMNS + [f"pred_{target}"]], on=ID_COLUMNS, suffixes=("_base", "_deep"))
                pred = alpha * merged[f"pred_{target}_base"] + (1 - alpha) * merged[f"pred_{target}_deep"]
                part = merged[ID_COLUMNS + [f"target_{target}", f"{target}_valid"]].copy()
                part[f"pred_{target}"] = pred.astype("float32")
                if hybrid is None:
                    hybrid = part
                else:
                    hybrid = hybrid.merge(part, on=ID_COLUMNS, how="outer")
            if hybrid is not None:
                result[split][hybrid_name] = hybrid
    return result


def write_report(output_root: Path, report_path: Path, overall: pd.DataFrame, order: pd.DataFrame, slices: pd.DataFrame, bootstrap: pd.DataFrame, alphas: pd.DataFrame) -> None:
    lines = [
        "# Stage2 Deep Modeling v2 metric-rich evaluation",
        "",
        "This report evaluates Stage2 models with ranking, tail detection, continuous ranking, decile, order-level, slice, bootstrap-CI, and validation-selected fusion metrics. AUC is treated as one diagnostic, not the decision criterion.",
        "",
        "## Test overall metrics",
        "",
        "| model | target | scope | rows | AUC | AP | Spearman | Kendall | P@Top5 | R@Top5 | Lift@Top5 | NDCG@Top5 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    test = overall[(overall["split"] == "test") & (overall["slice"] == "all")]
    for row in test.sort_values(["target", "ap"], ascending=[True, False]).itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.target} | {row.scope} | {int(row.rows)} | {row.auc:.3f} | {row.ap:.3f} | "
            f"{row.spearman:.3f} | {row.kendall_tau:.3f} | {row.precision_at_top5pct:.3f} | {row.recall_at_top5pct:.3f} | "
            f"{row.lift_at_top5pct:.2f} | {row.ndcg_top5pct:.3f} |"
        )
    if not alphas.empty:
        chosen = alphas.sort_values("validation_ap").groupby(["hybrid_model", "target"], as_index=False).tail(1)
        lines += ["", "## Hybrid alpha selected on validation", "", "| hybrid | target | alpha_tabular | validation AP | rows |", "|---|---:|---:|---:|---:|"]
        for row in chosen.sort_values(["target", "hybrid_model"]).itertuples(index=False):
            lines.append(f"| {row.hybrid_model} | {row.target} | {row.alpha_tabular:.2f} | {row.validation_ap:.3f} | {int(row.rows)} |")
    if not order.empty:
        order_test = order[(order["split"] == "test") & (order["pred_score"] == "pred_q90") & (order["gate"] == "top10")]
        lines += ["", "## Test order-level decision value (pred_q90 top10)", "", "| model | target | scope | orders | lift | stress capture | reduction proxy |", "|---|---:|---|---:|---:|---:|---:|"]
        for row in order_test.sort_values(["target", "lift"], ascending=[True, False]).itertuples(index=False):
            lines.append(f"| {row.model} | {row.target} | {row.scope} | {int(row.orders)} | {row.lift:.2f} | {row.true_stress_capture_share:.3f} | {row.true_stress_reduction_proxy:.3f} |")
    if not slices.empty:
        key = slices[(slices["split"] == "test") & (slices["slice"].isin(["peak", "rare_link", "endpoint_degree_ge3", "long_route", "iis_valid_subset"]))]
        lines += ["", "## Key test slice AP / Lift@Top5", "", "| model | target | slice | scope | rows | AP | Lift@Top5 |", "|---|---:|---|---|---:|---:|---:|"]
        for row in key.sort_values(["target", "slice", "ap"], ascending=[True, True, False]).head(120).itertuples(index=False):
            lines.append(f"| {row.model} | {row.target} | {row.slice} | {row.scope} | {int(row.rows)} | {row.ap:.3f} | {row.lift_at_top5pct:.2f} |")
    if not bootstrap.empty:
        boot_test = bootstrap[bootstrap["split"] == "test"]
        lines += ["", "## Test block-bootstrap 95% CI (order-block)", "", "| model | target | scope | AUC CI | AP CI | Lift@Top5 CI | Lift@Top10 CI |", "|---|---:|---|---:|---:|---:|---:|"]
        for row in boot_test.sort_values(["target", "model"]).itertuples(index=False):
            lines.append(
                f"| {row.model} | {row.target} | {row.scope} | [{row.auc_ci_low:.3f}, {row.auc_ci_high:.3f}] | "
                f"[{row.ap_ci_low:.3f}, {row.ap_ci_high:.3f}] | [{row.lift_at_top5pct_ci_low:.2f}, {row.lift_at_top5pct_ci_high:.2f}] | "
                f"[{row.lift_at_top10pct_ci_low:.2f}, {row.lift_at_top10pct_ci_high:.2f}] |"
            )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Overall ranking: tabular models remain the stronger general-purpose predictors for LCS/IIS/PMIS under the current data contract.",
        "- Tail detection: DualGraph should be judged by AP, Lift@TopK, NDCG, and order-level lift, not AUC alone. RTS is the main dimension where DualGraph deserves a dedicated branch.",
        "- Order-level decision value: top-order stress capture and reduction proxy are the relevant Stage3/4 bridge metrics; they should govern whether predictions are dispatch-useful.",
        "- Slice robustness: rare-link, endpoint-degree, long-route, peak, and IIS-valid slices are kept separate because an architecture can be valuable even without overall dominance.",
        "- Calibration readiness: no model should feed Stage3 as a calibrated vector until rolling / OOF predictions and calibration-drift checks are available.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    specs = model_specs(args)
    link_counts = read_link_counts(args.v2_baseline_root / "train_link_counts.parquet")
    frames_by_split: dict[str, dict[str, pd.DataFrame]] = {"validation": {}, "test": {}}
    for split in ["validation", "test"]:
        context = read_context(args.dataset_root, split)
        for spec in specs:
            frame = add_context(read_prediction(spec, split), context, link_counts)
            frames_by_split[split][spec.name] = frame

    alpha_grid, selected_alphas = choose_hybrid_alphas(frames_by_split, args.output_root)
    hybrid_frames = build_hybrid_frames(frames_by_split, selected_alphas)
    for split in ["validation", "test"]:
        frames_by_split[split].update(hybrid_frames.get(split, {}))

    overall_rows, slice_rows, decile_rows, order_rows, bootstrap_rows = [], [], [], [], []
    decile_summary_rows = []
    common_models = [m for m in ["full_tabular_30k", "sequence_bigru", "route_local_transformer", "dual_graph_route_transformer"] if m in frames_by_split["test"]]

    for split, frames in frames_by_split.items():
        for model, frame in frames.items():
            for target in TARGETS:
                scopes = {"native_nonnull": valid_target_frame(frame, target)}
                if model in common_models and len(common_models) >= 2:
                    support = common_support(frames, target, common_models)
                    scopes["common_deep_probe_rows"] = valid_target_frame(apply_support(frame, support), target)
                for scope, scoped in scopes.items():
                    if len(scoped) < 100:
                        continue
                    metrics = ranking_metrics(scoped, target, args.kendall_max_rows, args.seed)
                    metrics.update({"model": model, "split": split, "target": target.upper(), "scope": scope, "slice": "all"})
                    dec_rows, dec_summary = decile_stats(scoped, model, split, target, scope)
                    decile_rows.extend(dec_rows)
                    dec_summary.update({"model": model, "split": split, "target": target.upper(), "scope": scope})
                    decile_summary_rows.append(dec_summary)
                    metrics.update(dec_summary)
                    overall_rows.append(metrics)
                    order_rows.extend(order_metrics(scoped, model, split, target, scope))
                    should_bootstrap = split == "test" and (
                        scope == "common_deep_probe_rows" or model.startswith("hybrid_")
                    )
                    if should_bootstrap:
                        ci = bootstrap_ci(scoped, target, args.bootstrap_rounds, args.bootstrap_max_rows, args.seed, args.kendall_max_rows)
                        if ci:
                            ci.update({"model": model, "split": split, "target": target.upper(), "scope": scope})
                            bootstrap_rows.append(ci)
                    for slice_name, mask in slice_definitions(scoped, target).items():
                        if slice_name == "all":
                            continue
                        part = scoped.loc[mask.reindex(scoped.index).fillna(False)]
                        if len(part) < 100:
                            continue
                        sm = ranking_metrics(part, target, args.kendall_max_rows, args.seed)
                        sm.update({"model": model, "split": split, "target": target.upper(), "scope": scope, "slice": slice_name})
                        slice_rows.append(sm)

    overall = pd.DataFrame(overall_rows)
    slices = pd.DataFrame(slice_rows)
    deciles = pd.DataFrame(decile_rows)
    decile_summary = pd.DataFrame(decile_summary_rows)
    orders = pd.DataFrame(order_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)

    overall.to_csv(args.output_root / "overall_metrics.csv", index=False)
    slices.to_csv(args.output_root / "slice_metrics.csv", index=False)
    deciles.to_csv(args.output_root / "decile_stats.csv", index=False)
    decile_summary.to_csv(args.output_root / "decile_summary.csv", index=False)
    orders.to_csv(args.output_root / "order_level_metrics.csv", index=False)
    bootstrap.to_csv(args.output_root / "bootstrap_ci.csv", index=False)
    (args.output_root / "evaluation_manifest.json").write_text(
        json.dumps({
            "models": [s.name for s in specs] + list(selected_alphas.keys()),
            "targets": [t.upper() for t in TARGETS],
            "bootstrap_rounds": args.bootstrap_rounds,
            "bootstrap_unit": "order_id block",
            "hybrid_alpha_selection": "validation AP",
            "output_root": str(args.output_root),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(args.output_root, args.report_path, overall, orders, slices, bootstrap, alpha_grid)
    print(args.output_root)
    print(args.report_path)


if __name__ == "__main__":
    main()
