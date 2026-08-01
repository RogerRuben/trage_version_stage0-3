"""Generate the frozen Stage 2 v4 human reports and CSV attachments."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _metric_rows(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target, metrics in evaluation["continuous_metrics"].items():
        rows.append({"family": "continuous", "target": target, "variant": "deep", **metrics})
    for target, variants in evaluation["tail_metrics"].items():
        for variant, metrics in variants.items():
            rows.append({"family": "tail", "target": target, "variant": variant, **metrics})
    for target, metrics in evaluation["interval_metrics"].items():
        rows.append({"family": "interval_90", "target": target, "variant": "conformal", **metrics})
    return rows


def _subgroup_rows(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dimension, groups in evaluation["slice_metrics"].items():
        for subgroup, payload in groups.items():
            for target in ("lcs", "rts", "lcs_tail"):
                metrics = payload.get(target, {})
                rows.append(
                    {
                        "slice_dimension": dimension,
                        "subgroup": subgroup,
                        "row_count": payload.get("row_count"),
                        "target": target,
                        **metrics,
                    }
                )
    return rows


def _entry_time_audit(dataset_root: Path, dates: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    columns = [
        "order_id",
        "route_sequence",
        "entry_time_pass_1",
        "entry_time_pass_2",
        "estimated_entry_time",
    ]
    for date in dates:
        revealed = pd.read_parquet(
            dataset_root / "revealed_route_proxy" / f"day={date}.parquet",
            columns=columns,
        )
        oracle = pd.read_parquet(
            dataset_root / "oracle_timing" / f"day={date}.parquet",
            columns=["order_id", "route_sequence", "oracle_entry_time"],
        )
        if not revealed[["order_id", "route_sequence"]].equals(
            oracle[["order_id", "route_sequence"]]
        ):
            oracle = revealed[["order_id", "route_sequence"]].merge(
                oracle,
                on=["order_id", "route_sequence"],
                how="left",
                validate="one_to_one",
            )
        pass_gap = np.abs(
            pd.to_numeric(revealed["entry_time_pass_2"], errors="coerce")
            - pd.to_numeric(revealed["entry_time_pass_1"], errors="coerce")
        )
        oracle_gap = np.abs(
            pd.to_numeric(revealed["estimated_entry_time"], errors="coerce")
            - pd.to_numeric(oracle["oracle_entry_time"], errors="coerce")
        )
        for name, values in (
            ("pass2_minus_pass1_abs_s", pass_gap),
            ("revealed_minus_oracle_abs_s", oracle_gap),
        ):
            valid = values.dropna().to_numpy(float)
            rows.append(
                {
                    "date": date,
                    "split": "test" if date == "20161031" else "validation" if date >= "20161025" else "train",
                    "audit": name,
                    "count": int(len(valid)),
                    "mean_s": float(np.mean(valid)) if len(valid) else None,
                    "median_s": float(np.median(valid)) if len(valid) else None,
                    "p90_s": float(np.quantile(valid, 0.9)) if len(valid) else None,
                    "p99_s": float(np.quantile(valid, 0.99)) if len(valid) else None,
                }
            )
    return pd.DataFrame(rows)


def generate_reports(output_root: str | Path, docs_root: str | Path) -> dict[str, Any]:
    output = Path(output_root)
    docs = Path(docs_root)
    reports = output / "reports"
    preflight = _read(docs / "stage2_v4_preflight.json")
    dataset = _read(output / "route_conditioned_dataset/dataset_manifest.json")
    baseline = _read(output / "models/baselines/baseline_manifest.json")
    deep = _read(output / "models/rc_mstnet_v4/model_manifest.json")
    calibration = _read(output / "models/calibration/calibration_manifest.json")
    evaluation = _read(output / "evaluation/final_test/evaluation_report.json")

    metric_rows = _metric_rows(evaluation)
    _atomic_csv(reports / "metrics_by_target.csv", pd.DataFrame(metric_rows))
    _atomic_csv(reports / "metrics_by_subgroup.csv", pd.DataFrame(_subgroup_rows(evaluation)))

    bootstrap_rows = []
    for target, metrics in evaluation["order_cluster_bootstrap"].items():
        interval = metrics.get("mae_ci95", [None, None])
        bootstrap_rows.append(
            {
                "target": target,
                "metric": "mae",
                "replicates": metrics.get("replicates"),
                "seed": metrics.get("seed"),
                "ci95_lower": interval[0],
                "ci95_upper": interval[1],
            }
        )
    _atomic_csv(reports / "bootstrap_intervals.csv", pd.DataFrame(bootstrap_rows))

    calibration_rows = []
    for target, payload in calibration["tail_metrics"].items():
        for method in ("platt", "isotonic"):
            calibration_rows.append(
                {
                    "date": "20161027",
                    "target": target,
                    "method": method,
                    "selected": method == payload["selected_method"],
                    **payload[method],
                }
            )
    _atomic_csv(reports / "calibration_metrics.csv", pd.DataFrame(calibration_rows))

    order_rows = [
        {"target": target, **metrics}
        for target, metrics in evaluation["order_metrics"].items()
    ]
    _atomic_csv(reports / "order_aggregation_metrics.csv", pd.DataFrame(order_rows))

    ablation_rows = []
    for target, variants in baseline["validation_metrics"].items():
        for variant, metrics in variants.items():
            ablation_rows.append(
                {
                    "scope": "validation_baseline_component",
                    "target": target,
                    "variant": variant,
                    **metrics,
                }
            )
    _atomic_csv(reports / "ablation_results.csv", pd.DataFrame(ablation_rows))

    dates = list(dataset["tracks"]["revealed_route_proxy"]["order_count_by_date"])
    entry_audit = _entry_time_audit(output / "route_conditioned_dataset", dates)
    _atomic_csv(reports / "entry_time_gap_audit.csv", entry_audit)

    track = dataset["tracks"]["revealed_route_proxy"]
    _atomic_text(
        docs / "stage2_v4_dataset_report.md",
        "# Stage 2 v4 Dataset Report\n\n"
        f"- Engineering status: `{dataset['engineering_status']}`\n"
        f"- Orders: `{sum(track['order_count_by_date'].values()):,}`\n"
        f"- Route tokens per track: `{track['row_count']:,}`\n"
        f"- Supervised traversals/history events: `{dataset['history_event_count']:,}`\n"
        f"- Revealed and oracle tracks have identical row counts: `{dataset['tracks']['oracle_timing']['row_count'] == track['row_count']}`\n"
        f"- Time leakage violations: `{preflight['counters']['self_order_history_candidate_count']}`\n"
        "- Oracle timing is diagnostic only and is not Stage 3 eligible.\n"
        "- Entry-time diagnostics: `stage2/output_v4/reports/entry_time_gap_audit.csv`.\n",
    )

    baseline_lines = [
        "# Stage 2 v4 Baseline Report",
        "",
        f"- Engineering status: `{baseline['engineering_status']}`",
        f"- Fit dates: `{', '.join(baseline['fit_dates'])}`",
        f"- Validation dates: `{', '.join(baseline['validation_dates'])}`",
        f"- Test rows read: `{baseline['test_rows_read']}`",
        "",
        "| Target | Best validation variant | RMSE/Brier |",
        "|---|---:|---:|",
    ]
    for target, variants in baseline["validation_metrics"].items():
        key = "brier" if target.endswith("tail_event") else "rmse"
        available = [(name, value.get(key)) for name, value in variants.items() if value.get(key) is not None]
        name, value = min(available, key=lambda item: item[1])
        baseline_lines.append(f"| {target} | {name} | {_fmt(value)} |")
    baseline_lines.append("\nDetailed component comparisons are in `stage2/output_v4/reports/ablation_results.csv`.\n")
    _atomic_text(docs / "stage2_v4_baseline_report.md", "\n".join(baseline_lines))

    _atomic_text(
        docs / "stage2_v4_deep_report.md",
        "# Stage 2 v4 RC-MSTNet Report\n\n"
        f"- Engineering status: `{deep['engineering_status']}`\n"
        f"- Device: `{deep['device']}`\n"
        f"- Model ID: `{deep['model_id']}`\n"
        f"- Best epoch: `{deep['best_epoch']}`\n"
        f"- Best validation loss: `{_fmt(deep['best_validation_loss'])}`\n"
        f"- Test rows read during fitting: `{deep['test_rows_read']}`\n"
        f"- Runtime: `{deep['runtime_s'] / 3600:.2f} h`\n",
    )

    cal_lines = [
        "# Stage 2 v4 Calibration Report",
        "",
        f"- Engineering status: `{calibration['engineering_status']}`",
        f"- Fit date: `{calibration['fit_dates'][0]}` only",
        f"- Test rows read: `{calibration['test_rows_read']}`",
        f"- Calibration model ID: `{calibration['calibration_model_id']}`",
        "",
        "| Target | Selected method | Platt Brier | Isotonic Brier | Q90 residual |",
        "|---|---|---:|---:|---:|",
    ]
    for target in ("lcs", "rts"):
        item = calibration["tail_metrics"][target]
        cal_lines.append(
            f"| {target} | {item['selected_method']} | {_fmt(item['platt']['brier'])} | {_fmt(item['isotonic']['brier'])} | {_fmt(calibration['conformal_absolute_residual_q90'][target])} |"
        )
    _atomic_text(docs / "stage2_v4_calibration_report.md", "\n".join(cal_lines) + "\n")

    eval_lines = [
        "# Stage 2 v4 Final Evaluation",
        "",
        f"- Engineering status: `{evaluation['engineering_status']}`",
        f"- Frozen Test date: `{evaluation['test_dates'][0]}`",
        f"- Orders: `{evaluation['order_count']:,}`",
        f"- Traversal/route-token predictions: `{evaluation['traversal_row_count']:,}`",
        f"- Test tuning violations: `{evaluation['test_tuning_violation_count']}`",
        "",
        "| Target | Count | MAE | RMSE | Pearson |",
        "|---|---:|---:|---:|---:|",
    ]
    for target, item in evaluation["continuous_metrics"].items():
        eval_lines.append(f"| {target} | {item['count']:,} | {_fmt(item['mae'])} | {_fmt(item['rmse'])} | {_fmt(item['pearson'])} |")
    eval_lines.extend(["", "| Tail | Variant | Brier | ROC AUC | ECE |", "|---|---|---:|---:|---:|"])
    for target, variants in evaluation["tail_metrics"].items():
        for variant, item in variants.items():
            eval_lines.append(f"| {target} | {variant} | {_fmt(item['brier'])} | {_fmt(item['roc_auc'])} | {_fmt(item['ece'])} |")
    eval_lines.extend(["", "| Interval | Coverage | Mean width |", "|---|---:|---:|"])
    for target, item in evaluation["interval_metrics"].items():
        eval_lines.append(f"| {target} | {_fmt(item['coverage'])} | {_fmt(item['mean_width'])} |")
    eval_lines.append("\nSubgroup, bootstrap, order aggregation, calibration, entry-time and baseline-component details are under `stage2/output_v4/reports/`.\n")
    _atomic_text(docs / "stage2_v4_final_evaluation.md", "\n".join(eval_lines))

    return {
        "engineering_status": "PASS",
        "document_count": 5,
        "attachment_count": 7,
        "reports_root": str(reports),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--docs-root", type=Path, required=True)
    args = parser.parse_args()
    result = generate_reports(args.output_root, args.docs_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
