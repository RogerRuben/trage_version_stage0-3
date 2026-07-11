"""Generate Stage1 monotonicity, spatial, POI, and threshold-validity audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DIMENSIONS = ["lcs", "iis", "gns", "rts", "pmis"]
INDICATORS = {
    "lcs": ["low_speed_ratio", "stop_duration_ratio", "speed_cv"],
    "iis": ["intersection_low_speed_time", "intersection_stop_time", "turn_angle"],
    "gns": ["curvature_deg_per_km_link", "link_fragmentation", "endpoint_degree"],
    "rts": ["excess_time_ratio", "tail_delay_ratio", "travel_time_sec"],
    "pmis": ["delay_on_poi_link", "low_speed_ratio_on_poi_link", "stop_time_on_poi_link"],
}

CONTEXT_FEATURES = {
    "iis": ["node_degree"],
    "gns": ["minor_road"],
    "pmis": ["activity_intensity_index"],
}


def add_derived_audit_fields(frame: pd.DataFrame) -> pd.DataFrame:
    if "activity_intensity_index" in frame.columns:
        poi_exposed = frame["activity_intensity_index"].fillna(0).gt(0)
        if "low_speed_ratio_on_poi_link" not in frame.columns and "low_speed_ratio" in frame.columns:
            frame["low_speed_ratio_on_poi_link"] = frame["low_speed_ratio"].where(poi_exposed, 0.0)
        if "stop_time_on_poi_link" not in frame.columns and "stop_time_sec" in frame.columns:
            frame["stop_time_on_poi_link"] = frame["stop_time_sec"].where(poi_exposed, 0.0)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("stage1/output/prediction_split"))
    parser.add_argument("--date", required=True)
    parser.add_argument("--roads", type=Path, required=True)
    parser.add_argument("--matched-dir", type=Path)
    parser.add_argument("--poi-exposure", type=Path)
    parser.add_argument("--split", choices=["train", "validation", "test"])
    parser.add_argument("--precomputed-sensitivity", type=Path)
    return parser.parse_args()


def monotonicity(primitives_dir: Path, labels_dir: Path) -> pd.DataFrame:
    totals: list[pd.DataFrame] = []
    for primitive_path in sorted(primitives_dir.glob("*.parquet")):
        part = primitive_path.stem.split("=")[-1].split("_")[-1]
        label_path = labels_dir / f"part={part}.parquet"
        primitive = add_derived_audit_fields(pd.read_parquet(primitive_path))
        label = pd.read_parquet(label_path, columns=["order_id", "link_seq"] + [f"{d}_pct_link" for d in DIMENSIONS])
        frame = primitive.merge(label, on=["order_id", "link_seq"], how="inner", validate="one_to_one")
        for dimension in DIMENSIONS:
            valid_label = frame[f"{dimension}_pct_link"].notna()
            decile = np.minimum((frame.loc[valid_label, f"{dimension}_pct_link"] * 10).astype(int), 9) + 1
            for indicator in INDICATORS[dimension]:
                value = frame.loc[valid_label, indicator]
                value = value.abs() if indicator == "turn_angle" else value
                summary = pd.DataFrame({"decile": decile, "value": value}).groupby("decile").value.agg(["sum", "count"]).reset_index()
                summary["dimension"] = dimension; summary["indicator"] = indicator
                totals.append(summary)
    combined = pd.concat(totals, ignore_index=True)
    result = combined.groupby(["dimension", "indicator", "decile"], as_index=False)[["sum", "count"]].sum()
    result["mean"] = result["sum"] / result["count"]
    result["spearman_decile_mean"] = result.groupby(["dimension", "indicator"])["mean"].transform(
        lambda x: pd.Series(x.to_numpy()).corr(pd.Series(range(1, len(x) + 1)), method="spearman")
    )
    return result


def context_descriptors(primitives_dir: Path, labels_dir: Path) -> pd.DataFrame:
    """Summarize contextual variables by label decile without treating them as monotonic anchors."""
    totals: list[pd.DataFrame] = []
    for primitive_path in sorted(primitives_dir.glob("*.parquet")):
        part = primitive_path.stem.split("=")[-1].split("_")[-1]
        label_path = labels_dir / f"part={part}.parquet"
        primitive = add_derived_audit_fields(pd.read_parquet(primitive_path))
        label = pd.read_parquet(label_path, columns=["order_id", "link_seq"] + [f"{d}_pct_link" for d in DIMENSIONS])
        frame = primitive.merge(label, on=["order_id", "link_seq"], how="inner", validate="one_to_one")
        for dimension, features in CONTEXT_FEATURES.items():
            valid_label = frame[f"{dimension}_pct_link"].notna()
            if not valid_label.any():
                continue
            decile = np.minimum((frame.loc[valid_label, f"{dimension}_pct_link"] * 10).astype(int), 9) + 1
            for feature in features:
                if feature not in frame.columns:
                    continue
                summary = pd.DataFrame({
                    "decile": decile,
                    "value": frame.loc[valid_label, feature],
                }).groupby("decile").value.agg(["sum", "count"]).reset_index()
                summary["dimension"] = dimension
                summary["feature"] = feature
                totals.append(summary)
    if not totals:
        return pd.DataFrame(columns=["dimension", "feature", "decile", "sum", "count", "mean"])
    result = pd.concat(totals, ignore_index=True).groupby(
        ["dimension", "feature", "decile"], as_index=False
    )[["sum", "count"]].sum()
    result["mean"] = result["sum"] / result["count"]
    return result


def plot_monotonicity(table: pd.DataFrame, figures: Path) -> None:
    for dimension in DIMENSIONS:
        data = table[table.dimension == dimension]
        indicators = data.indicator.unique()
        fig, axes = plt.subplots(1, len(indicators), figsize=(4 * len(indicators), 3.5), squeeze=False)
        for ax, indicator in zip(axes.flat, indicators):
            shown = data[data.indicator == indicator]
            ax.plot(shown.decile, shown["mean"], marker="o")
            rho = shown.spearman_decile_mean.iloc[0]
            ax.set_title(f"{indicator}\nSpearman={rho:.2f}")
            ax.set_xlabel(f"{dimension.upper()} decile"); ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(figures / f"{dimension}_decile_validation.png", dpi=180); plt.close(fig)


def spatial_plot(labels_dir: Path, roads_path: Path, figures: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(labels_dir.glob("*.parquet")):
        frame = pd.read_parquet(path, columns=["link_id"] + [f"{d}_pct_link" for d in DIMENSIONS])
        rows.append(frame.groupby("link_id")[[f"{d}_pct_link" for d in DIMENSIONS]].agg(["sum", "count"]))
    combined = pd.concat(rows).groupby(level=0).sum()
    summary = pd.DataFrame(index=combined.index)
    for dimension in DIMENSIONS:
        summary[dimension] = combined[(f"{dimension}_pct_link", "sum")] / combined[(f"{dimension}_pct_link", "count")]
    roads = gpd.read_parquet(roads_path).to_crs(32649).merge(summary, left_on="link_id", right_index=True, how="left")
    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    for ax, dimension in zip(axes, DIMENSIONS):
        roads.plot(ax=ax, column=dimension, cmap="magma", linewidth=0.8, vmin=0.5, vmax=1.0, missing_kwds={"color": "#dddddd"})
        ax.set_title(f"Mean {dimension.upper()} percentile"); ax.axis("off")
    fig.tight_layout(); fig.savefig(figures / "high_stress_spatial.png", dpi=180); plt.close(fig)
    return summary.reset_index()


def poi_hourly(primitives_dir: Path, labels_dir: Path) -> pd.DataFrame:
    output = []
    categories = ["school", "hospital", "commercial", "transit"]
    for primitive_path in sorted(primitives_dir.glob("*.parquet")):
        part = primitive_path.stem.split("=")[-1].split("_")[-1]
        primitive = pd.read_parquet(primitive_path)
        label = pd.read_parquet(labels_dir / f"part={part}.parquet", columns=["order_id", "link_seq", "pmis_pct_link"])
        frame = primitive.merge(label, on=["order_id", "link_seq"], how="inner")
        hour = pd.to_datetime(frame.enter_time, unit="s", utc=True).dt.tz_convert("Asia/Shanghai").dt.hour
        for category in categories:
            density = frame.get(f"poi_density_100m_{category}", pd.Series(0, index=frame.index))
            selected = density.gt(0) & frame.pmis_pct_link.notna()
            if selected.any():
                data = pd.DataFrame({
                    "hour": hour[selected], "pmis": frame.loc[selected, "pmis_pct_link"],
                    "stop_time": frame.loc[selected, "stop_time_sec"], "low_speed": frame.loc[selected, "low_speed_ratio"],
                }).groupby("hour").agg(pmis_sum=("pmis", "sum"), stop_sum=("stop_time", "sum"), low_sum=("low_speed", "sum"), count=("pmis", "size")).reset_index()
                data["category"] = category; output.append(data)
    result = pd.concat(output, ignore_index=True).groupby(["category", "hour"], as_index=False).sum()
    for column in ["pmis", "stop", "low"]:
        result[f"mean_{column}"] = result[f"{column}_sum"] / result["count"]
    return result


def plot_poi(table: pd.DataFrame, figures: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, category in zip(axes.flat, ["school", "hospital", "commercial", "transit"]):
        shown = table[table.category == category]
        ax.plot(shown.hour, shown.mean_pmis, marker="o", label="PMIS")
        ax.set_title(category); ax.set_xticks(range(0, 24, 3)); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(figures / "poi_pmis_by_hour.png", dpi=180); plt.close(fig)


def order_monotonicity(primitives_dir: Path, order_path: Path) -> pd.DataFrame:
    parts = []
    for path in sorted(primitives_dir.glob("*.parquet")):
        frame = pd.read_parquet(path)
        frame["poi_low_speed"] = frame.low_speed_ratio * frame.activity_intensity_index
        summary = frame.groupby("order_id").agg(
            avg_low_speed_ratio=("low_speed_ratio", "mean"), stop_count=("stop_count", "sum"),
            actual_travel_time=("travel_time_sec", "sum"), intersection_delay=("intersection_low_speed_time", "sum"),
            poi_low_speed=("poi_low_speed", "mean"), tail_delay=("tail_delay_ratio", "max"),
            mean_curvature=("curvature_deg_per_km_link", "mean"),
        ).reset_index()
        parts.append(summary)
    primitive_orders = pd.concat(parts, ignore_index=True)
    labels = pd.read_parquet(order_path)
    frame = labels.merge(primitive_orders, on="order_id", how="left", validate="one_to_one")
    mapping = {
        "lcs": ["avg_low_speed_ratio", "stop_count"],
        "iis": ["intersection_delay"], "gns": ["mean_curvature"],
        "rts": ["actual_travel_time", "tail_delay"], "pmis": ["poi_low_speed"],
    }
    rows = []
    for dimension, indicators in mapping.items():
        valid_label = frame[f"{dimension}_mean"].notna()
        ranked = frame.loc[valid_label, f"{dimension}_mean"].rank(pct=True)
        decile = np.minimum((ranked * 10).astype(int), 9) + 1
        for indicator in indicators:
            summary = pd.DataFrame({
                "decile": decile,
                "value": frame.loc[valid_label, indicator],
            }).groupby("decile").value.mean().reset_index(name="mean")
            summary["dimension"] = dimension; summary["indicator"] = indicator
            summary["spearman_decile_mean"] = pd.Series(summary["mean"].to_numpy()).corr(
                pd.Series(summary.decile.to_numpy()), method="spearman"
            )
            rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def plot_order_monotonicity(table: pd.DataFrame, figures: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    for ax, dimension in zip(axes, DIMENSIONS):
        shown_dimension = table[table.dimension == dimension]
        for indicator, shown in shown_dimension.groupby("indicator"):
            normalized = shown["mean"] / max(float(shown["mean"].max()), 1e-9)
            ax.plot(shown.decile, normalized, marker="o", label=indicator)
        ax.set_title(dimension.upper()); ax.set_xlabel("Order decile"); ax.grid(alpha=0.25); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figures / "order_decile_validation.png", dpi=180); plt.close(fig)


def threshold_sensitivity(matched_dir: Path | None, poi_path: Path | None) -> dict:
    result: dict[str, object] = {}
    if matched_dir:
        low = {3: [0.0, 0.0], 5: [0.0, 0.0], 8: [0.0, 0.0]}
        intersection = {20: [0.0, 0.0], 30: [0.0, 0.0], 50: [0.0, 0.0]}
        stop_runs = {3: [0, 0.0], 5: [0, 0.0], 10: [0, 0.0]}
        for path in sorted(matched_dir.glob("*.parquet")):
            frame = pd.read_parquet(path, columns=["order_id", "timestamp", "dt_s", "speed_kmh", "intersection_distance_m"])
            dt = frame.dt_s.fillna(0).clip(0, 120)
            for speed in low:
                low[speed][0] += float(dt[frame.speed_kmh < speed].sum()); low[speed][1] += float(dt.sum())
            for radius in intersection:
                intersection[radius][0] += float(dt[(frame.intersection_distance_m <= radius) & (frame.speed_kmh < 5)].sum())
                intersection[radius][1] += float(dt.sum())
            stopped = frame.speed_kmh.lt(2)
            same = frame.order_id.eq(frame.order_id.shift())
            run_id = (stopped & (~stopped.shift(fill_value=False) | ~same)).cumsum()
            durations = dt[stopped].groupby(run_id[stopped]).sum()
            for threshold in stop_runs:
                valid = durations[durations >= threshold]
                stop_runs[threshold][0] += int(len(valid)); stop_runs[threshold][1] += float(valid.sum())
        result["low_speed_threshold"] = {str(k): v[0] / v[1] if v[1] else None for k, v in low.items()}
        result["intersection_buffer"] = {str(k): v[0] / v[1] if v[1] else None for k, v in intersection.items()}
        result["stop_duration"] = {str(k): {"count": v[0], "seconds": v[1]} for k, v in stop_runs.items()}
    if poi_path:
        exposure = pd.read_parquet(poi_path)
        result["poi_buffer"] = {
            str(radius): int(exposure[[c for c in exposure if c.startswith(f"poi_count_{radius}m_")]].sum(axis=1).gt(0).sum())
            for radius in [50, 100, 200]
        }
    return result


def main() -> None:
    args = parse_args()
    primitives = args.output_root / "primitives" / f"day={args.date}"
    labels = args.output_root / "link_labels" / f"day={args.date}"
    report_dir = args.output_root / "validity"
    if args.split:
        report_dir = report_dir / args.split
    report_dir = report_dir / f"day={args.date}"
    figures = report_dir / "figures"; figures.mkdir(parents=True, exist_ok=True)
    mono = monotonicity(primitives, labels); mono.to_csv(report_dir / "link_decile_monotonicity.csv", index=False)
    context = context_descriptors(primitives, labels)
    context.to_csv(report_dir / "link_decile_context_descriptors.csv", index=False)
    plot_monotonicity(mono, figures)
    order_path = args.output_root / "order_labels" / f"day={args.date}.parquet"
    order_mono = order_monotonicity(primitives, order_path)
    order_mono.to_csv(report_dir / "order_decile_monotonicity.csv", index=False)
    plot_order_monotonicity(order_mono, figures)
    spatial = spatial_plot(labels, args.roads, figures); spatial.to_parquet(report_dir / "link_spatial_summary.parquet", index=False)
    poi = poi_hourly(primitives, labels); poi.to_csv(report_dir / "poi_hourly_validity.csv", index=False); plot_poi(poi, figures)
    sensitivity = threshold_sensitivity(args.matched_dir, args.poi_exposure)
    if args.precomputed_sensitivity and args.precomputed_sensitivity.exists():
        sensitivity.update(json.loads(args.precomputed_sensitivity.read_text(encoding="utf-8")))
    orders = pd.read_parquet(order_path)
    maximum = orders[[f"{dimension}_max" for dimension in DIMENSIONS]].max(axis=1)
    high_rates = {str(x): float(maximum.ge(x).mean()) for x in [0.85, 0.90, 0.95]}
    sensitivity["high_stress_threshold"] = high_rates
    (report_dir / "sensitivity.json").write_text(json.dumps(sensitivity, ensure_ascii=False, indent=2), encoding="utf-8")
    correlations = mono.groupby(["dimension", "indicator"]).spearman_decile_mean.first()
    report = f"""# Stage1 label validity report: {args.date}

## Link-level monotonicity

The decile audit covers LCS, IIS, GNS, RTS, and PMIS. Detailed decile means and Spearman coefficients are stored in `link_decile_monotonicity.csv`; dimension plots are in `figures/`.

Median indicator-decile Spearman correlation: **{correlations.median():.3f}**.

`activity_intensity_index`, `minor_road`, and `node_degree` are treated as context descriptors rather than core monotonicity anchors; their decile summaries are stored in `link_decile_context_descriptors.csv`.

Order-level decile validation is stored in `order_decile_monotonicity.csv` and `order_decile_validation.png`.

## Spatial and POI validity

`high_stress_spatial.png` maps mean link percentiles. `poi_pmis_by_hour.png` and `poi_hourly_validity.csv` audit school, hospital, commercial, and transit patterns by hour.

## Robustness

Sensitivity results cover low-speed thresholds 3/5/8 km/h, stop-duration thresholds 3/5/10 s, intersection buffers 20/30/50 m, POI buffers 50/100/200 m, and high-stress cutoffs 0.85/0.90/0.95 when the relevant inputs are supplied. See `sensitivity.json`.
"""
    (report_dir / "stage1_label_validity_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
