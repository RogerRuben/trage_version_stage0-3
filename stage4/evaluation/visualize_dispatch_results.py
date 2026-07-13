"""Create compact Stage4 pricing-dispatch visualization artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("stage4/output/pricing_dispatch"))
    parser.add_argument("--summary", type=Path, default=Path("stage4/docs/results/scenario_summary.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("stage4/docs/figures"))
    return parser.parse_args()


def save(fig: plt.Figure, root: Path, name: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(root / f"{name}.png", dpi=160)
    fig.savefig(root / f"{name}.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.summary)

    strategy = summary[summary["experiment_family"].eq("strategy")].groupby("dispatch_strategy", as_index=False).agg(
        match_rate=("match_rate", "mean"),
        cancel_rate=("cancel_rate", "mean"),
        platform_profit=("platform_profit", "mean"),
        passenger_gc=("mean_passenger_generalized_cost", "mean"),
        hv_stress=("HV_stress_burden", "mean"),
        av_stress=("AV_stress_exposure", "mean"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col, title in zip(axes.ravel(), ["match_rate", "cancel_rate", "platform_profit", "passenger_gc", "hv_stress", "av_stress"], ["Match rate", "Cancel rate", "Platform profit", "Passenger generalized cost", "HV stress burden", "AV stress exposure"]):
        ax.barh(strategy["dispatch_strategy"], strategy[col])
        ax.set_title(title)
    save(fig, args.output_root, "strategy_comparison")

    window_paths = sorted(args.run_root.glob("fold=*/exp=*strategy/window_log.parquet"))
    if window_paths:
        window = pd.read_parquet(window_paths[0])
        window["window_time"] = pd.to_datetime(window["window_time"], errors="coerce")
        fig, ax = plt.subplots(figsize=(12, 5))
        for col in ["new_orders", "pending_orders", "matched_orders", "cancelled_orders", "available_AV", "available_HV"]:
            if col in window:
                ax.plot(window["window_time"], window[col], label=col)
        ax.set_title("Window-level dispatch dynamics")
        ax.legend()
        save(fig, args.output_root, "window_time_series")

    order_paths = sorted(args.run_root.glob("fold=*/exp=*/order_log.parquet"))
    if order_paths:
        sample = pd.concat([pd.read_parquet(path) for path in order_paths[: min(12, len(order_paths))]], ignore_index=True)
        served = sample[sample.get("served", False).fillna(False)].copy()
        if len(served):
            served["stress_decile"] = pd.qcut(served[["lcs_expected", "pmis_expected", "rts_expected"]].mean(axis=1), 10, duplicates="drop")
            fare = served.groupby(["quoted_vehicle_type", "stress_decile"], observed=True)["quoted_fare"].mean().reset_index()
            fig, ax = plt.subplots(figsize=(12, 5))
            for vehicle_type, group in fare.groupby("quoted_vehicle_type"):
                ax.plot(range(len(group)), group["quoted_fare"], marker="o", label=vehicle_type)
            ax.set_title("Mean passenger fare by stress decile and vehicle type")
            ax.set_xlabel("Stress decile")
            ax.set_ylabel("Fare")
            ax.legend()
            save(fig, args.output_root, "fare_by_stress_decile")

            flow = served.assign(stress_level=np.where(served["core_overall_high_stress_probability"].ge(0.5), "high", "normal"))
            flow = flow.groupby(["stress_level", "quoted_vehicle_type", "served"], as_index=False).size()
            fig, ax = plt.subplots(figsize=(8, 5))
            labels = flow.apply(lambda r: f"{r['stress_level']}→{r['quoted_vehicle_type']}→served={r['served']}", axis=1)
            ax.barh(labels, flow["size"])
            ax.set_title("Stress → AV/HV assignment → served flow")
            save(fig, args.output_root, "stress_assignment_alluvial")

            spatial = served.copy()
            if {"origin_lon", "origin_lat"}.issubset(spatial.columns):
                spatial["lon_bin"] = pd.cut(spatial["origin_lon"], bins=12)
                spatial["lat_bin"] = pd.cut(spatial["origin_lat"], bins=12)
            else:
                zone_parts = spatial["zone"].astype(str).str.split(":", expand=True)
                spatial["lon_bin"] = zone_parts[0]
                spatial["lat_bin"] = zone_parts[1]
            pivot = spatial.pivot_table(index="lat_bin", columns="lon_bin", values="quoted_fare", aggfunc="mean")
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(pivot.fillna(pivot.mean().mean()), origin="lower", aspect="auto")
            ax.set_title("Spatial heatmap: average fare")
            fig.colorbar(im, ax=ax)
            save(fig, args.output_root, "spatial_average_fare_heatmap")

    welfare = summary.groupby("pricing_scenario", as_index=False).agg(
        passenger=("mean_passenger_generalized_cost", "mean"),
        platform=("platform_profit", "mean"),
        driver=("HV_net_income", "mean"),
        service=("match_rate", "mean"),
        odd=("AV_ODD_violation_rate", "mean"),
    )
    metrics = ["service", "platform", "driver", "passenger", "odd"]
    scaled = welfare.copy()
    for col in metrics:
        values = scaled[col].astype(float)
        if col in {"passenger", "odd"}:
            values = -values
        denom = values.max() - values.min()
        scaled[col] = 0.5 if denom == 0 else (values - values.min()) / denom
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    for _, row in scaled.iterrows():
        vals = [row[m] for m in metrics] + [row[metrics[0]]]
        ax.plot(angles, vals, label=row["pricing_scenario"])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_title("Three-stakeholder normalized welfare radar")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.05))
    save(fig, args.output_root, "stakeholder_welfare_radar")

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(summary["platform_profit"], summary["HV_stress_burden"], s=summary["match_rate"] * 250, c=pd.factorize(summary["pricing_scenario"])[0], alpha=0.75)
    ax.set_xlabel("Platform profit")
    ax.set_ylabel("HV stress burden")
    ax.set_title("Efficiency-fairness Pareto proxy")
    save(fig, args.output_root, "efficiency_fairness_pareto")

    heat = summary[summary["experiment_family"].isin(["av_penetration", "odd_profile"])]
    if not heat.empty:
        pivot = summary.pivot_table(index="av_penetration", columns="odd_profile", values="match_rate", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(pivot.fillna(np.nan), aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title("AV penetration × ODD profile: match rate")
        fig.colorbar(im, ax=ax)
        save(fig, args.output_root, "av_penetration_odd_heatmap")

    catalog = pd.DataFrame({
        "figure": sorted([path.name for path in args.output_root.glob("*.png")]),
        "source": "stage4 dynamic pricing-dispatch experiment",
    })
    catalog.to_csv(args.output_root / "figure_catalog.csv", index=False)
    print(f"wrote {len(catalog)} PNG figures to {args.output_root}")


if __name__ == "__main__":
    main()
