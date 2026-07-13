"""Create compact figures for the 2016-10-23 single-day ABM run."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FIG = Path("stage4/docs/figures/single_day_abm")
RESULTS = Path("stage4/docs/results")


def _save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=160)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


def main() -> None:
    orders = pd.read_parquet("stage3/output/full_day_20161023/stage4_inputs/stage4_inputs.parquet")
    orders["decision_time"] = pd.to_datetime(orders["decision_time"], utc=True, errors="coerce").dt.tz_convert("Asia/Shanghai")
    hourly = orders.set_index("decision_time").resample("30min").agg(
        orders=("order_id", "count"),
        lcs=("lcs_tail_probability", "mean"),
        pmis=("pmis_tail_probability", "mean"),
        rts=("rts_tail_probability", "mean"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(hourly["decision_time"], hourly["orders"])
    ax.set_title("2016-10-23 full-day order arrivals")
    ax.set_ylabel("orders / 30 min")
    _save(fig, "order_arrival_curve")

    fig, ax = plt.subplots(figsize=(9, 4))
    for col in ["lcs", "pmis", "rts"]:
        ax.plot(hourly["decision_time"], hourly[col], label=col.upper())
    ax.set_title("Predicted Stage3 stress by time of day")
    ax.set_ylabel("mean tail probability")
    ax.legend()
    _save(fig, "stage3_stress_time_series")

    sessions = pd.read_parquet("stage4/data/hv_agent_sessions_20161023.parquet")
    sessions["online_start"] = pd.to_datetime(sessions["online_start"], utc=True).dt.tz_convert("Asia/Shanghai")
    sessions["online_end"] = pd.to_datetime(sessions["online_end"], utc=True).dt.tz_convert("Asia/Shanghai")
    times = pd.date_range(orders["decision_time"].min().floor("h"), orders["decision_time"].max().ceil("h"), freq="30min", tz="Asia/Shanghai")
    online = [((sessions["online_start"] <= t) & (sessions["online_end"] >= t)).sum() for t in times]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(times, online)
    ax.set_title("Observed HV sessions online over time")
    ax.set_ylabel("online HV sessions")
    _save(fig, "hv_online_sessions")

    depots = pd.read_parquet("stage4/data/av_depots.parquet")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(orders["origin_lon"], orders["origin_lat"], s=1, alpha=0.05, label="orders")
    ax.scatter(depots["lon"], depots["lat"], s=60, marker="^", label="AV depots")
    ax.set_title("Training-data AV depots and test-day origins")
    ax.legend()
    _save(fig, "av_depots_map")

    summary = pd.read_csv(RESULTS / "single_day_dispatch_summary.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(summary["strategy"], summary["match_rate"])
    ax.set_ylim(0, 1)
    ax.set_title("Single-day strategy match rate")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, "strategy_match_rate")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(summary["strategy"], summary["platform_profit"])
    ax.set_title("Single-day strategy platform profit")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, "strategy_platform_profit")

    radius = pd.read_csv(RESULTS / "dynamic_radius_summary.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot = radius.pivot_table(index="strategy", columns="search_radius_m", values="orders", aggfunc="sum", fill_value=0)
    pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Radius stage distribution among served/cancelled logged orders")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, "dynamic_radius_distribution")


if __name__ == "__main__":
    main()
