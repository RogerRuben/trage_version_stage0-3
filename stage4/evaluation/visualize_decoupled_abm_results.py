"""Create compact figures for the decoupled ABM run."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS = Path("stage4/docs/results")
FIG = Path("stage4/docs/figures/decoupled_abm")


def save_bar(frame: pd.DataFrame, x: str, y: str, title: str, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    ax = frame.plot(kind="bar", x=x, y=y, legend=False, figsize=(8, 4))
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", dpi=160)
    plt.savefig(FIG / f"{name}.pdf")
    plt.close()


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    if (RESULTS / "operational_module_comparison.csv").exists():
        op = pd.read_csv(RESULTS / "operational_module_comparison.csv")
        op = op[op["replication_id"].eq(1)].sort_values("operation_setting")
        save_bar(op, "operation_setting", "match_rate", "O0-O3 match-rate comparison", "operational_match_rate")
        save_bar(op, "operation_setting", "empty_vehicle_km", "O0-O3 empty vehicle-km comparison", "operational_empty_km")
        save_bar(op, "operation_setting", "av_assignment_share", "O0-O3 AV assignment share", "operational_av_share")
    if (RESULTS / "request_time_sensitivity_summary.csv").exists():
        rt = pd.read_csv(RESULTS / "request_time_sensitivity_summary.csv")
        rt = rt[rt["replication_id"].eq(1)].sort_values("request_time_scenario")
        save_bar(rt, "request_time_scenario", "match_rate", "Request-time sensitivity: match rate", "request_time_match_rate")
    if (RESULTS / "full_day_odd_infeasibility_decomposition.csv").exists():
        odd = pd.read_csv(RESULTS / "full_day_odd_infeasibility_decomposition.csv")
        odd = odd[odd["vehicle_profile"].eq("moderate_av")]
        cols = ["lcs_hard_violation", "pmis_hard_violation", "rts_hard_violation", "core_uncertainty_violation", "iis_violation"]
        plot = odd[cols].T.reset_index()
        plot.columns = ["violation", "orders"]
        save_bar(plot, "violation", "orders", "Moderate AV infeasibility decomposition", "odd_infeasibility_moderate")
    if (RESULTS / "av_vehicle_hour_summary.csv").exists():
        av = pd.read_csv(RESULTS / "av_vehicle_hour_summary.csv")
        save_bar(av, "replication_id", "av_vehicle_hour_share", "AV vehicle-hour share by replication", "av_vehicle_hour_share")
    if (RESULTS / "decoupled_hv_supply_summary.csv").exists():
        supply = pd.read_csv(RESULTS / "decoupled_hv_supply_summary.csv")
        s1 = supply[supply["replication_id"].eq(1)].copy()
        ax = s1.plot(x="time_bin", y=["target_online_HV", "generated_online_HV"], figsize=(9, 4))
        ax.set_title("Target vs generated HV online curve (replication 1)")
        ax.set_xlabel("30-minute time bin")
        ax.set_ylabel("online HV")
        plt.tight_layout()
        plt.savefig(FIG / "hv_supply_curve_rep1.png", dpi=160)
        plt.savefig(FIG / "hv_supply_curve_rep1.pdf")
        plt.close()
    print({"status": "PASS", "figure_dir": str(FIG)})


if __name__ == "__main__":
    main()
