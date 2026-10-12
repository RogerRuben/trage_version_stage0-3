"""Render one compact offline figure from aggregate mechanism JSON only."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

from .scheme_a_layout_mechanism import DOC, OUT


def render(root):
    started = perf_counter()
    root = Path(root).resolve()
    target = root / DOC / "mechanism_overview.png"
    if target.exists():
        raise ValueError("existing figure retained; no silent overwrite")
    os.environ["MPLCONFIGDIR"] = str(root / OUT / "mplconfig")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    import numpy as np
    import psutil

    data = json.loads((root / DOC / "mechanism.json").read_text(encoding="utf-8"))
    groups = ("ALL_HOT_COMMON_CONTROL", "COMPATIBLE_HOT_COMMON_CONTROL", "OPPORTUNITY_JOINT_COMMON_CONTROL")
    labels, colors = ("All-demand hotspots", "Compatible-head hotspots", "Joint opportunity layout"), ("#3D6EA5", "#D48B32", "#31886B")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9), constrained_layout=True)
    for name, label, color in zip(groups, labels, colors):
        values = data["groups"][name]["first_customer_cumulative_all_slot_rates"]
        axes[0].plot([r["delay_s"] / 60 for r in values], [100 * r["assignment_share"] for r in values],
            marker="o", color=color, linewidth=2, markersize=4, label=label)
    axes[0].set(title="A. Access to a first customer", xlabel="Minutes since the original activation time",
        ylabel="Share of all 851 C slots assigned a first customer (%)", xlim=(0, 61), ylim=(0, 70))
    axes[0].grid(alpha=.22)
    axes[0].legend(loc="lower right", fontsize=8, frameon=False)
    axes[0].text(.02, .97, "Unserved windows retained; cohort proportions,\nnot a fitted survival distribution.",
        transform=axes[0].transAxes, va="top", fontsize=8, color="#555555")
    key = f"{groups[0]}_to_{groups[2]}"
    cells = data["paired"][key]["birth_cells_after_layout"]
    delta = np.asarray([r["loaded_delta_h"] for r in cells])
    span = max(1., float(np.max(np.abs(delta))))
    norm = TwoSlopeNorm(vmin=-span, vcenter=0, vmax=span)
    x = [r["center_lon_coarse"] for r in cells]
    y = [r["center_lat_coarse"] for r in cells]
    sizes = [30 + 2 * r["slots"] for r in cells]
    plot = axes[1].scatter(x, y, s=sizes, c=delta, cmap="RdBu_r", norm=norm, alpha=.87, edgecolor="#555555", linewidth=.55)
    small = [r for r in cells if not r["supported_for_rank"]]
    axes[1].scatter([r["center_lon_coarse"] for r in small], [r["center_lat_coarse"] for r in small],
        marker="x", s=32, c="#202020", linewidth=1, label="Fewer than 10 slots")
    axes[1].set(title="B. Joint minus all-demand hotspots", xlabel="Coarse birth-cell longitude (WGS84)",
        ylabel="Coarse birth-cell latitude (WGS84)")
    axes[1].set_aspect(1.2)
    axes[1].grid(alpha=.18)
    axes[1].ticklabel_format(useOffset=False, style="plain")
    bar = fig.colorbar(plot, ax=axes[1], fraction=.047, pad=.025)
    bar.set_label("Difference in C loaded hours", fontsize=9)
    axes[1].text(.01, -.21, "Attributed to the joint birth cell, not trip location or a local causal effect.\nPoint area = slot count; x = fewer than 10 slots. All cells are retained.",
        transform=axes[1].transAxes, fontsize=8, color="#555555", va="top")
    fig.savefig(target, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    memory = psutil.Process().memory_info()
    return dict(file=target.as_posix(), runtime_s=perf_counter() - started,
        peak_rss_mib=max(memory.rss, getattr(memory, "peak_wset", memory.rss)) / 2**20,
        input_is_aggregated_json=True, new_simulation=False, new_routing=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(render(args.root)), flush=True)


if __name__ == "__main__":
    main()
