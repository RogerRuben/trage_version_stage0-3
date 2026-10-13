"""One compact trade-off figure from completed aggregate JSON, no experiments."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

from .scheme_a_layout_confirmation import DOC, OUT


def render(root):
    started = perf_counter()
    root = Path(root).resolve()
    target = root / DOC / "six_case_tradeoffs.png"
    if target.exists():
        raise ValueError("existing figure retained; no silent overwrite")
    os.environ["MPLCONFIGDIR"] = str(root / OUT / "mplconfig")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import psutil

    data = json.loads((root / DOC / "six_case_results.json").read_text(encoding="utf-8"))
    names = ("Q05_REFERENCE_LABELS", "Q10_REFERENCE_EXISTING_NOT_RERUN", "Q10_LABEL_SEED2", "Q20_REFERENCE_LABELS")
    labels = ("5%", "10% reference\n(existing)", "10% new seed", "20%")
    colors = ("#4477AA", "#777777", "#EE7733", "#228833")
    rows = [data["conditions"][name]["joint_minus_compatible"] for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    for row, label, color in zip(rows, labels, colors):
        axes[0].scatter(row["C_loaded_utilization_pp"], row["completed_orders"], s=90, c=color, edgecolor="white", zorder=3)
        axes[0].annotate(label.replace("\n", " "), (row["C_loaded_utilization_pp"], row["completed_orders"]),
            xytext=(6, 7), textcoords="offset points", fontsize=9)
    axes[0].axhline(0, color="#555555", linewidth=.8)
    axes[0].axvline(0, color="#555555", linewidth=.8)
    axes[0].set(title="A. Resource conversion versus fleet service", xlabel="C loaded-hour utilization gain (percentage points)",
        ylabel="Change in whole-fleet completed orders", xlim=(-.05, 1.47), ylim=(-30, 60))
    axes[0].grid(alpha=.2)
    axes[0].text(.02, .025, "Joint minus compatible hotspots; one date, not confidence intervals.",
        transform=axes[0].transAxes, fontsize=8, color="#555555")
    empty = [row["total_empty_km"] for row in rows]
    axes[1].bar(range(4), empty, color=["#AA4455" if value > 0 else "#228833" for value in empty], width=.58)
    axes[1].axhline(0, color="#555555", linewidth=.8)
    axes[1].set_xticks(range(4), labels, fontsize=9)
    axes[1].set(title="B. Empty-distance trade-off", ylabel="Change in total empty distance (km)", ylim=(-380, 145))
    axes[1].grid(axis="y", alpha=.2)
    axes[1].set_axisbelow(True)
    for i, value in enumerate(empty):
        axes[1].text(i, value + (8 if value >= 0 else -10), f"{value:+.1f}", ha="center",
            va="bottom" if value >= 0 else "top", fontsize=10)
    fig.suptitle("Fixed common control: operational gains are not universal dominance", fontsize=13)
    fig.savefig(target, dpi=170, facecolor="white")
    plt.close(fig)
    memory = psutil.Process().memory_info()
    return dict(file=target.as_posix(), runtime_s=perf_counter() - started,
        peak_rss_mib=max(memory.rss, getattr(memory, "peak_wset", memory.rss)) / 2**20,
        input_is_aggregate_only=True, new_experiment=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(render(args.root)), flush=True)


if __name__ == "__main__":
    main()
