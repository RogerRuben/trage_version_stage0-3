"""Render the fixed D1--D3 static figures from completed postprocessed products."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil
import pyarrow.parquet as pq
from pyproj import Transformer

from .scheme_a_visualization_data import BASE, BATCH, DOC, EDGES, PRIVATE, sha, write_json

BLUE, ORANGE, TEAL, GREY = "#0077BB", "#EE7733", "#009988", "#666666"


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def setup(root):
    os.environ["MPLCONFIGDIR"] = str(root / PRIVATE / "mplconfig")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        family = font_manager.FontProperties(fname=str(font)).get_name()
    else:
        family = "DejaVu Sans"
    plt.rcParams.update({"font.family": family, "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.unicode_minus": False, "figure.dpi": 300, "savefig.dpi": 300,
        "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    return plt, family


def save(fig, root, name):
    paths = []
    for extension in ("pdf", "png"):
        path = root / DOC / f"{name}.{extension}"
        fig.savefig(path, dpi=300, facecolor="white")
        paths.append(path.relative_to(root).as_posix())
    return paths


def d1(root, plt, summary):
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    scope = summary["spatial"]
    bounds = scope["wgs84_extent"]
    center = ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)
    zone = int((center[0] + 180) // 6) + 1
    epsg = (32600 if center[1] >= 0 else 32700) + zone
    transformer = Transformer.from_crs(4326, epsg, always_xy=True)
    coverage = read_csv(root / PRIVATE / "route_coverage_delta.csv")
    changed = [r for r in coverage if int(r["delta_orders"]) and r["geometry_available"] == "True"]
    needed = {r["geometry_edge_uid"] for r in changed}
    wgs = np.asarray([[bounds[0], bounds[1]], [bounds[0], bounds[3]],
                      [bounds[2], bounds[1]], [bounds[2], bounds[3]]])
    bx, by = transformer.transform(wgs[:, 0], wgs[:, 1])
    limits = [min(bx)/1000, min(by)/1000, max(bx)/1000, max(by)/1000]
    padx, pady = .035*(limits[2]-limits[0]), .035*(limits[3]-limits[1])
    limits = [limits[0]-padx, limits[1]-pady, limits[2]+padx, limits[3]+pady]
    fig = plt.figure(figsize=(6.9, 8.8))
    grid = fig.add_gridspec(2, 2, left=.09, right=.98, bottom=.15, top=.90, wspace=.24, hspace=.48)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
    for ax in axes:
        ax.set_xlim(limits[0], limits[2]); ax.set_ylim(limits[1], limits[3])
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_locator(MaxNLocator(3)); ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.set_xlabel("东向坐标（km）"); ax.set_ylabel("北向坐标（km）")
    selected_geometry = {}
    background_count = 0
    # Only a bounded batch of background geometry exists at once.
    for batch in pq.ParquetFile(root / EDGES).iter_batches(batch_size=2048, columns=["stage3_edge_uid", "geometry"]):
        lines = []
        for uid, encoded in zip(*batch.to_pydict().values()):
            line = np.asarray(json.loads(encoded), dtype=float)
            if len(line) < 2 or not np.isfinite(line).all():
                continue
            if uid not in needed and (line[:, 0].max() < bounds[0] or line[:, 0].min() > bounds[2]
                    or line[:, 1].max() < bounds[1] or line[:, 1].min() > bounds[3]):
                continue
            x, y = transformer.transform(line[:, 0], line[:, 1])
            projected = np.column_stack([x, y]) / 1000
            lines.append(projected)
            if uid in needed:
                selected_geometry[uid] = projected
        if lines:
            background_count += len(lines)
            for ax in axes:
                ax.add_collection(LineCollection(lines, colors="#D1D5D9", linewidths=.13,
                                                rasterized=True, zorder=1))
    positions = read_csv(root / PRIVATE / "deployment_delta.csv")
    delta = np.asarray([float(r["delta_hours"]) for r in positions])
    x, y = transformer.transform([float(r["lon_wgs84"]) for r in positions],
                                 [float(r["lat_wgs84"]) for r in positions])
    maximum = max(abs(delta))
    for positive in (True, False):
        mask = delta > 1e-9 if positive else delta < -1e-9
        axes[0].scatter(np.asarray(x)[mask]/1000, np.asarray(y)[mask]/1000,
            s=100*abs(delta[mask])/maximum, facecolors=BLUE if positive else "white",
            edgecolors=BLUE if positive else ORANGE, linewidths=.7, zorder=4)
    axes[0].legend(handles=[Line2D([], [], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor=BLUE, label="车时增加"),
        Line2D([], [], marker="o", color="none", markerfacecolor="white", markeredgecolor=ORANGE, label="车时减少")],
        loc="upper left", bbox_to_anchor=(0, -.16), ncol=2, frameon=False, handletextpad=.25, columnspacing=.6)
    axes[0].text(.02, .98, f"符号面积对应车时差\n最大绝对差 {maximum:.1f} h",
                 transform=axes[0].transAxes, va="top", fontsize=8,
                 bbox=dict(facecolor="white", edgecolor="none", alpha=.9))
    for sign in (-1, 1):
        for overlay in (False, True):
            rows = [r for r in changed if np.sign(int(r["delta_orders"])) == sign
                    and (r["reverse_geometry"] == "True") == overlay]
            lines = [selected_geometry[r["geometry_edge_uid"]][::-1] if overlay
                     else selected_geometry[r["geometry_edge_uid"]] for r in rows]
            if lines:
                axes[1].add_collection(LineCollection(lines, colors=BLUE if sign > 0 else ORANGE,
                    linewidths=[.16*abs(int(r["delta_orders"])) for r in rows],
                    linestyles="dotted" if overlay else "solid" if sign > 0 else "dashed", zorder=3+int(sign > 0)))
    axes[1].legend(handles=[Line2D([], [], color=BLUE, lw=1.2, label="C覆盖增加"),
        Line2D([], [], color=ORANGE, lw=1.2, ls="--", label="C覆盖减少")],
        loc="upper left", bbox_to_anchor=(0, -.16), ncol=2, frameon=False,
        handlelength=1.2, handletextpad=.25, columnspacing=.6)
    axes[1].text(.02, .98, "线宽：订单覆盖数差\n点线：历史反向身份",
                 transform=axes[1].transAxes, va="top", fontsize=8,
                 bbox=dict(facecolor="white", edgecolor="none", alpha=.9))
    changes = read_csv(root / PRIVATE / "service_change_origins.csv")
    for kind, color, marker, label in (("GAIN", BLUE, "o", "新增服务"), ("LOSS", ORANGE, "x", "失去服务")):
        rows = [r for r in changes if r["change_type"] == kind]
        x, y = transformer.transform([float(r["lon_wgs84"]) for r in rows],
                                     [float(r["lat_wgs84"]) for r in rows])
        axes[2].scatter(np.asarray(x)/1000, np.asarray(y)/1000,
            s=[16*int(r["order_count"]) for r in rows], c=color, marker=marker,
            linewidths=.8, alpha=.85, label=f"{label} {scope['service_changes'][kind]}单", zorder=4)
    axes[2].legend(loc="upper left", frameon=True, facecolor="white")
    axes[2].text(.98, .98, "全队净变化 +45单", ha="right", va="top", transform=axes[2].transAxes,
                 fontsize=9, bbox=dict(facecolor="white", edgecolor="none", alpha=.9))
    axes[0].set_title("A  上线资源配置差异", loc="left")
    axes[1].set_title("B  C服务订单的路段覆盖差异", loc="left")
    axes[2].set_title("C  全队新增与失去服务的请求起点", loc="left")
    for ax in axes:
        length = 2.0
        sx, sy = limits[0]+.07*(limits[2]-limits[0]), limits[1]+.07*(limits[3]-limits[1])
        ax.plot([sx, sx+length], [sy, sy], color="#333333", lw=1.5, zorder=5)
        ax.text(sx+length/2, sy+.02*(limits[3]-limits[1]), "2 km", ha="center", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.8))
    fig.suptitle("D1  部署变化与服务足迹：原10%参考", fontsize=12, y=.97)
    fig.text(.5, .935, "联合机会布局 - C兼容热点；20161031；同供给与共同在线控制", ha="center", fontsize=9)
    fig.text(.5, .028, "路线覆盖不是实测车流；上线位置差异不是调位轨迹。\n"
             f"共同人口28,367；位置缺失{scope['coordinates_missing']}；"
             f"覆盖几何缺失{scope['missing_coverage_geometry_keys']}。", ha="center", fontsize=8)
    paths = save(fig, root, "d1_network_deployment")
    plt.close(fig); gc.collect()
    return dict(files=paths, crs=f"EPSG:{epsg}", projected_extent_km=limits,
                background_directed_edges=background_count, change_identity_keys=len(changed),
                reference_condition=scope["condition"])


def d2(root, plt):
    from matplotlib.patches import Rectangle, Patch
    values = read_csv(root / DOC / "service_transitions.csv")
    keys = ("C", "HV", "UNSERVED")
    matrix = np.asarray([[next(int(r["orders"]) for r in values if r["condition"] == "Q20_REFERENCE_LABELS"
                              and r["before"] == a and r["after"] == b) for b in keys] for a in keys])
    fig, (ax, bar) = plt.subplots(1, 2, figsize=(6.9, 5.2), gridspec_kw={"width_ratios": [1, 1.1]})
    fig.subplots_adjust(left=.10, right=.97, bottom=.29, top=.79, wspace=.65)
    for i in range(3):
        for j in range(3):
            color = "#E6E8EB" if i == j else "#D2E4F1" if i < 2 and j < 2 else "#D1E7DF" if i == 2 else "#F4DDC9"
            ax.add_patch(Rectangle((j-.5, i-.5), 1, 1, facecolor=color, edgecolor="white", linewidth=2))
            ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center", fontsize=10)
    ax.set_xlim(-.5, 2.5); ax.set_ylim(2.5, -.5); ax.set_aspect("equal")
    ax.set_xticks(range(3), ["C", "HV", "未服务"])
    ax.set_yticks(range(3), ["C", "HV", "未服务"])
    ax.set_xlabel("联合布局的服务者"); ax.set_ylabel("兼容热点的服务者")
    ax.set_title("A  同一订单的归属转换（20%）", loc="left")
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.text(.5, -.36, "新增113 / 失去128 / 全队净-15\nC净增加165；总人口28,367",
            transform=ax.transAxes, ha="center", fontsize=9)
    rows = read_csv(root / DOC / "loaded_decomposition.csv")
    for y, row in enumerate(rows):
        count, intensity, net = (float(row[k]) for k in ("working_count_h", "conditional_intensity_h", "net_loaded_h"))
        bar.barh(y, count, color=BLUE, height=.45)
        bar.barh(y, intensity, left=max(count, 0), color=ORANGE, height=.45, hatch="///", edgecolor="white")
        bar.plot(net, y, "D", color="black", markersize=4, zorder=4)
        bar.annotate(f"{net:+.2f}", (max(count, 0)+intensity, y), xytext=(5, 0),
                     textcoords="offset points", va="center", fontsize=9)
    bar.axvline(0, color=GREY, lw=.7)
    bar.set_yticks(range(4), [r["label"] for r in rows]); bar.invert_yaxis()
    bar.set_xlim(-2, 29); bar.set_xlabel("C窗口内载客增量（h）")
    bar.set_title("B  载客增量的账面来源", loc="left")
    bar.legend(handles=[Patch(facecolor=BLUE, label="获服务窗口数量项"),
        Patch(facecolor=ORANGE, hatch="///", label="条件载客强度项")],
        loc="upper left", bbox_to_anchor=(-.05, -.18), frameon=False)
    fig.suptitle("D2  服务重新分工与资源转换", fontsize=12, y=.97)
    fig.text(.5, .905, "订单换手不等于新增服务；参与数量与条件强度不等于模块因果贡献。", ha="center", fontsize=9)
    fig.text(.5, .07, "灰：归属不变；蓝：C/HV换手；绿：新增服务；橙：失去服务。\n"
             "黑色菱形为净载客增量；5%的负数量项保留。", ha="center", fontsize=8)
    paths = save(fig, root, "d2_service_conversion")
    plt.close(fig)
    return dict(files=paths, matrix_population=int(matrix.sum()), no_causal_component_claim=True)


def d3(root, plt):
    rows = read_csv(root / DOC / "effects.csv")
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 3.8), sharey=True,
                            gridspec_kw={"width_ratios": [1.35, 1, 1]})
    fig.subplots_adjust(left=.17, right=.98, bottom=.27, top=.77, wspace=.45)
    metrics = ("C_loaded_utilization_pp", "completed_orders", "total_empty_km")
    titles = ("A  C载客利用率", "B  全队完成量", "C  总空驶")
    units = ("增量（百分点）", "增量（单）", "增量（km）")
    limits = ((-.05, 1.45), (-35, 75), (-520, 250))
    for ax, metric, title, unit, limit in zip(axes, metrics, titles, units, limits):
        ax.axvline(0, color=GREY, lw=.8)
        for y, row in enumerate(rows):
            value = float(row[metric])
            color = BLUE if value >= 0 else ORANGE
            ax.plot([0, value], [y, y], color="#B5BEC6", lw=1)
            ax.plot(value, y, "D" if row["existing_reference"] == "True" else "o",
                    color=color, markersize=5, markeredgecolor="white", markeredgewidth=.5)
            label = f"{value:+.3f}" if metric == metrics[0] else f"{value:+.0f}" if metric == metrics[1] else f"{value:+.1f}"
            ax.annotate(label, (value, y), xytext=(0, 8),
                        textcoords="offset points", va="bottom", ha="center", fontsize=9)
        ax.set_xlim(*limit); ax.set_xlabel(unit); ax.set_title(title, loc="left")
    axes[0].set_yticks(range(4), [r["label"] for r in rows])
    axes[0].set_ylim(3.6, -.6)
    fig.suptitle("D3  C利用改善并不意味着全部运营指标占优", fontsize=12, y=.96)
    fig.text(.5, .865, "联合机会布局 - C兼容热点；四个固定对照点", ha="center", fontsize=9)
    fig.text(.5, .10, "菱形：已有原10%参考；圆点：新增条件。\n"
             "空驶负值表示减少；单日条件性结果，无独立重复置信区间。", ha="center", fontsize=8)
    paths = save(fig, root, "d3_operational_tradeoffs")
    plt.close(fig)
    return dict(files=paths, all_four_conditions=True, fabricated_error_bars=False)


def run(root, replace=False):
    started = perf_counter()
    root = Path(root).resolve()
    if not replace and any((root / DOC / f"{stem}.{suffix}").exists()
            for stem in ("d1_network_deployment", "d2_service_conversion", "d3_operational_tradeoffs")
            for suffix in ("png", "pdf")):
        raise ValueError("rendered figures exist; use --replace only for authorized figure refinement")
    summary = json.loads((root / DOC / "data_summary.json").read_text(encoding="utf-8"))
    plt, family = setup(root)
    outputs = {}
    for name, callback in (("D1", lambda: d1(root, plt, summary)),
                           ("D2", lambda: d2(root, plt)), ("D3", lambda: d3(root, plt))):
        outputs[name] = callback()
        gc.collect()
    memory = psutil.Process().memory_info()
    result = dict(status="D1_D2_D3_RENDERED_PENDING_VISUAL_CHECK", figures=outputs,
        font_family=family, renderer_sha256=sha(__file__),
        runtime_s=perf_counter()-started, peak_rss_mib=max(memory.rss, getattr(memory, "peak_wset", memory.rss))/2**20,
        new_experiment=False, external_map_requests=False, gpu_used=False, coordinates_in_public_tables=False)
    write_json(root / DOC / "render_summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.replace), ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
