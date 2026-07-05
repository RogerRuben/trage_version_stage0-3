"""Generate representative full-day Stage0 case maps."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer

from run_full_day_2017 import gcj02_to_wgs84, order_bucket


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roads", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--buckets", type=int, default=128)
    return parser.parse_args()


def choose_cases(orders: pd.DataFrame):
    substantial = orders[(orders.clean_point_count >= 20) & (orders.distance_m >= 500)]
    good = substantial[substantial.quality_flag.eq("good")]
    failed = substantial[~substantial.matching_success]
    jump = orders[orders.jump_removed_count > 0]
    illustrative_good = good[
        good.distance_m.ge(3000) & good.clean_point_count.between(100, 800) & good.turn_count.ge(3)
    ]
    complex_pool = good[
        good.duration_s.between(300, 3600) & good.turn_count.ge(5) & good.matching_confidence.ge(0.7)
    ]
    choices = [
        ("high_quality", illustrative_good.sort_values(["matching_confidence", "distance_m"], ascending=False).iloc[0]),
        ("matching_failure", (failed if len(failed) else substantial).sort_values("matching_confidence").iloc[0]),
        ("stop_and_go", good[good.distance_m >= 1000].sort_values("stop_count_km", ascending=False).iloc[0]),
        ("complex_intersection", complex_pool.sort_values("intersection_delay_s", ascending=False).iloc[0]),
        ("gps_jump", (jump if len(jump) else orders).sort_values(["jump_removed_count", "max_speed_kmh"], ascending=False).iloc[0]),
    ]
    used = set()
    result = []
    for label, row in choices:
        if row.order_id not in used:
            used.add(row.order_id)
            result.append((label, row))
    return result


def main():
    args = parse_args()
    figures = args.output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    orders = pd.read_parquet(args.output_dir / "full_day_stage0_orders.parquet")
    roads = gpd.read_parquet(args.roads).to_crs(32649)
    transformer = Transformer.from_crs(4326, 32649, always_xy=True)
    index_rows = []
    for label, order in choose_cases(orders):
        oid = str(order.order_id)
        bucket = int(order_bucket(pd.Series([oid]), args.buckets)[0])
        matched = pd.read_parquet(
            args.output_dir / "matched_points" / f"part_{bucket:03d}.parquet",
            filters=[[('order_id', '==', oid)]],
        ).sort_values("timestamp")
        raw = pd.read_parquet(
            args.output_dir / "order_buckets" / f"bucket_{bucket:03d}.parquet",
            filters=[[('order_id', '==', oid)]],
        ).sort_values("timestamp")
        lon, lat = gcj02_to_wgs84(raw.lon.to_numpy(), raw.lat.to_numpy())
        raw_x, raw_y = transformer.transform(lon, lat)
        clean_x, clean_y = transformer.transform(matched.lon.to_numpy(), matched.lat.to_numpy())
        clean_rows = set(matched.source_row.astype(int))
        removed = ~raw.source_row.astype(int).isin(clean_rows)
        xmin, xmax = min(raw_x.min(), matched.snap_x.min()) - 250, max(raw_x.max(), matched.snap_x.max()) + 250
        ymin, ymax = min(raw_y.min(), matched.snap_y.min()) - 250, max(raw_y.max(), matched.snap_y.max()) + 250
        nearby = roads.cx[xmin:xmax, ymin:ymax]
        fig, ax = plt.subplots(figsize=(8, 8))
        nearby.plot(ax=ax, color="#c8c8c8", linewidth=0.65, zorder=1)
        ax.plot(raw_x, raw_y, color="#d95f02", linewidth=1, alpha=0.55, label="raw GPS", zorder=2)
        ax.plot(clean_x, clean_y, color="#1f78b4", linewidth=1.3, label="clean GPS", zorder=3)
        ax.plot(matched.snap_x, matched.snap_y, color="#1b9e77", linewidth=2, alpha=0.8, label="matched", zorder=4)
        if removed.any():
            ax.scatter(raw_x[removed], raw_y[removed], marker="x", s=32, color="#e7298a", label="removed", zorder=5)
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{label}: {oid[:12]}…")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        filename = f"full_day_case_{label}.png"
        fig.savefig(figures / filename, dpi=180)
        plt.close(fig)
        index_rows.append({
            "case_type": label, "order_id": oid, "bucket": bucket,
            "matching_confidence": order.matching_confidence,
            "matching_success": order.matching_success,
            "figure": f"figures/{filename}",
        })
    pd.DataFrame(index_rows).to_csv(args.output_dir / "full_day_visual_case_index.csv", index=False)
    print(pd.DataFrame(index_rows).to_string(index=False))


if __name__ == "__main__":
    main()
