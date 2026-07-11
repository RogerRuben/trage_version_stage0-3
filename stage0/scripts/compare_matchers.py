"""Create daily geometric-vs-production-matcher quality tables and Markdown reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order-base", type=Path, required=True)
    parser.add_argument("--hmm-quality-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("stage0_output"))
    parser.add_argument("--date", required=True)
    parser.add_argument("--geometric-matched-dir", type=Path)
    parser.add_argument("--hmm-matched-dir", type=Path)
    parser.add_argument("--roads", type=Path)
    parser.add_argument("--comparison-collection", default="matcher_comparison")
    parser.add_argument("--matcher-label", default="local_topology_fmm")
    parser.add_argument("--quality-report-collection", default="hmm_quality_reports")
    parser.add_argument("--manifest-glob", default="day={date}.hmm.worker=*.json")
    return parser.parse_args()


def q(series: pd.Series, p: float) -> float:
    return float(series.replace([np.inf, -np.inf], np.nan).dropna().quantile(p))


def plot_cases(comparison: pd.DataFrame, args: argparse.Namespace, out_dir: Path) -> None:
    if not (args.geometric_matched_dir and args.hmm_matched_dir and args.roads):
        return
    candidates = []
    improvement = comparison.assign(improvement=comparison.geo_topology_gap_count - comparison.topology_gap_count)
    if len(improvement): candidates.append(("topology_improvement", improvement.sort_values("improvement", ascending=False).iloc[0]))
    changed = comparison[~comparison.fallback_used]
    if len(changed): candidates.append(("sequence_change", changed.sort_values("link_sequence_change_ratio", ascending=False).iloc[0]))
    fallback = comparison[comparison.fallback_used]
    if len(fallback): candidates.append(("fallback", fallback.iloc[0]))
    roads = gpd.read_parquet(args.roads).to_crs(32649)
    figures = out_dir / "figures"; figures.mkdir(parents=True, exist_ok=True)
    used = set()
    for label, row in candidates:
        order_id = str(row.order_id)
        if order_id in used: continue
        used.add(order_id); part = str(row.part)
        geo_path = next(iter(args.geometric_matched_dir.glob(f"*{part}.parquet")))
        hmm_path = next(iter(args.hmm_matched_dir.glob(f"*{part}.parquet")))
        geo = pd.read_parquet(geo_path, filters=[[('order_id', '==', order_id)]]).sort_values("timestamp")
        hmm = pd.read_parquet(hmm_path, filters=[[('order_id', '==', order_id)]]).sort_values("timestamp")
        if geo.empty or hmm.empty: continue
        xmin = min(geo.snap_x.min(), hmm.proj_x_hmm.min()) - 200; xmax = max(geo.snap_x.max(), hmm.proj_x_hmm.max()) + 200
        ymin = min(geo.snap_y.min(), hmm.proj_y_hmm.min()) - 200; ymax = max(geo.snap_y.max(), hmm.proj_y_hmm.max()) + 200
        nearby = roads.cx[xmin:xmax, ymin:ymax]
        fig, ax = plt.subplots(figsize=(8, 8))
        nearby.plot(ax=ax, color="#d0d0d0", linewidth=0.7)
        ax.plot(geo.snap_x, geo.snap_y, color="#d95f02", linewidth=1.5, label="geometric")
        ax.plot(hmm.proj_x_hmm, hmm.proj_y_hmm, color="#1b9e77", linewidth=1.5, label=args.matcher_label)
        ax.scatter(hmm.proj_x_hmm, hmm.proj_y_hmm, s=3, color="#1b9e77")
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_aspect("equal"); ax.axis("off"); ax.legend()
        ax.set_title(f"{label}: {order_id[:12]}…")
        fig.tight_layout(); fig.savefig(figures / f"matcher_case_{label}.png", dpi=180); plt.close(fig)


def main() -> None:
    args = parse_args()
    files = sorted(args.hmm_quality_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no matcher quality partitions in {args.hmm_quality_dir}")
    quality_frames = []
    for path in files:
        frame = pd.read_parquet(path)
        frame["part"] = path.stem.split("=")[-1].split("_")[-1]
        quality_frames.append(frame)
    hmm = pd.concat(quality_frames, ignore_index=True)
    geo = pd.read_parquet(args.order_base)[[
        "order_id", "p90_gps_to_link_dist_m", "route_length_ratio", "topology_gap_count",
        "matching_confidence", "matching_success",
    ]].rename(columns={
        "p90_gps_to_link_dist_m": "geo_p90_match_dist", "route_length_ratio": "geo_route_length_ratio",
        "topology_gap_count": "geo_topology_gap_count", "matching_confidence": "geo_match_confidence",
        "matching_success": "geo_matching_success",
    })
    comparison = hmm.merge(geo, on="order_id", how="left")
    comparison["matcher_matching_success"] = (
        comparison.matched_fraction.ge(0.85) & comparison.p90_match_dist.le(50)
        & comparison.route_length_ratio.between(0.8, 1.3) & comparison.topology_gap_count.le(1)
    )
    comparison["hmm_matching_success"] = comparison["matcher_matching_success"]
    out_dir = args.output_root / args.comparison_collection / f"day={args.date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_parquet(out_dir / "order_comparison.parquet", index=False, compression="zstd")
    summary = {
        "date": args.date, "orders": int(len(comparison)),
        "geometric_p90_distance_distribution": {
            "p50": q(comparison.geo_p90_match_dist, 0.5), "p90": q(comparison.geo_p90_match_dist, 0.9),
            "p95": q(comparison.geo_p90_match_dist, 0.95),
        },
        "matcher_p90_distance_distribution": {
            "p50": q(comparison.p90_match_dist, 0.5), "p90": q(comparison.p90_match_dist, 0.9),
            "p95": q(comparison.p90_match_dist, 0.95),
        },
        "geometric_route_ratio_good": float(comparison.geo_route_length_ratio.between(0.8, 1.3).mean()),
        "matcher_route_ratio_good": float(comparison.route_length_ratio.between(0.8, 1.3).mean()),
        "geometric_topology_gaps": int(comparison.geo_topology_gap_count.fillna(0).sum()),
        "matcher_topology_gaps": int(comparison.topology_gap_count.fillna(0).sum()),
        "geometric_failure_ratio": float((~comparison.geo_matching_success.fillna(False)).mean()),
        "matcher_failure_ratio": float((~comparison.matcher_matching_success).mean()),
        "fallback_ratio": float(comparison.fallback_used.mean()),
        "mean_link_sequence_change_ratio": float(comparison.link_sequence_change_ratio.mean()),
    }
    manifest_dir = args.output_root / "manifests"
    worker_seconds = []
    manifest_pattern = args.manifest_glob.format(date=args.date)
    for path in manifest_dir.glob(manifest_pattern):
        worker_seconds.append(float(json.loads(path.read_text(encoding="utf-8")).get("seconds", 0)))
    run_state = manifest_dir / f"day={args.date}.run_state.json"
    summary["matcher_runtime_seconds"] = max(worker_seconds) if worker_seconds else None
    summary["geometric_runtime_seconds"] = (
        float(json.loads(run_state.read_text(encoding="utf-8")).get("runtime_seconds"))
        if run_state.exists() and json.loads(run_state.read_text(encoding="utf-8")).get("runtime_seconds") is not None
        else None
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = f"""# Matcher comparison: {args.date}

| Metric | Geometric/topology | {args.matcher_label} |
|---|---:|---:|
| Order P90 distance P50 | {summary['geometric_p90_distance_distribution']['p50']:.2f} m | {summary['matcher_p90_distance_distribution']['p50']:.2f} m |
| Order P90 distance P90 | {summary['geometric_p90_distance_distribution']['p90']:.2f} m | {summary['matcher_p90_distance_distribution']['p90']:.2f} m |
| Order P90 distance P95 | {summary['geometric_p90_distance_distribution']['p95']:.2f} m | {summary['matcher_p90_distance_distribution']['p95']:.2f} m |
| Route ratio in 0.8–1.3 | {summary['geometric_route_ratio_good']:.2%} | {summary['matcher_route_ratio_good']:.2%} |
| Topology gaps | {summary['geometric_topology_gaps']:,} | {summary['matcher_topology_gaps']:,} |
| Failure ratio | {summary['geometric_failure_ratio']:.2%} | {summary['matcher_failure_ratio']:.2%} |
| Matcher fallback ratio | — | {summary['fallback_ratio']:.2%} |
| Runtime | {summary['geometric_runtime_seconds'] if summary['geometric_runtime_seconds'] is not None else 'not recorded'} s | {summary['matcher_runtime_seconds'] if summary['matcher_runtime_seconds'] is not None else 'not recorded'} s |

Mean point-level link-sequence change relative to the geometric matcher: {summary['mean_link_sequence_change_ratio']:.2%}.
"""
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    quality_report = args.output_root / args.quality_report_collection / f"day={args.date}.md"
    quality_report.parent.mkdir(parents=True, exist_ok=True)
    quality_report.write_text(report, encoding="utf-8")
    plot_cases(comparison, args, out_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
