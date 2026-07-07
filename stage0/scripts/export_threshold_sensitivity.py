"""Persist point-level threshold sensitivity before matched-point pruning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--date", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    low = {3: [0.0, 0.0], 5: [0.0, 0.0], 8: [0.0, 0.0]}
    intersection = {20: [0.0, 0.0], 30: [0.0, 0.0], 50: [0.0, 0.0]}
    stops = {3: [0, 0.0], 5: [0, 0.0], 10: [0, 0.0]}
    for path in sorted(args.matched_dir.glob("*.parquet")):
        frame = pd.read_parquet(path, columns=["order_id", "dt_s", "speed_kmh", "intersection_distance_m"])
        dt = frame.dt_s.fillna(0).clip(0, 120)
        for threshold in low:
            low[threshold][0] += float(dt[frame.speed_kmh < threshold].sum()); low[threshold][1] += float(dt.sum())
        for radius in intersection:
            intersection[radius][0] += float(dt[(frame.intersection_distance_m <= radius) & (frame.speed_kmh < 5)].sum())
            intersection[radius][1] += float(dt.sum())
        stopped = frame.speed_kmh.lt(2); same = frame.order_id.eq(frame.order_id.shift()).fillna(False)
        run_id = (stopped & (~stopped.shift(fill_value=False) | ~same)).cumsum()
        durations = dt[stopped].groupby(run_id[stopped]).sum()
        for threshold in stops:
            valid = durations[durations >= threshold]
            stops[threshold][0] += int(len(valid)); stops[threshold][1] += float(valid.sum())
    result = {
        "date": args.date,
        "low_speed_threshold": {str(k): v[0] / v[1] if v[1] else None for k, v in low.items()},
        "intersection_buffer": {str(k): v[0] / v[1] if v[1] else None for k, v in intersection.items()},
        "stop_duration": {str(k): {"count": v[0], "seconds": v[1]} for k, v in stops.items()},
    }
    target = args.output_root / "reports" / "threshold_sensitivity" / f"day={args.date}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

