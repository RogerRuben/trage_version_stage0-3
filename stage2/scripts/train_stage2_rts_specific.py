"""Run the RTS-specific Stage2 branch.

RTS is treated separately because it is more route-propagation-like and showed
the strongest temporal drift. This wrapper starts with the controlled strong
tabular RTS baseline; route/graph RTS variants should be compared against it
with the same order budgets.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("stage2/output/link_dataset"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/deep_baselines/rts_specific"))
    parser.add_argument("--prediction-root", type=Path, default=Path("stage2/output/deep_predictions_rts"))
    parser.add_argument("--max-train-orders", type=int, default=None)
    parser.add_argument("--max-train-rows", default="3000000")
    parser.add_argument("--profile-scope", choices=["all_train", "sampled_train_orders"], default="all_train")
    parser.add_argument("--tail-weight", type=float, default=5.0)
    parser.add_argument("--num-boost-round", type=int, default=800)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        sys.executable,
        "stage2/scripts/train_stage2_full_tabular.py",
        "--dataset-root", str(args.dataset_root),
        "--output-root", str(args.output_root / "full_tabular_rts"),
        "--prediction-root", str(args.prediction_root),
        "--max-train-rows", str(args.max_train_rows),
        "--profile-scope", args.profile_scope,
        "--tail-weight", str(args.tail_weight),
        "--num-boost-round", str(args.num_boost_round),
        "--targets", "RTS",
        "--seed", str(args.seed),
    ]
    if args.max_train_orders is not None:
        cmd.extend(["--max-train-orders", str(args.max_train_orders)])
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
