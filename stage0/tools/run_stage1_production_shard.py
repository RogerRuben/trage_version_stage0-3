"""Run one date-disjoint Stage 1 production shard.

This orchestration helper intentionally lives outside ``stage0.v6`` so that
parallel scheduling does not change the frozen matcher code identity recorded
in bucket manifests.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from stage0.v6.config import Stage0V6Config, load_config
from stage0.v6.stage1_production import build_stage1_input


class ShardConfig:
    """Config view with a restricted schedule and the original frozen digest."""

    def __init__(
        self,
        base: Stage0V6Config,
        data: dict[str, Any],
    ) -> None:
        self.data = data
        self.source = base.source
        self.repo_root = base.repo_root
        self._digest = base.digest

    def section(self, name: str) -> dict[str, Any]:
        value = self.data.get(name)
        if not isinstance(value, dict):
            raise KeyError(f"configuration section is missing or invalid: {name}")
        return value

    def path(self, name: str) -> Path:
        value = self.section("paths").get(name)
        if value is None:
            raise KeyError(f"configuration path is missing: {name}")
        path = Path(str(value))
        return (
            path
            if path.is_absolute()
            else (self.repo_root / path).resolve()
        )

    @property
    def digest(self) -> str:
        return self._digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--test-owner", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be in [0, --shard-count)")
    if not 0 <= args.test_owner < args.shard_count:
        parser.error("--test-owner must be in [0, --shard-count)")

    base = load_config(args.config)
    data = copy.deepcopy(base.data)
    production = data["production"]
    train_dates = list(production["train_dates"])
    validation_dates = list(production["validation_dates"])
    production["train_dates"] = [
        date
        for index, date in enumerate(train_dates)
        if index % args.shard_count == args.shard_index
    ]
    production["validation_dates"] = [
        date
        for index, date in enumerate(validation_dates)
        if index % args.shard_count == args.shard_index
    ]
    if args.shard_index != args.test_owner:
        owned_dates = (
            production["train_dates"] + production["validation_dates"]
        )
        production["test_date"] = owned_dates[0]
        production["test_target"] = 0

    result = build_stage1_input(
        ShardConfig(base, data),
        resume=True,
    )
    result["shard_index"] = args.shard_index
    result["shard_count"] = args.shard_count
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
