"""Configuration loading for the Stage 0 v6 prototype."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Stage0V6Config:
    """Validated configuration plus repository-relative path resolution."""

    data: dict[str, Any]
    source: Path
    repo_root: Path

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
        return path if path.is_absolute() else (self.repo_root / path).resolve()

    @property
    def digest(self) -> str:
        payload = json.dumps(self.data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_config(path: str | Path) -> Stage0V6Config:
    source = Path(path).resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Stage 0 v6 configuration must be a mapping")
    if raw.get("schema_version") != "stage0_v6_valhalla.1":
        raise ValueError(f"unsupported schema_version: {raw.get('schema_version')!r}")
    repo_root = source.parents[2]
    return Stage0V6Config(raw, source, repo_root)
