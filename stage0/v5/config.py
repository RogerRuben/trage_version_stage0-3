"""Central configuration and reproducibility helpers."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(*values: object, seed: int = 0) -> int:
    payload = "\x1f".join([str(seed), *(str(value) for value in values)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_sha(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


@dataclass(frozen=True)
class Stage0Config:
    source: Path
    values: dict[str, Any]
    digest: str

    @classmethod
    def load(cls, path: Path) -> "Stage0Config":
        values = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(path.resolve(), values, config_hash(values))

    def section(self, name: str) -> dict[str, Any]:
        value = self.values.get(name)
        if not isinstance(value, dict):
            raise KeyError(f"missing mapping section: {name}")
        return value

    def path(self, name: str, repo: Path) -> Path:
        raw = Path(self.section("paths")[name])
        return raw if raw.is_absolute() else (repo / raw).resolve()
