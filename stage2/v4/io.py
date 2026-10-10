"""Small deterministic I/O helpers for Stage 2 v4."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.tmp-",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_parquet(
    frame: "pd.DataFrame",
    path: str | Path,
    *,
    compression: str = "zstd",
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.tmp-",
        suffix=".parquet",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False, compression=compression)
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def stage2_v4_code_identity(relative_files: Iterable[str] | None = None) -> str:
    repository = Path(__file__).resolve().parents[2]
    if relative_files is None:
        files = sorted(
            repository.joinpath("stage2", "v4").rglob("*.py"),
            key=lambda path: path.as_posix(),
        )
    else:
        files = sorted(
            (repository / path for path in relative_files),
            key=lambda path: path.as_posix(),
        )
        missing = [path for path in files if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Stage 2 v4 identity files are missing: {missing}")
    records = [
        {
            "path": path.relative_to(repository).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in files
        if "__pycache__" not in path.parts
    ]
    return f"stage2-v4-content.{sha256_bytes(canonical_json_bytes(records))}"
