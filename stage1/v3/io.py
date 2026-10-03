"""Content identity and atomic-output helpers for Stage 1 v3."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import pandas as pd
import pyarrow.parquet as pq

from .schema import ALL_INPUT_PRODUCTS, ContractError


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
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def combined_file_sha256(files: Mapping[str, str | Path]) -> str:
    records = [
        {"name": str(name), "sha256": sha256_file(path)}
        for name, path in sorted(files.items())
    ]
    return sha256_bytes(canonical_json_bytes(records))


def stage1_v3_code_identity() -> str:
    """Hash the executable v3 source tree instead of trusting a CLI label."""

    repository = Path(__file__).resolve().parents[2]
    stage1_root = repository / "stage1"
    files = sorted(
        [
            *stage1_root.joinpath("v3").glob("*.py"),
            stage1_root / "scripts" / "build_stage1_labels_v3.py",
        ],
        key=lambda path: path.as_posix(),
    )
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise ContractError(f"Stage1 v3 source files are missing: {missing}")
    records = [
        {
            "path": path.relative_to(repository).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    return f"stage1-v3-content.{sha256_bytes(canonical_json_bytes(records))}"


def parquet_schema_sha256(path: str | Path) -> str:
    schema = pq.read_schema(Path(path))
    fields = [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": bool(field.nullable),
        }
        for field in schema
    ]
    return sha256_bytes(canonical_json_bytes(fields))


def parquet_row_count(path: str | Path) -> int:
    metadata = pq.ParquetFile(Path(path)).metadata
    return int(metadata.num_rows)


def parquet_column_names(path: str | Path) -> tuple[str, ...]:
    return tuple(pq.read_schema(Path(path)).names)


def bucket_input_identity(bucket_path: str | Path) -> dict[str, Any]:
    root = Path(bucket_path)
    files = {"manifest.json": root / "manifest.json"}
    files.update(
        {
            f"{product}.parquet": root / f"{product}.parquet"
            for product in ALL_INPUT_PRODUCTS
        }
    )
    missing = sorted(name for name, path in files.items() if not path.is_file())
    if missing:
        raise ContractError(f"cannot hash incomplete Stage0 bucket: {missing}")
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot inspect Stage0 bucket manifest: {root}") from exc
    counts = manifest.get("product_row_counts") if isinstance(manifest, dict) else None
    if not isinstance(counts, dict):
        raise ContractError(f"Stage0 bucket has no product row counts: {root}")
    for product in ALL_INPUT_PRODUCTS:
        expected = counts.get(product)
        if (
            isinstance(expected, bool)
            or not isinstance(expected, int)
            or expected < 0
        ):
            raise ContractError(
                f"Stage0 bucket has invalid row count for {product}: {root}"
            )
        actual = parquet_row_count(root / f"{product}.parquet")
        if actual != expected:
            raise ContractError(
                f"Stage0 bucket row count mismatch for {product}: "
                f"manifest={expected}, parquet={actual}"
            )
    file_hashes = {
        name: sha256_file(path)
        for name, path in sorted(files.items())
    }
    schema_hashes = {
        product: parquet_schema_sha256(root / f"{product}.parquet")
        for product in ALL_INPUT_PRODUCTS
    }
    return {
        "files": file_hashes,
        "schemas": schema_hashes,
        "product_row_counts": {
            product: int(counts[product]) for product in ALL_INPUT_PRODUCTS
        },
        "bucket_sha": sha256_bytes(canonical_json_bytes(file_hashes)),
    }


def frame_schema_sha256(frame: pd.DataFrame) -> str:
    schema = [
        {"name": str(name), "dtype": str(dtype)}
        for name, dtype in zip(frame.columns, frame.dtypes)
    ]
    return sha256_bytes(canonical_json_bytes(schema))


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


def atomic_write_parquet(frame: pd.DataFrame, path: str | Path) -> None:
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
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


@contextmanager
def atomic_output_directory(target: str | Path) -> Iterator[Path]:
    """Yield a sibling temporary directory and publish it only when complete."""

    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ContractError(
            f"refusing to overwrite an existing Stage1 v3 output: {destination}"
        )
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
    )
    try:
        yield temporary
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            if (
                temporary.parent.resolve() != destination.parent.resolve()
                or not temporary.name.startswith(f".{destination.name}.tmp-")
            ):
                raise ContractError(
                    f"refusing to clean an unexpected temporary path: {temporary}"
                )
            shutil.rmtree(temporary)
        raise
