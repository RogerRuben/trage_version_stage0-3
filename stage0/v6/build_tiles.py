"""Reproducible Valhalla config/tile build and manifest recording."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def environment_info() -> dict[str, Any]:
    import valhalla

    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "pyvalhalla_version": metadata.version("pyvalhalla"),
        "valhalla_version": getattr(valhalla, "__version__", metadata.version("pyvalhalla")),
        "operating_system": platform.platform(),
        "installation_method": "conda Python 3.12 environment + PyPI wheel",
        "environment_name": "stage0-valhalla",
    }


def write_manifest(
    *,
    pbf: Path,
    config: Path,
    tiles: Path,
    started_at: datetime,
    ended_at: datetime,
    status: str,
    adopted_existing: bool,
    error: str | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": "stage0_v6_valhalla_tile_build.1",
        "pbf_path": str(pbf.resolve()),
        "pbf_size_bytes": pbf.stat().st_size,
        "pbf_sha256": sha256_file(pbf),
        **environment_info(),
        "config_path": str(config.resolve()),
        "tile_path": str(tiles.resolve()),
        "build_started_at": started_at.astimezone(timezone.utc).isoformat(),
        "build_ended_at": ended_at.astimezone(timezone.utc).isoformat(),
        "build_elapsed_s": (ended_at - started_at).total_seconds(),
        "tile_size_bytes": directory_size(tiles),
        "build_status": status,
        "adopted_existing_build": adopted_existing,
        "error": error,
    }
    target = config.parent / "build_manifest.json"
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def create_config(config: Path, tiles: Path) -> None:
    from valhalla import get_config

    tiles.mkdir(parents=True, exist_ok=True)
    payload = get_config("", tiles)
    payload["mjolnir"].pop("tile_extract", None)
    payload["mjolnir"].pop("traffic_extract", None)
    payload["mjolnir"]["keep_all_osm_node_ids"] = True
    payload["mjolnir"]["keep_osm_node_ids"] = True
    payload["meili"]["default"]["max_search_radius"] = 200
    payload["service_limits"]["trace"]["max_search_radius"] = 200.0
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build(pbf: Path, config: Path, tiles: Path, *, force: bool = False) -> dict[str, Any]:
    existing_tiles = list(tiles.rglob("*.gph")) if tiles.exists() else []
    if existing_tiles and not force:
        raise FileExistsError(
            f"{len(existing_tiles)} graph tiles already exist; refusing to rebuild without --force"
        )
    create_config(config, tiles)
    started = datetime.now(timezone.utc)
    error = None
    status = "FAIL"
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "valhalla",
                "valhalla_build_tiles",
                "-c",
                str(config),
                str(pbf),
            ],
            check=True,
        )
        status = "PASS"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        ended = datetime.now(timezone.utc)
        manifest = write_manifest(
            pbf=pbf,
            config=config,
            tiles=tiles,
            started_at=started,
            ended_at=ended,
            status=status,
            adopted_existing=False,
            error=error,
        )
    return manifest


def record_existing(pbf: Path, config: Path, tiles: Path) -> dict[str, Any]:
    graph_tiles = list(tiles.rglob("*.gph"))
    if not graph_tiles:
        raise FileNotFoundError(f"no Valhalla .gph tiles found under {tiles}")
    timestamps = [item.stat().st_mtime for item in graph_tiles]
    started = datetime.fromtimestamp(min(timestamps), tz=timezone.utc)
    ended = datetime.fromtimestamp(max(timestamps), tz=timezone.utc)
    return write_manifest(
        pbf=pbf,
        config=config,
        tiles=tiles,
        started_at=started,
        ended_at=ended,
        status="PASS",
        adopted_existing=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pbf", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tiles", type=Path, required=True)
    parser.add_argument("--record-existing", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    payload = (
        record_existing(args.pbf, args.config, args.tiles)
        if args.record_existing
        else build(args.pbf, args.config, args.tiles, force=args.force)
    )
    payload["command_wall_s"] = time.perf_counter() - started
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
