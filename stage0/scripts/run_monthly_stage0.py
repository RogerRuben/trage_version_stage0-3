"""Resume-safe daily extraction and geometric Stage0 orchestration for the monthly RAR."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--roads", type=Path, required=True)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("stage0_output"))
    parser.add_argument("--unrar", type=Path, default=Path(r"C:\Program Files\WinRAR\UnRAR.exe"))
    parser.add_argument("--start-date", default="20161001")
    parser.add_argument("--end-date", default="20161031")
    parser.add_argument("--buckets", type=int, default=128)
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def archive_days(archive: Path) -> list[tuple[str, str]]:
    listing = subprocess.check_output(["tar", "-tf", str(archive)], text=True, encoding="utf-8", errors="replace")
    rows: list[tuple[str, str]] = []
    for line in listing.splitlines():
        match = re.search(r"(?:^|/)10-(\d{1,2})/([^/]+\.tar\.gz)$", line)
        if match:
            date = f"201610{int(match.group(1)):02d}"
            rows.append((date, line.strip()))
    return sorted(set(rows))


def run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + subprocess.list2cmdline(command) + "\n")
        log.flush()
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output_root.resolve()
    entries = [row for row in archive_days(args.archive) if args.start_date <= row[0] <= args.end_date]
    if not entries:
        raise ValueError("no daily archives in requested date range")
    inventory = {"archive": str(args.archive.resolve()), "days": [{"date": d, "member": m} for d, m in entries]}
    save_json(output / "manifests" / "archive_inventory.json", inventory)
    if args.dry_run:
        print(json.dumps(inventory, ensure_ascii=False, indent=2)); return
    if not args.unrar.exists():
        raise FileNotFoundError(f"UnRAR not found: {args.unrar}")

    for date, member in entries:
        standard_manifest = output / "manifests" / f"day={date}.standardize.json"
        if standard_manifest.exists() and json.loads(standard_manifest.read_text(encoding="utf-8")).get("complete"):
            print(f"day={date} already complete", flush=True); continue
        day_started = time.time()
        work = output / "_work" / f"day={date}"
        work.mkdir(parents=True, exist_ok=True)
        nested = work / Path(member).name
        log = output / "logs" / f"day={date}.geometric.log"
        state_path = output / "manifests" / f"day={date}.run_state.json"
        state = {"date": date, "member": member, "status": "extracting", "complete": False}
        save_json(state_path, state)
        if not nested.exists():
            unrar_member = member.replace("/", "\\")
            run([str(args.unrar), "e", "-o+", "-idq", str(args.archive.resolve()), unrar_member, str(work.resolve()) + "\\"], log)
        state["status"] = "geometric_stage0"; save_json(state_path, state)
        daily_raw = work / "geometric"
        run([
            sys.executable, str(root / "stage0" / "scripts" / "run_full_day_2017.py"),
            "--input", str(nested), "--roads", str(args.roads), "--nodes", str(args.nodes),
            "--output-dir", str(daily_raw), "--input-crs", "gcj02", "--buckets", str(args.buckets),
        ], log)
        state["status"] = "standardizing"; save_json(state_path, state)
        run([
            sys.executable, str(root / "stage0" / "scripts" / "standardize_stage0_day.py"),
            "--source-dir", str(daily_raw), "--output-root", str(output), "--date", date, "--link-mode", "hardlink",
        ], log)
        state.update({"status": "complete", "complete": True, "runtime_seconds": time.time() - day_started}); save_json(state_path, state)
        if not args.keep_work:
            shutil.rmtree(work)
        print(f"day={date} complete", flush=True)


if __name__ == "__main__":
    main()
