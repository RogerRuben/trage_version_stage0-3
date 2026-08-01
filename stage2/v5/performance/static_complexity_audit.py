"""AST-based guard against prohibited full-data implementation patterns."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "stage2_v5_static_complexity_audit.1"
EXCLUDED = {"reference.py", "static_complexity_audit.py", "benchmark_hotspots.py"}
ALLOWED_PARTITION_READS = {"data.py", "service_time_audit.py", "preflight.py"}


class Scanner(ast.NodeVisitor):
    def __init__(self, path: Path):
        self.path = path
        self.findings: list[dict[str, Any]] = []
        self.loop_depth = 0

    def _add(self, node: ast.AST, pattern: str, blocking: bool = True, reason: str = "") -> None:
        self.findings.append(
            {
                "file": self.path.as_posix(),
                "line": int(getattr(node, "lineno", 0)),
                "pattern": pattern,
                "blocking": bool(blocking),
                "justification": reason,
            }
        )

    def visit_For(self, node: ast.For) -> None:
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    visit_AsyncFor = visit_For

    def visit_Call(self, node: ast.Call) -> None:
        attribute = node.func.attr if isinstance(node.func, ast.Attribute) else ""
        if attribute in {"iterrows", "itertuples"}:
            self._add(node, attribute)
        if attribute == "apply":
            axis_one = any(keyword.arg == "axis" and isinstance(keyword.value, ast.Constant) and keyword.value.value == 1 for keyword in node.keywords)
            if axis_one:
                self._add(node, "DataFrame.apply(axis=1)")
            if isinstance(node.func.value, ast.Call) and isinstance(node.func.value.func, ast.Attribute) and node.func.value.func.attr == "groupby":
                self._add(node, "groupby.apply")
        if attribute == "concat" and self.loop_depth:
            self._add(node, "concat_inside_loop")
        if attribute in {"read_parquet", "read_table"} and self.loop_depth:
            allowed = self.path.name in ALLOWED_PARTITION_READS
            self._add(
                node,
                "partition_streaming_read" if allowed else "repeated_parquet_read_inside_loop",
                blocking=not allowed,
                reason="bounded one-partition-at-a-time audit scan" if allowed else "",
            )
        self.generic_visit(node)


def audit(root: str | Path) -> dict[str, Any]:
    source_root = Path(root)
    findings: list[dict[str, Any]] = []
    scanned: list[str] = []
    digest = hashlib.sha256()
    for path in sorted(source_root.rglob("*.py")):
        if path.name in EXCLUDED or "__pycache__" in path.parts:
            continue
        raw = path.read_bytes()
        digest.update(path.relative_to(source_root).as_posix().encode())
        digest.update(raw)
        scanner = Scanner(path.relative_to(source_root.parent.parent))
        scanner.visit(ast.parse(raw.decode("utf-8")))
        findings.extend(scanner.findings)
        scanned.append(path.as_posix())
    blocking = [finding for finding in findings if finding["blocking"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if not blocking else "FAIL",
        "scanned_file_count": len(scanned),
        "source_sha256": digest.hexdigest(),
        "blocking_finding_count": len(blocking),
        "allowlisted_finding_count": len(findings) - len(blocking),
        "findings": findings,
        "prohibited_patterns": [
            "groupby.apply", "DataFrame.apply(axis=1)", "iterrows", "itertuples",
            "concat_inside_loop", "repeated_parquet_read_inside_loop",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="stage2/v5")
    parser.add_argument("--output", default="stage2/docs/v5/stage2_v5_static_complexity_audit.json")
    args = parser.parse_args()
    report = audit(args.root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
