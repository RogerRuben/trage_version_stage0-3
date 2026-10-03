"""Stage-0-environment launcher for the isolated PyTorch worker."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from ..config import Stage2V4Config
from ..contracts import Stage2V4ContractError


def train_deep_model(
    config_path: str | Path,
    tensor_root: str | Path,
    output_root: str | Path,
    prediction_root: str | Path,
    config: Stage2V4Config,
    *,
    resume: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    python = Path(str(config.section("deep")["python_executable"]))
    if not python.is_file():
        raise Stage2V4ContractError(f"configured PyTorch interpreter is missing: {python}")
    command = [
        str(python),
        "-m",
        "stage2.v4.models.train_worker",
        "--config",
        str(config_path),
        "--tensor-root",
        str(tensor_root),
        "--output",
        str(output_root),
        "--prediction-root",
        str(prediction_root),
    ]
    if resume:
        command.append("--resume")
    if force:
        command.append("--force")
    repository = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        command,
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise Stage2V4ContractError(
            "PyTorch worker failed:\n"
            + (result.stdout[-4000:] if result.stdout else "")
            + (result.stderr[-4000:] if result.stderr else "")
        )
    manifest_path = Path(output_root) / "model_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage2V4ContractError("PyTorch worker produced no valid manifest") from exc
    if manifest.get("engineering_status") != "PASS":
        raise Stage2V4ContractError("PyTorch worker manifest is not PASS")
    return manifest
