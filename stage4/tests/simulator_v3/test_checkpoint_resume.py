import json
from pathlib import Path


def test_checkpoint_resume_audit_file_passes_if_present():
    path = Path("stage4/docs/results/simulator_v3/simulator_v3_checkpoint_resume_audit.json")
    if path.exists():
        data = json.loads(path.read_text())
        assert data["checkpoint_resume_pass"] == "PASS"

