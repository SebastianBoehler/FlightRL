from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from flightrl.sim2real.pipeline import path_provenance
from flightrl.sim2real.pipeline_verify import verify_pipeline


ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_verify_accepts_fresh_inputs(tmp_path: Path) -> None:
    pipeline = build_ready_pipeline(tmp_path)

    report = verify_pipeline(pipeline)

    assert report["passed"] is True
    assert report["failures"] == []


def test_pipeline_verify_detects_modified_input(tmp_path: Path) -> None:
    pipeline = build_ready_pipeline(tmp_path)
    data = json.loads(pipeline.read_text())
    hardware = Path(data["inputs"]["hardware_config"]["path"])
    hardware.write_text(hardware.read_text() + "\n# changed\n")

    report = verify_pipeline(pipeline)

    assert report["passed"] is False
    assert any("sha256_changed" in failure or "size_changed" in failure for failure in report["failures"])


def test_pipeline_verify_cli_exits_nonzero_for_stale_input(tmp_path: Path) -> None:
    pipeline = build_ready_pipeline(tmp_path)
    data = json.loads(pipeline.read_text())
    Path(data["inputs"]["hardware_config"]["path"]).write_text("changed\n")
    output = tmp_path / "verify.json"

    result = subprocess.run(
        [sys.executable, "scripts/verify_sim2real_pipeline.py", "--pipeline", str(pipeline), "--output", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert output.exists()
    assert "passed=False" in result.stdout


def build_ready_pipeline(tmp_path: Path) -> Path:
    input_file = tmp_path / "hardware.toml"
    input_file.write_text("[sim2real]\nmeasured = true\n")
    pipeline = tmp_path / "pipeline.json"
    pipeline.write_text(
        json.dumps(
            {
                "inputs": {
                    "hardware_config": path_provenance(input_file),
                    "hardware_blockers": ["none"],
                }
            }
        )
    )
    return pipeline
