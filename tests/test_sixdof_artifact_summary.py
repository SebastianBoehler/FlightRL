from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_artifact_summary_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    gate = tmp_path / "gate.json"
    parity = tmp_path / "parity.json"
    output = tmp_path / "summary.json"
    gate.write_text(
        json.dumps(
            {
                "tasks": ["obstacle_avoidance"],
                "gate": {"passed": True, "failures": []},
                "metrics": {
                    "mean_position_error_m": 0.1,
                    "mean_completed_fraction": 1.0,
                    "clearance_p01_m": 0.5,
                    "min_clearance_m": 0.2,
                    "action_saturation_fraction": 0.0,
                    "teacher_action_l2_mean": 0.01,
                    "teacher_action_l2_p95": 0.02,
                },
            }
        )
    )
    parity.write_text(json.dumps({"evidence_scope": "desktop_cpu_only", "model": "model.ts", "parity": {"max_abs_error": 0.0, "mean_abs_error": 0.0}}))
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "summarize_sixdof_artifact.py"),
            "--name",
            "test",
            "--checkpoint",
            "policy.pt",
            "--gate",
            str(gate),
            "--desktop-parity",
            str(parity),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(output.read_text())
    assert summary["gate"]["passed"] is True
    assert summary["desktop_parity"]["evidence_scope"] == "desktop_cpu_only"
    assert "edge_parity" not in summary
    assert output.with_suffix(".md").exists()
