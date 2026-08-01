from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from flightrl.evidence_scope import EDGE_DEPLOYMENT_VERIFIER_MISSING
from sim2real_pipeline_test_support import write_json, write_ready_inputs


ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_cli_rebuilds_blocked_current_style_chain(tmp_path: Path) -> None:
    paths = write_ready_inputs(tmp_path)
    output_dir = tmp_path / "out"
    blockers = write_json(tmp_path / "blockers.json", {"blockers": []})

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_sim2real_pipeline.py",
            "--label",
            "cli",
            "--output-dir",
            str(output_dir),
            "--hardware-config",
            str(paths["hardware_config"]),
            "--base-config",
            str(paths["base_config"]),
            "--output-config",
            str(paths["output_config"]),
            "--motor-calibration",
            str(paths["motor_calibration"]),
            "--stationary-noise",
            str(paths["stationary_noise"]),
            "--hardware-latency",
            str(paths["hardware_latency"]),
            "--calibration-quality",
            str(paths["calibration_quality"]),
            "--deployment-readiness",
            str(paths["deployment_readiness"]),
            "--replay-comparison",
            str(paths["replay_comparison"]),
            "--motor-bench",
            str(paths["motor_bench"]),
            "--sim-readiness",
            str(paths["sim_readiness"]),
            "--room-report",
            str(paths["room_report"]),
            "--live-script",
            str(paths["live_scripts"][0]),
            "--hardware-blockers-file",
            str(blockers),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "transfer_approved=False" in result.stdout
    pipeline = json.loads(
        (output_dir / "sim2real_pipeline_cli.json").read_text()
    )
    assert EDGE_DEPLOYMENT_VERIFIER_MISSING in pipeline["gate_failures"]
    assert (output_dir / "sim2real_pipeline_cli.json").exists()
    assert (output_dir / "sim2real_evidence_gap_cli.json").exists()


def test_pipeline_cli_uses_hardware_blocker_file_by_default(tmp_path: Path) -> None:
    paths = write_ready_inputs(tmp_path)
    blockers = write_json(tmp_path / "blockers.json", {"blockers": ["m3_motor_issue"]})
    output_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_sim2real_pipeline.py",
            "--label",
            "blocked",
            "--output-dir",
            str(output_dir),
            "--hardware-config",
            str(paths["hardware_config"]),
            "--base-config",
            str(paths["base_config"]),
            "--output-config",
            str(paths["output_config"]),
            "--motor-calibration",
            str(paths["motor_calibration"]),
            "--stationary-noise",
            str(paths["stationary_noise"]),
            "--hardware-latency",
            str(paths["hardware_latency"]),
            "--calibration-quality",
            str(paths["calibration_quality"]),
            "--deployment-readiness",
            str(paths["deployment_readiness"]),
            "--replay-comparison",
            str(paths["replay_comparison"]),
            "--motor-bench",
            str(paths["motor_bench"]),
            "--sim-readiness",
            str(paths["sim_readiness"]),
            "--room-report",
            str(paths["room_report"]),
            "--live-script",
            str(paths["live_scripts"][0]),
            "--hardware-blockers-file",
            str(blockers),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    pipeline = json.loads((output_dir / "sim2real_pipeline_blocked.json").read_text())
    assert "transfer_approved=False" in result.stdout
    assert "m3_motor_issue" in pipeline["blocking_items"]
    assert pipeline["inputs"]["hardware_blockers_file"]["path"] == str(blockers)
    assert len(pipeline["inputs"]["hardware_blockers_file"]["sha256"]) == 64
    assert pipeline["inputs"]["hardware_blockers"] == ["m3_motor_issue"]
