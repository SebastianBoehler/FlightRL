from __future__ import annotations

from pathlib import Path
import subprocess
import sys


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/evaluate_puffer_fixed_door_checkpoint.py"
)


def test_checkpoint_evaluator_help_exposes_isolated_challenge_inputs() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--challenge" in result.stdout
    assert "--control-report" in result.stdout
    assert "--output" in result.stdout


def test_checkpoint_evaluator_rejects_challenge_without_control_first() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--lineage-report",
            "/tmp/not-read.json",
            "--challenge",
            "pixel-noise",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--challenge requires --control-report" in result.stderr
