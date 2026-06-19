from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_directional_controller_dry_run() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/crazyflie_avoidance_policy.py",
            "--controller",
            "directional",
            "--dry-run",
            "--target-direction-deg",
            "0",
            "--target-speed-m-s",
            "0.16",
            "--height-m",
            "0.50",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "dry_run avoidance command" in result.stdout
    assert "shadow=None" in result.stdout
