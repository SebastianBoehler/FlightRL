from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/crazyflie_fixed_door_control.py"


@pytest.mark.parametrize(
    ("argument", "value", "expected"),
    (
        ("--height-m", "0.19", "live height"),
        ("--height-m", "nan", "live height"),
        ("--duration-s", "15.01", "live duration"),
        ("--duration-s", "inf", "live duration"),
    ),
)
def test_control_rejects_invalid_envelope_before_confirmation(
    argument: str,
    value: str,
    expected: str,
) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), argument, value],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert expected in result.stderr
    assert "--confirm-flight is required" not in result.stderr
