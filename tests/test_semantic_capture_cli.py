from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "crazyflie_semantic_find.py"


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_semantic_runner_is_capture_and_shadow_only() -> None:
    result = _run("--prompt", "door", "--dry-run")

    assert result.returncode == 0
    assert "mode=camera_only" in result.stdout
    assert "controls_drone=false" in result.stdout


def test_semantic_runner_has_no_legacy_flight_authority_flags() -> None:
    result = _run("--prompt", "door", "--dry-run", "--flight")

    assert result.returncode != 0
    assert "unrecognized arguments: --flight" in result.stderr
