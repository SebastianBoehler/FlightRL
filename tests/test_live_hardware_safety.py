from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from flightrl.sim2real.live_safety import build_live_safety_report, scan_live_script


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CRAZYFLIE_SCRIPTS = {
    "scripts/crazyflie_bringup.py",
    "scripts/crazyflie_log.py",
    "scripts/crazyflie_motor_bench.py",
    "scripts/crazyflie_semantic_find.py",
}


def test_live_safety_report_accepts_current_crazyflie_scripts() -> None:
    scripts = sorted((ROOT / "scripts").glob("crazyflie_*.py"))

    report = build_live_safety_report(scripts)

    assert {str(path.relative_to(ROOT)) for path in scripts} == EXPECTED_CRAZYFLIE_SCRIPTS
    assert report["summary"]["passed"] is True
    assert report["summary"]["learned_checkpoint_hardware_scripts"] == 0


def test_live_safety_rejects_checkpoint_control_without_approval(tmp_path: Path) -> None:
    script = tmp_path / "unsafe.py"
    script.write_text(
        """
from flightrl.hardware.cflib_bridge import require_cflib
import torch

checkpoint = torch.load("policy.pt")
modules = require_cflib()
commander.send_hover_setpoint(0, 0, 0, 0.3)
""".strip()
    )

    record = scan_live_script(script)

    assert record["passed"] is False
    assert any("checkpoint_control_without_hardware_approval" in failure for failure in record["failures"])


def test_live_safety_rejects_checkpoint_monitor_without_metadata(tmp_path: Path) -> None:
    script = tmp_path / "monitor.py"
    script.write_text(
        """
from flightrl.hardware.cflib_bridge import require_cflib
import torch

checkpoint = torch.load("policy.pt")
modules = require_cflib()
print(checkpoint)
""".strip()
    )

    record = scan_live_script(script)

    assert record["passed"] is False
    assert any("checkpoint_monitor_without_monitor_only_metadata" in failure for failure in record["failures"])


def test_live_safety_cli_writes_report(tmp_path: Path) -> None:
    output = tmp_path / "live_safety.json"

    result = subprocess.run(
        [sys.executable, "scripts/build_live_hardware_safety_report.py", "--output", str(output)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert output.exists()
    assert output.with_suffix(".md").exists()
    assert "passed=True" in result.stdout
