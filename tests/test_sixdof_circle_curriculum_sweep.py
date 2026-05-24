from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_sixdof_circle_curriculum_sweep", ROOT / "scripts" / "run_sixdof_circle_curriculum_sweep.py")
assert SPEC and SPEC.loader
SWEEP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SWEEP
SPEC.loader.exec_module(SWEEP)


def test_default_circle_sweep_covers_circle_profiles() -> None:
    variants = SWEEP.default_variants()
    profiles = {profile for variant in variants for profile in variant.profiles}

    assert {"circle_easy", "circle_recovery"} <= profiles
    assert {variant.hidden_size for variant in variants} >= {128, 256}
    assert "history1" in {variant.observation_mode for variant in variants}


def test_circle_sweep_dry_run_writes_manifest(tmp_path: Path) -> None:
    output = tmp_path / "circle.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_sixdof_circle_curriculum_sweep.py"),
            "--max-variants",
            "1",
            "--steps",
            "2",
            "--epochs",
            "1",
            "--report",
            str(output),
            "--output-dir",
            str(tmp_path / "circle"),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text())
    commands = report["records"][0]["commands"]

    assert report["run"] is False
    assert "--task" in commands[0]
    assert "circle" in commands[0]
    assert "--eval-reset-profile" in commands[-3]
    assert commands[-2][1].endswith("evaluate_sixdof_checkpoint.py")
    assert output.with_suffix(".md").exists()


def test_circle_sweep_summary_ranks_recovery_gate() -> None:
    records = [
        record("weak", completed=0.2, position_error=1.0, yaw_p95=1.0),
        record("strong", completed=0.5, position_error=2.0, yaw_p95=2.0),
    ]

    assert SWEEP.summarize(records)["best"]["name"] == "strong"


def record(name: str, *, completed: float, position_error: float, yaw_p95: float) -> dict:
    gate = {"passed": False, "failures": ["completion"], "completed": completed, "position_error": position_error, "yaw_p95": yaw_p95, "clearance": 0.1}
    return {"variant": {"name": name}, "checkpoint": f"{name}.pt", "results": [{"returncode": 0}], "gates": {"recovery": gate, "broad": gate}}
