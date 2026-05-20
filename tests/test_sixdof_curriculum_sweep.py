from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_sixdof_curriculum_sweep", ROOT / "scripts" / "run_sixdof_curriculum_sweep.py")
assert SPEC and SPEC.loader
SWEEP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SWEEP
SPEC.loader.exec_module(SWEEP)


def test_default_curriculum_sweep_covers_staged_profiles() -> None:
    variants = SWEEP.default_variants()
    profiles = {profile for variant in variants for profile in variant.profiles}
    assert {"position_yaw_easy", "position_yaw_medium", "broad"} <= profiles
    assert {variant.hidden_size for variant in variants} >= {128, 256}


def test_curriculum_sweep_dry_run_writes_manifest(tmp_path: Path) -> None:
    output = tmp_path / "sweep.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_sixdof_curriculum_sweep.py"),
            "--max-variants",
            "1",
            "--report",
            str(output),
            "--output-dir",
            str(tmp_path / "artifacts"),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text())
    assert report["run"] is False
    assert len(report["records"]) == 1
    commands = report["records"][0]["commands"]
    assert any("--reset-profile" in command for command in commands)
    assert any("--eval-reset-profile" in command for command in commands)
    assert output.with_suffix(".md").exists()


def test_load_gate_summary_compacts_metrics(tmp_path: Path) -> None:
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "gate": {"passed": False, "failures": ["completion"]},
                "metrics": {
                    "mean_position_error_m": 1.2,
                    "clearance_p01_m": 0.3,
                    "mean_completed_fraction": 0.4,
                    "mean_survival_fraction": 0.8,
                },
            }
        )
    )
    summary = SWEEP.load_gate_summary(str(gate))
    assert summary["passed"] is False
    assert summary["mean_survival_fraction"] == 0.8


def test_sweep_summary_ranks_best_gate_candidate() -> None:
    records = [
        record("early", completed=0.2, survival=0.7, position_error=1.0),
        record("late", completed=0.5, survival=0.6, position_error=2.0),
    ]
    summary = SWEEP.sweep_summary(records)
    assert summary["completed"] == 2
    assert summary["best_medium"]["name"] == "late"


def record(name: str, *, completed: float, survival: float, position_error: float) -> dict:
    gate = {
        "passed": False,
        "failures": ["completion"],
        "mean_completed_fraction": completed,
        "mean_survival_fraction": survival,
        "mean_position_error_m": position_error,
        "clearance_p01_m": 0.1,
    }
    return {
        "variant": {"name": name},
        "checkpoint": f"{name}.pt",
        "results": [{"returncode": 0}],
        "gates": {"medium": gate, "broad": gate},
    }
