from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_sixdof_ppo_sweep", ROOT / "scripts" / "run_sixdof_ppo_sweep.py")
assert SPEC and SPEC.loader
SWEEP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SWEEP
SPEC.loader.exec_module(SWEEP)


def test_default_ppo_sweep_covers_reference_and_action_knobs() -> None:
    variants = SWEEP.default_variants()
    assert {variant.reference_coef for variant in variants} >= {1.0, 2.0}
    assert {variant.action_std for variant in variants} >= {0.04, 0.06}
    assert {variant.reward_mode for variant in variants} >= {"env", "progress"}


def test_ppo_sweep_dry_run_writes_manifest(tmp_path: Path) -> None:
    output = tmp_path / "ppo_sweep.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_sixdof_ppo_sweep.py"),
            "--max-variants",
            "1",
            "--report",
            str(output),
            "--output-dir",
            str(tmp_path / "ppo"),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text())
    assert report["run"] is False
    assert len(report["records"]) == 1
    command = report["records"][0]["commands"][0]
    assert "--reference-coef" in command
    assert "--reward-mode" in command
    assert output.with_suffix(".md").exists()


def test_ppo_sweep_summary_ranks_completion() -> None:
    summary = SWEEP.sweep_summary(
        [
            record("low", completed=0.1, survival=0.9, position_error=1.0),
            record("high", completed=0.2, survival=0.5, position_error=5.0),
        ]
    )
    assert summary["completed"] == 2
    assert summary["best_medium"]["name"] == "high"


def record(name: str, *, completed: float, survival: float, position_error: float) -> dict:
    gate = {
        "passed": False,
        "failures": ["completion"],
        "mean_completed_fraction": completed,
        "mean_survival_fraction": survival,
        "mean_position_error_m": position_error,
        "clearance_p01_m": 0.1,
    }
    return {"variant": {"name": name}, "checkpoint": f"{name}.pt", "results": [{"returncode": 0}], "gates": {"medium": gate, "broad": gate}}
