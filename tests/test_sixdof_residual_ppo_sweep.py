from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_sixdof_residual_ppo_sweep", ROOT / "scripts" / "run_sixdof_residual_ppo_sweep.py")
assert SPEC and SPEC.loader
SWEEP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SWEEP
SPEC.loader.exec_module(SWEEP)


def test_default_residual_variants_cover_scale_and_safety_knobs() -> None:
    variants = SWEEP.default_variants()

    assert {variant.residual_scale for variant in variants} >= {0.05, 0.1, 0.15, 0.2}
    assert {variant.reference_coef for variant in variants} >= {2.0, 4.0}
    assert {variant.action_std for variant in variants} >= {0.01, 0.02}
    assert {variant.reward_mode for variant in variants} == {"progress_yaw_clearance"}


def test_residual_ppo_sweep_dry_run_writes_manifest(tmp_path: Path) -> None:
    output = tmp_path / "residual_sweep.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_sixdof_residual_ppo_sweep.py"),
            "--max-variants",
            "1",
            "--updates",
            "1",
            "--report",
            str(output),
            "--output-dir",
            str(tmp_path / "ppo"),
            "--train-num-envs",
            "8",
            "--horizon",
            "4",
            "--eval-num-envs",
            "4",
            "--gate-steps",
            "5",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text())
    record = report["records"][0]

    assert report["run"] is False
    assert report["thresholds"]["max_teacher_action_l2_mean"] == 0.02
    assert len(report["records"]) == 1
    assert len(record["commands"]) == 3
    assert "--zero-weights" in record["commands"][0]
    assert "--controller" in record["commands"][1]
    assert "teacher_residual" in record["commands"][1]
    assert "--max-yaw-p95-error-rad" in record["commands"][2]
    assert output.with_suffix(".md").exists()


def test_residual_sweep_summary_penalizes_teacher_deviation() -> None:
    safe = record("safe", passed=True, teacher_l2=0.001, completed=0.9)
    risky = record("risky", passed=False, teacher_l2=0.1, completed=1.0)

    summary = SWEEP.sweep_summary([risky, safe])

    assert summary["completed"] == 0
    assert summary["best"]["name"] == "safe"


def record(name: str, *, passed: bool, teacher_l2: float, completed: float) -> dict:
    gate = {
        "passed": passed,
        "sim_gate_passed": True,
        "failures": [],
        "mean_completed_fraction": completed,
        "mean_survival_fraction": 1.0,
        "mean_position_error_m": 0.1,
        "mean_yaw_error_rad": 0.2,
        "yaw_error_p95_rad": 0.3,
        "clearance_p01_m": 0.2,
        "teacher_action_l2_mean": teacher_l2,
        "teacher_action_l2_p95": teacher_l2,
    }
    return {"variant": {"name": name}, "checkpoint": f"{name}.pt", "gate": gate}
