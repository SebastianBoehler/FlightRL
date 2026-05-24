from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_sixdof_recovery_dagger_sweep", ROOT / "scripts" / "run_sixdof_recovery_dagger_sweep.py")
assert SPEC and SPEC.loader
SWEEP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SWEEP
SPEC.loader.exec_module(SWEEP)


def test_recovery_dagger_variants_target_closed_loop_profiles() -> None:
    variants = SWEEP.default_variants()
    assert {variant.reset_profile for variant in variants} >= {"position_yaw_recovery", "position_yaw_medium"}
    assert {variant.action_weighting for variant in variants} >= {"none", "inverse_std"}
    assert any(variant.beta == 0.0 for variant in variants)


def test_recovery_dagger_dry_run_writes_manifest(tmp_path: Path) -> None:
    output = tmp_path / "sweep.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_sixdof_recovery_dagger_sweep.py"),
            "--max-variants",
            "1",
            "--report",
            str(output),
            "--output-dir",
            str(tmp_path / "dagger"),
            "--diagnostic-steps",
            "4",
            "--diagnostic-num-envs",
            "4",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text())
    assert report["run"] is False
    assert report["profiles"] == ["position_yaw_easy", "position_yaw_medium", "position_yaw_recovery", "broad"]
    assert len(report["records"]) == 1
    commands = report["records"][0]["commands"]
    assert commands[0][1].endswith("train_sixdof_dagger.py")
    assert "--reset-profile" in commands[0]
    assert "--select-by-eval" in commands[0]
    assert commands[1][1].endswith("diagnose_sixdof_policy.py")
    assert "--profiles" in commands[1]
    assert output.with_suffix(".md").exists()


def test_recovery_sweep_summary_ranks_survival_first() -> None:
    records = [
        record("precise", survival=0.3, clearance=0.4, position_error=0.5),
        record("survivor", survival=0.7, clearance=0.1, position_error=3.0),
    ]
    summary = SWEEP.sweep_summary(records)
    assert summary["completed"] == 2
    assert summary["best_recovery"]["name"] == "survivor"


def record(name: str, *, survival: float, clearance: float, position_error: float) -> dict:
    final = {
        "survival_fraction": survival,
        "clearance_p01_m": clearance,
        "position_error_mean_m": position_error,
        "yaw_error_mean_rad": 0.1,
    }
    return {
        "variant": {"name": name},
        "checkpoint": f"{name}.pt",
        "results": [{"returncode": 0}, {"returncode": 0}],
        "diagnostics": {"position_yaw_recovery": final, "broad": final},
    }
