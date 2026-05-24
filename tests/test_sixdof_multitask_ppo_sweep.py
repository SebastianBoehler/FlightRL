from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_sixdof_multitask_ppo_sweep", ROOT / "scripts" / "run_sixdof_multitask_ppo_sweep.py")
assert SPEC and SPEC.loader
SWEEP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SWEEP
SPEC.loader.exec_module(SWEEP)


def test_default_variants_cover_multitask_ppo_knobs() -> None:
    variants = SWEEP.default_variants()
    probabilities = {variant.name: dict(variant.task_probabilities) for variant in variants}

    assert probabilities["balanced_h64_ref2_std002"] == {}
    assert probabilities["py_focus4_h64_ref2_std002"] == {"position_yaw": 4.0}
    assert probabilities["py_yaw_focus4_h64_ref2_std002"] == {"position_yaw": 4.0}
    assert probabilities["py_circle3_h64_ref2_std002"] == {"position_yaw": 3.0, "circle": 3.0}
    assert {variant.action_std for variant in variants} >= {0.01, 0.02}
    assert "progress_yaw_clearance" in {variant.reward_mode for variant in variants}


def test_multitask_ppo_sweep_dry_run_builds_profile_matrix_commands(tmp_path: Path) -> None:
    output = tmp_path / "sweep.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_sixdof_multitask_ppo_sweep.py"),
            "--max-variants",
            "1",
            "--report",
            str(output),
            "--output-dir",
            str(tmp_path / "ppo"),
            "--updates",
            "1",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text())

    assert report["run"] is False
    assert len(report["records"]) == 2
    assert len(report["records"][0]["commands"]) == 0
    assert "--train-tasks" in report["records"][1]["commands"][0]
    assert report["records"][1]["variant"]["updates"] == 1
    assert "--candidate" in report["commands"][0]
    assert report["commands"][0].count("--candidate") == 2
    assert report["commands"][-1][1].endswith("build_sixdof_profile_matrix.py")
    assert output.with_suffix(".md").exists()


def test_multitask_ppo_summary_uses_profile_order() -> None:
    report = {
        "records": [{"variant": {"name": "a"}}, {"variant": {"name": "b"}}],
        "results": [{"returncode": 0}],
        "profile_summary": [
            {
                "label": "b",
                "checkpoint": "b.pt",
                "passed_all_profiles": False,
                "worst_completed_fraction": 0.8,
                "worst_position_error_m": 1.0,
                "worst_yaw_error_rad": 0.4,
                "worst_clearance_p01_m": 0.1,
            }
        ],
    }

    summary = SWEEP.summarize(report)

    assert summary["completed"] is True
    assert summary["best"]["label"] == "b"
    assert summary["best"]["worst_yaw_error_rad"] == 0.4
