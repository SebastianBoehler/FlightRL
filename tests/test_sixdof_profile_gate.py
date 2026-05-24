from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_sixdof_profile_gate", ROOT / "scripts" / "run_sixdof_profile_gate.py")
assert SPEC and SPEC.loader
GATE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GATE
SPEC.loader.exec_module(GATE)


def test_profile_gate_selects_position_yaw_candidates_once() -> None:
    selected = GATE.profile_candidates(matrix())

    assert [record["label"] for record in selected] == ["history", "multi"]


def test_profile_gate_dry_run_writes_suite_and_matrix_commands(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix()))
    output = tmp_path / "gate.json"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_sixdof_profile_gate.py"),
            "--matrix",
            str(matrix_path),
            "--profiles",
            "position_yaw_recovery",
            "broad",
            "--output-dir",
            str(tmp_path / "profiles"),
            "--output",
            str(output),
            "--max-candidates",
            "1",
            "--steps",
            "4",
            "--num-envs",
            "4",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output.read_text())
    assert report["run"] is False
    assert len(report["commands"]) == 3
    assert report["commands"][0][1].endswith("evaluate_sixdof_suite.py")
    assert "--candidate" in report["commands"][0]
    assert report["commands"][-1][1].endswith("build_sixdof_profile_matrix.py")
    assert output.with_suffix(".md").exists()


def test_profile_gate_markdown_includes_loaded_summary(tmp_path: Path) -> None:
    profile_matrix = tmp_path / "profile_matrix.json"
    profile_matrix.write_text(json.dumps({"records": [profile_record()]}))
    summary = GATE.load_profile_summary(profile_matrix)
    markdown = GATE.render_markdown(
        {
            "profiles": ["broad"],
            "candidates": [{"label": "history", "tasks": ["position_yaw"]}],
            "profile_matrix_output": str(profile_matrix),
            "results": [{"returncode": 0}],
            "profile_summary": summary,
            "safety": "safe",
        }
    )

    assert summary[0]["worst_survival_fraction"] == 0.7
    assert "worst survival" in markdown
    assert "| history | False | 0.7000 |" in markdown


def matrix() -> dict:
    return {
        "best_by_task": {
            "obstacle_avoidance": candidate("obstacle", ["obstacle_avoidance"]),
            "position_yaw": candidate("history", ["position_yaw"]),
        },
        "best_multitask": candidate("multi", ["position_yaw", "obstacle_avoidance", "circle"]),
    }


def candidate(label: str, tasks: list[str]) -> dict:
    return {"label": label, "checkpoint": f"{label}.pt", "tasks": tasks}


def profile_record() -> dict:
    return {
        "label": "history",
        "tasks": ["position_yaw"],
        "passed_all_profiles": False,
        "worst_survival_fraction": 0.7,
        "worst_completed_fraction": 0.3,
        "worst_position_error_m": 4.0,
        "worst_yaw_error_rad": 0.2,
        "worst_clearance_p01_m": 0.05,
    }
