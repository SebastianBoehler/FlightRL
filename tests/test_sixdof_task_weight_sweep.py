from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_sixdof_task_weight_sweep", ROOT / "scripts" / "run_sixdof_task_weight_sweep.py")
assert SPEC and SPEC.loader
SWEEP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SWEEP
SPEC.loader.exec_module(SWEEP)


def test_default_variants_cover_task_weight_focuses() -> None:
    variants = SWEEP.default_variants()
    weights = {variant.name: dict(variant.task_weights) for variant in variants}

    assert weights["balanced_control"] == {}
    assert weights["focus_position_circle_15"] == {"position_yaw": 1.5, "circle": 1.5}
    assert weights["focus_circle_2"] == {"circle": 2.0}


def test_task_weight_sweep_dry_run_writes_manifest(tmp_path: Path) -> None:
    output = tmp_path / "sweep.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_sixdof_task_weight_sweep.py"),
            "--max-variants",
            "2",
            "--report",
            str(output),
            "--output-dir",
            str(tmp_path / "tw"),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text())
    assert report["run"] is False
    assert len(report["records"]) == 2
    assert "--task-weight" in report["records"][1]["commands"][0]
    assert "evaluate_sixdof_suite.py" in report["records"][0]["commands"][1][1]
    assert output.with_suffix(".md").exists()


def test_load_suite_summary_compacts_per_task_gates(tmp_path: Path) -> None:
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "gate": {"passed": False, "failures": ["completion"]},
                        "per_task_gate": {"circle": {"passed": False, "failures": ["completion"]}},
                        "metrics": {
                            "mean_completed_fraction": 0.2,
                            "mean_survival_fraction": 0.7,
                            "mean_position_error_m": 3.0,
                            "clearance_p01_m": 0.05,
                        },
                    }
                ]
            }
        )
    )

    summary = SWEEP.load_suite_summary(str(suite))

    assert summary["failures"] == ["completion"]
    assert summary["per_task_gate"]["circle"]["failures"] == ["completion"]


def test_sweep_summary_ranks_completion() -> None:
    records = [
        record("weak", completed=0.2, survival=0.9, position_error=1.0),
        record("strong", completed=0.5, survival=0.7, position_error=2.0),
    ]

    assert SWEEP.sweep_summary(records)["best"]["name"] == "strong"


def record(name: str, *, completed: float, survival: float, position_error: float) -> dict:
    return {
        "variant": {"name": name},
        "checkpoint": f"{name}.pt",
        "results": [{"returncode": 0}],
        "gate": {
            "passed": False,
            "failures": ["completion"],
            "mean_completed_fraction": completed,
            "mean_survival_fraction": survival,
            "mean_position_error_m": position_error,
            "clearance_p01_m": 0.1,
            "per_task_gate": {},
        },
    }
