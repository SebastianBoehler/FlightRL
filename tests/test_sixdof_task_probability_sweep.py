from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_sixdof_task_probability_sweep", ROOT / "scripts" / "run_sixdof_task_probability_sweep.py")
assert SPEC and SPEC.loader
SWEEP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SWEEP
SPEC.loader.exec_module(SWEEP)


def test_default_variants_cover_task_probability_focuses() -> None:
    variants = SWEEP.default_variants()
    probabilities = {variant.name: dict(variant.task_probabilities) for variant in variants}

    assert probabilities["uniform_dagger"] == {}
    assert probabilities["sample_position_circle_2"] == {"position_yaw": 2.0, "circle": 2.0}
    assert probabilities["sample_circle_3"] == {"circle": 3.0}


def test_task_probability_sweep_dry_run_writes_manifest(tmp_path: Path) -> None:
    output = tmp_path / "sweep.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_sixdof_task_probability_sweep.py"),
            "--seed-dataset",
            str(tmp_path / "seed.npz"),
            "--initial-checkpoint",
            str(tmp_path / "initial.pt"),
            "--max-variants",
            "2",
            "--report",
            str(output),
            "--output-dir",
            str(tmp_path / "tp"),
            "--baseline-checkpoint",
            "baseline.pt",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text())

    assert report["run"] is False
    assert len(report["records"]) == 3
    assert report["records"][0]["variant"]["name"] == "baseline"
    assert "--task-probability" in report["records"][2]["commands"][0]
    assert "evaluate_sixdof_suite.py" in report["records"][0]["commands"][0][1]
    assert output.with_suffix(".md").exists()
