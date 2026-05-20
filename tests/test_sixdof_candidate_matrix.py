from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import torch

from flightrl.sixdof import SixDofPolicy


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_sixdof_candidate_matrix", ROOT / "scripts" / "build_sixdof_candidate_matrix.py")
assert SPEC and SPEC.loader
MATRIX = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MATRIX
SPEC.loader.exec_module(MATRIX)


def test_candidate_matrix_cli_ranks_and_reads_parity(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.pt"
    torch.save(
        {
            "state_dict": SixDofPolicy(hidden_size=16).state_dict(),
            "hidden_size": 16,
            "observation_dim": 28,
            "observation_mode": "base",
        },
        checkpoint,
    )
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"records": [suite_record("candidate", checkpoint)]}))
    parity = tmp_path / "parity.json"
    parity.write_text(json.dumps({"model": "candidate.ts", "observation": {"mode": "base"}, "parity": {"max_abs_error": 0.0}}))
    output = tmp_path / "matrix.json"

    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_sixdof_candidate_matrix.py"), "--suite", str(suite), "--parity", f"candidate={parity}", "--output", str(output)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output.read_text())
    assert report["records"][0]["edge_parity"]["passed"] is True
    assert report["records"][0]["checkpoint_meta"]["observation_mode"] == "base"
    assert report["best_by_task"]["position_yaw"]["label"] == "candidate"
    assert output.with_suffix(".md").exists()


def test_candidate_matrix_prefers_passing_candidates() -> None:
    records = [
        {"label": "failed", "tasks": ["position_yaw"], "passed": False, "edge_parity": {"present": True}, "mean_completed_fraction": 0.9, "mean_survival_fraction": 0.9, "mean_position_error_m": 0.1, "clearance_p01_m": 0.2, "checkpoint": "a", "failures": ["completion"]},
        {"label": "passed", "tasks": ["position_yaw"], "passed": True, "edge_parity": {"present": False}, "mean_completed_fraction": 0.2, "mean_survival_fraction": 0.2, "mean_position_error_m": 2.0, "clearance_p01_m": 0.1, "checkpoint": "b", "failures": []},
    ]
    assert MATRIX.best_by_task(records)["position_yaw"]["label"] == "passed"


def suite_record(label: str, checkpoint: Path) -> dict:
    return {
        "label": label,
        "controller": "checkpoint",
        "checkpoint": str(checkpoint),
        "tasks": ["position_yaw"],
        "gate": {"passed": True, "failures": []},
        "metrics": {
            "mean_completed_fraction": 1.0,
            "mean_survival_fraction": 1.0,
            "mean_position_error_m": 0.1,
            "min_clearance_m": 0.2,
            "clearance_p01_m": 0.2,
        },
    }
