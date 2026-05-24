from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("train_sixdof_dagger", ROOT / "scripts" / "train_sixdof_dagger.py")
assert SPEC and SPEC.loader
DAGGER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DAGGER
SPEC.loader.exec_module(DAGGER)


def test_dagger_select_best_uses_yaw_after_position() -> None:
    reports = [
        report("yaw_drift", position_error=0.5, yaw_error=1.0, yaw_p95=2.0),
        report("yaw_aligned", position_error=0.5, yaw_error=0.2, yaw_p95=0.4),
    ]

    assert DAGGER.select_best(reports)["checkpoint"] == "yaw_aligned.pt"


def report(name: str, *, position_error: float, yaw_error: float, yaw_p95: float) -> dict:
    return {
        "checkpoint": f"{name}.pt",
        "gate": {"passed": False},
        "metrics": {
            "mean_completed_fraction": 1.0,
            "mean_survival_fraction": 1.0,
            "mean_position_error_m": position_error,
            "mean_yaw_error_rad": yaw_error,
            "yaw_error_p95_rad": yaw_p95,
            "min_clearance_m": 0.5,
            "clearance_p01_m": 0.5,
        },
    }
