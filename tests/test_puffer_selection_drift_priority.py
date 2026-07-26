from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_puffer_sixdof_closed_loop.py"
SPEC = importlib.util.spec_from_file_location("train_puffer_sixdof_closed_loop", TRAIN_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
train_puffer_sixdof_closed_loop = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_puffer_sixdof_closed_loop)


def test_selection_score_prioritizes_open_space_drift_over_small_position_gain() -> None:
    stable = metric(position_error=0.55, open_speed=0.45, horizontal_speed=0.50, tilt=8.0)
    drifty = metric(position_error=0.35, open_speed=1.10, horizontal_speed=1.20, tilt=20.0)

    assert train_puffer_sixdof_closed_loop.score_metrics(stable) > train_puffer_sixdof_closed_loop.score_metrics(drifty)


def metric(*, position_error: float, open_speed: float, horizontal_speed: float, tilt: float) -> dict[str, float]:
    return {
        "mean_completed_fraction": 1.0,
        "mean_survival_fraction": 1.0,
        "clearance_p01_m": 0.5,
        "mean_position_error_m": position_error,
        "horizontal_speed_p95_m_s": horizontal_speed,
        "open_space_horizontal_speed_p95_m_s": open_speed,
        "tilt_p95_deg": tilt,
        "action_saturation_fraction": 0.0,
    }
