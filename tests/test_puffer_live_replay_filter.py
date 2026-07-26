from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("train_puffer_sixdof_live_replay", ROOT / "scripts" / "train_puffer_sixdof_live_replay.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def args() -> SimpleNamespace:
    return SimpleNamespace(
        max_abs_tilt_deg=35.0,
        min_zrange_m=0.18,
        min_state_height_m=0.20,
        max_state_height_m=1.20,
        max_speed_m_s=3.0,
        target=[0.0, 0.0, 0.50],
        target_thrust_min=-0.2,
        target_thrust_max=0.45,
        target_rate_clip_abs=0.55,
    )


def safe_row(**overrides: float) -> dict[str, float]:
    row = {
        "sys.isTumbled": 0.0,
        "sys.canfly": 1.0,
        "stabilizer.roll": 2.0,
        "stabilizer.pitch": -3.0,
        "stateEstimate.z": 0.50,
        "stateEstimate.vx": 0.1,
        "stateEstimate.vy": 0.0,
        "stateEstimate.vz": 0.0,
        "range.zrange": 500.0,
    }
    row.update(overrides)
    return row


def test_live_row_filter_keeps_precontact_rows() -> None:
    assert MODULE.live_row_allowed(safe_row(**{"range.back": 146.0}), args())


def test_live_row_filter_drops_tumbled_and_floor_contact_rows() -> None:
    assert not MODULE.live_row_allowed(safe_row(**{"sys.isTumbled": 1.0}), args())
    assert not MODULE.live_row_allowed(safe_row(**{"range.zrange": 11.0}), args())
    assert not MODULE.live_row_allowed(safe_row(**{"stabilizer.pitch": -79.0}), args())


def test_shape_targets_clips_training_labels() -> None:
    shaped = MODULE.shape_targets(
        MODULE.np.asarray([[1.0, -0.8, 0.7, 0.2], [-1.0, 0.1, -0.1, 0.9]], dtype=MODULE.np.float32),
        args(),
    )
    expected = MODULE.np.asarray([[0.45, -0.55, 0.55, 0.2], [-0.2, 0.1, -0.1, 0.55]], dtype=MODULE.np.float32)
    assert MODULE.np.allclose(shaped, expected)
