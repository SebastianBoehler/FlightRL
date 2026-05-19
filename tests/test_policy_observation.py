from __future__ import annotations

from pathlib import Path

import numpy as np

from flightrl import load_config
from flightrl.hardware.policy_observation import build_policy_observation, initial_policy_state, update_previous_action


ROOT = Path(__file__).resolve().parents[1]


def test_hardware_policy_observation_matches_config_dim() -> None:
    config = load_config(ROOT / "configs" / "tasks" / "crazyflie_hover.toml")
    state = initial_policy_state(config)
    update_previous_action(state, np.array([0.1, -0.2], dtype=np.float32))
    telemetry = {
        "stateEstimate.z": 0.4,
        "stabilizer.pitch": 2.0,
        "pm.vbat": 3.8,
        "range.front": 500.0,
        "range.back": 600.0,
        "range.left": 700.0,
        "range.right": 800.0,
        "range.up": 1500.0,
        "range.zrange": 400.0,
    }

    obs = build_policy_observation(config, telemetry, state, target=(0.0, 0.0, 0.45))

    assert obs.shape == (config.observation_dim,)
    assert np.isfinite(obs).all()
    assert obs[-2:].tolist() == [np.float32(0.1), np.float32(-0.2)]
