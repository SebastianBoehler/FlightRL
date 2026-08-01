from __future__ import annotations

import numpy as np
import pytest

from flightrl.mujoco import (
    is_mujoco_available,
    is_mujoco_rendering_available,
)


def test_mujoco_vision_puffer_env_contract_when_available() -> None:
    if not is_mujoco_available():
        pytest.skip("MuJoCo optional dependency is not installed")
    if not is_mujoco_rendering_available():
        pytest.skip("MuJoCo rendering backend is unavailable")
    pytest.importorskip("pufferlib")
    from flightrl.mujoco.vision_env import INTENT_DIM, MuJoCoVisionPufferEnv

    env = MuJoCoVisionPufferEnv(num_envs=2, seed=7)
    try:
        observations, _ = env.reset(seed=7)
        next_observations, rewards, terminals, truncations, _ = env.step(
            np.zeros((2, 4), dtype=np.float32)
        )
    finally:
        env.close()

    assert observations.shape == (2, 3 * 48 * 64 + INTENT_DIM)
    assert next_observations.shape == observations.shape
    assert np.isfinite(next_observations).all()
    assert np.isfinite(rewards).all()
    assert terminals.shape == (2,)
    assert truncations.shape == (2,)
