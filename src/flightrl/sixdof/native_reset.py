from __future__ import annotations

import numpy as np

from .physics import sample_physics


def native_reset_one(env, idx: int, rng: np.uint32) -> np.uint32:
    def rnd(lo: float, hi: float) -> float:
        nonlocal rng
        rng = np.uint32((1664525 * int(rng) + 1013904223) & 0xFFFFFFFF)
        unit = float(int(rng) >> 8) / 16777215.0
        return lo + unit * (hi - lo)

    env.position[idx] = [rnd(-0.8, 0.8), rnd(-0.8, 0.8), rnd(0.35, 0.9)]
    env.velocity[idx] = 0.0
    env.body_rates[idx] = 0.0
    yaw = rnd(-np.pi, np.pi)
    env.quaternion[idx] = [np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)]
    env.target_position[idx] = [rnd(-1.0, 1.0), rnd(-1.0, 1.0), rnd(0.45, 0.9)]
    env.target_yaw[idx] = rnd(-np.pi, np.pi)
    env.physics_params[idx] = sample_physics(
        env.physics_profile,
        env.domain_randomization,
        np.random.default_rng(int(rng)),
        1,
        action_mode=env.action_mode,
    )[0]
    env.thrust_state[idx] = 1.0
    env.previous_action[idx] = 0.0
    env.step_count[idx] = 0
    env.rewards[idx] = 0.0
    env.terminals[idx] = 0
    env.truncations[idx] = 0
    return rng
