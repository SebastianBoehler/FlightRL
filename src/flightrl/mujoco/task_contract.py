from __future__ import annotations

import numpy as np

from flightrl.sixdof.env import REWARD_MODE_IDS, TASK_IDS, quat_to_yaw
from flightrl.sixdof.sensor_model import noisy_values, observed_ranges
from flightrl.sixdof.task_context import (
    default_task_reward,
    task_target_yaw,
    wrap_angle,
    write_yaw_observation,
)
from flightrl.sixdof.validation import finite_batch, task_id_batch


def build_observation(env) -> np.ndarray:
    position = noisy_values(
        env.position,
        env.sensor_profile.state_noise_std_m,
        env.rng,
    )
    velocity = noisy_values(
        env.velocity,
        env.sensor_profile.velocity_noise_std_m_s,
        env.rng,
    )
    body_rates = noisy_values(
        env.body_rates,
        env.sensor_profile.body_rate_noise_std_rad_s,
        env.rng,
    )
    ranges = observed_ranges(
        env.ranges_m,
        max_range_m=env.room.max_range_m,
        profile=env.sensor_profile,
        rng=env.rng,
    )
    yaw_error = wrap_angle(env.observation_target_yaw() - quat_to_yaw(env.quaternion))
    observation = np.concatenate(
        [
            (env.target_position - position)
            / np.asarray([2.0, 2.0, 1.5], dtype=np.float32),
            velocity / 3.0,
            env.quaternion,
            body_rates / env.max_rate,
            env.target_position
            / np.asarray([2.0, 2.0, 2.5], dtype=np.float32),
            np.sin(yaw_error)[:, None],
            np.cos(yaw_error)[:, None],
            ranges / env.room.max_range_m,
            env.previous_action,
        ],
        axis=1,
    )
    return observation.astype(np.float32)


def apply_task_context(
    env,
    *,
    task_indices: np.ndarray | None = None,
    tasks: tuple[str, ...] | None = None,
    reward_mode: str = "env",
    previous_error: np.ndarray | None = None,
) -> None:
    if reward_mode not in REWARD_MODE_IDS:
        raise ValueError(f"unknown MuJoCo reward mode {reward_mode!r}")
    if task_indices is None:
        env.native_task_ids.fill(TASK_IDS[env.task])
    else:
        task_names = tasks or (env.task,)
        env.native_task_ids[:] = task_id_batch(
            task_indices,
            task_names,
            num_envs=env.num_envs,
            task_ids=TASK_IDS,
        )
    env.native_reward_mode_id = REWARD_MODE_IDS[reward_mode]
    if previous_error is None:
        env.native_previous_error.fill(0.0)
    else:
        env.native_previous_error[:] = finite_batch(
            previous_error,
            "previous error",
            env.num_envs,
        )
    write_yaw_observation(
        env.observations,
        target_yaw_for_env(env),
        quat_to_yaw(env.quaternion),
    )


def target_yaw_for_env(env) -> np.ndarray:
    return task_target_yaw(
        env.position,
        env.target_position,
        env.target_yaw,
        env.native_task_ids,
    )


def reward_for_env(env, actions: np.ndarray) -> np.ndarray:
    return default_task_reward(env, actions, quat_to_yaw(env.quaternion))
