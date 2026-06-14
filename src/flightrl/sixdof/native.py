from __future__ import annotations

import numpy as np

from flightrl import _binding
from flightrl.sixdof.physics import LEGACY_PHYSICS


def native_step(
    position: np.ndarray,
    velocity: np.ndarray,
    quaternion: np.ndarray,
    body_rates: np.ndarray,
    ranges_m: np.ndarray,
    actions: np.ndarray,
    dt: float,
    room_bounds: np.ndarray | None = None,
    thrust_state: np.ndarray | None = None,
    physics_params: np.ndarray | None = None,
) -> None:
    num_envs = position.shape[0]
    resolved_thrust = thrust_state if thrust_state is not None else np.ones(num_envs, dtype=np.float32)
    resolved_physics = physics_params if physics_params is not None else np.repeat(LEGACY_PHYSICS.as_array()[None, :], num_envs, axis=0)
    _binding.sixdof_step(
        _float32(position),
        _float32(velocity),
        _float32(quaternion),
        _float32(body_rates),
        _float32(ranges_m),
        _float32(resolved_thrust),
        _float32(actions),
        _float32(resolved_physics),
        _float32(room_bounds if room_bounds is not None else DEFAULT_ROOM_BOUNDS),
        float(dt),
    )


def native_step_env(env, actions: np.ndarray) -> None:
    if not env.native_context_required:
        _binding.sixdof_step_env(
            _float32(env.position),
            _float32(env.velocity),
            _float32(env.quaternion),
            _float32(env.body_rates),
            _float32(env.ranges_m),
            _float32(env.thrust_state),
            _float32(env.physics_params),
            _float32(env.target_position),
            _float32(env.target_yaw),
            _float32(env.previous_action),
            _int32(env.step_count),
            _float32(actions),
            _float32(env.observations),
            _float32(env.rewards),
            _uint8(env.terminals),
            _uint8(env.truncations),
            _float32(env.room_bounds),
            float(env.dt),
        )
        return
    _binding.sixdof_step_env_context(
        _float32(env.position),
        _float32(env.velocity),
        _float32(env.quaternion),
        _float32(env.body_rates),
        _float32(env.ranges_m),
        _float32(env.thrust_state),
        _float32(env.physics_params),
        _float32(env.target_position),
        _float32(env.target_yaw),
        _float32(env.previous_action),
        _int32(env.step_count),
        _float32(actions),
        _float32(env.observations),
        _float32(env.rewards),
        _uint8(env.terminals),
        _uint8(env.truncations),
        _float32(env.room_bounds),
        _int32(env.native_task_ids),
        int(env.native_reward_mode_id),
        _float32(env.native_previous_error),
        float(env.dt),
    )


def _float32(values: np.ndarray) -> np.ndarray:
    if values.dtype == np.float32 and values.flags.c_contiguous:
        return values
    raise ValueError("native 6-DoF arrays must be C-contiguous float32")


def _int32(values: np.ndarray) -> np.ndarray:
    if values.dtype == np.int32 and values.flags.c_contiguous:
        return values
    raise ValueError("native 6-DoF integer arrays must be C-contiguous int32")


def _uint8(values: np.ndarray) -> np.ndarray:
    if values.dtype == np.uint8 and values.flags.c_contiguous:
        return values
    raise ValueError("native 6-DoF flag arrays must be C-contiguous uint8")


DEFAULT_ROOM_BOUNDS = np.asarray([-2.0, 2.0, -2.0, 2.0, 0.0, 2.5, 4.0], dtype=np.float32)
