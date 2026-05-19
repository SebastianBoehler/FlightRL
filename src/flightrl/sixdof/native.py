from __future__ import annotations

import numpy as np

from flightrl import _binding


def native_step(
    position: np.ndarray,
    velocity: np.ndarray,
    quaternion: np.ndarray,
    body_rates: np.ndarray,
    ranges_m: np.ndarray,
    actions: np.ndarray,
    dt: float,
) -> None:
    _binding.sixdof_step(
        _float32(position),
        _float32(velocity),
        _float32(quaternion),
        _float32(body_rates),
        _float32(ranges_m),
        _float32(actions),
        float(dt),
    )


def native_step_env(env, actions: np.ndarray) -> None:
    _binding.sixdof_step_env(
        _float32(env.position),
        _float32(env.velocity),
        _float32(env.quaternion),
        _float32(env.body_rates),
        _float32(env.ranges_m),
        _float32(env.target_position),
        _float32(env.target_yaw),
        _float32(env.previous_action),
        _int32(env.step_count),
        _float32(actions),
        _float32(env.observations),
        _float32(env.rewards),
        _uint8(env.terminals),
        _uint8(env.truncations),
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
