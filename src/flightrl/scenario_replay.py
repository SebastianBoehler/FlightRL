"""Bounded native camera recordings; diagnostic scripts, never flight authority."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from flightrl import _binding
from flightrl.artifact_identity import sha256_file
from flightrl.scenario_bundle import load_scenario_bundle
from flightrl.sixdof.native import native_step


SCHEMA = "flightrl.scenario_replay.v1"
CAMERA = {
    "shape_hw": [48, 64],
    "encoding": "gray4_expanded_uint8_multiples_of_17",
    "vertical_fov_rad": 1.099557429,
    "body_offset_m": [0.035, 0.0, 0.012],
    "optical_axes": "forward=body_x,right=-body_y,down=-body_z",
    "origin_boundary_clamp_m": 0.005,
    "ray_max_distance_m": 8.0,
    "materials": "native_sixdof_vision_hardcoded_procedural_v1",
    "target_mean": 60.0,
}
STATE_FIELDS = [
    "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s",
    "qw", "qx", "qy", "qz", "wx_rad_s", "wy_rad_s", "wz_rad_s",
    "thrust_state",
]


def capture_scenario_replay(
    scenario_dir: str | Path,
    *,
    actions: np.ndarray,
    initial_position_m: np.ndarray,
    connected: np.ndarray,
    dt_s: float,
    scene_seed: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Capture T intervals/N worlds in preallocated arrays, one native batch per tick.

    Frame/state k precedes action k; final frame/state T has no outgoing action.
    Only frames are sensor observations. States are evaluator-only simulator truth.
    connected is an explicit scripted link observation, not an RF model.
    """
    bundle = load_scenario_bundle(scenario_dir)
    if bundle.arrays["terrain_obstacles"].size:
        raise ValueError("native replay does not support interior obstacles")
    if np.any(bundle.arrays["sensor_parameters"] != 0):
        raise ValueError("native replay supports only ideal sensor parameters")
    if bundle.arrays["mission_steps"].size:
        raise ValueError("native replay does not execute mission rows; use an empty mission")
    _require_array(actions, "actions", np.float32, 3)
    ticks, count, width = actions.shape
    if not (1 <= ticks <= 1000 and 1 <= count <= 64 and width == 4):
        raise ValueError("actions must have shape (1..1000,1..64,4)")
    if np.any(np.abs(actions) > 1):
        raise ValueError("actions must be normalized within [-1,1]")
    _require_array(initial_position_m, "initial_position_m", np.float32, 2)
    if initial_position_m.shape != (count, 3):
        raise ValueError("initial_position_m must have shape (N,3)")
    _require_array(connected, "connected", np.bool_, 2)
    if connected.shape != (ticks + 1, count):
        raise ValueError("connected must have shape (T+1,N)")
    if isinstance(dt_s, bool) or not np.isfinite(dt_s) or not 0 < dt_s <= 0.05:
        raise ValueError("dt_s must be finite and within (0,0.05]")
    if type(scene_seed) is not int or not 0 <= scene_seed <= 2147483647:
        raise ValueError("scene_seed must be a nonnegative int32")
    room = bundle.arrays["terrain_bounds"]
    if np.any(initial_position_m <= room[:6:2]) or np.any(initial_position_m >= room[1:6:2]):
        raise ValueError("initial position must be strictly inside the room")
    if _binding.core_abi_version() != 1:
        raise ValueError("unsupported native core ABI")

    position = initial_position_m.copy()
    velocity = np.zeros((count, 3), dtype=np.float32)
    quaternion = np.zeros((count, 4), dtype=np.float32)
    quaternion[:, 0] = 1
    rates = np.zeros((count, 3), dtype=np.float32)
    ranges = np.empty((count, 6), dtype=np.float32)
    thrust = np.ones(count, dtype=np.float32)
    physics = np.repeat(bundle.arrays["vehicle_physics"][None, :], count, axis=0)
    means = np.full(count, CAMERA["target_mean"], dtype=np.float32)
    seeds = np.full(count, scene_seed, dtype=np.int32)
    frames = np.empty((ticks + 1, count, 48, 64), dtype=np.uint8)
    states = np.empty((ticks + 1, count, 14), dtype="<f4")
    for tick in range(ticks + 1):
        if not np.isfinite(position).all() or np.any(position <= room[:6:2]) or np.any(position >= room[1:6:2]):
            raise ValueError("native replay left the supported room interior")
        states[tick, :, :3] = position
        states[tick, :, 3:6] = velocity
        states[tick, :, 6:10] = quaternion
        states[tick, :, 10:13] = rates
        states[tick, :, 13] = thrust
        _binding.sixdof_render_gray4(position, quaternion, room, means, seeds, frames[tick])
        if tick < ticks:
            native_step(position, velocity, quaternion, rates, ranges, actions[tick],
                        dt_s, room, thrust, physics)
    if not np.isfinite(states).all():
        raise ValueError("native replay produced nonfinite states")
    arrays = {
        "time_s": np.arange(ticks + 1, dtype="<f8") * dt_s,
        "states_truth": states,
        "actions": np.array(actions, dtype="<f4", copy=True),
        "frames_local": frames,
        "connected": connected.copy(),
    }
    metadata = {
        "schema": SCHEMA,
        "authority": "simulation_only",
        "deployment_authority": False,
        "scenario_sha256": bundle.manifest["sha256"],
        "policy": None,
        "controller": "open_loop_action_array_diagnostic",
        "mission_execution": False,
        "state_source": "simulator_truth_evaluator_only_no_estimator",
        "link_model": "scripted_observed_status_no_rf_physics",
        "clock": "simulation_seconds_frame_k_before_action_k",
        "state_fields": STATE_FIELDS,
        "action_fields": ["thrust", "roll_rate", "pitch_rate", "yaw_rate"],
        "action_units": "normalized_native_sixdof",
        "frames": dict(bundle.manifest["frames"]),
        "camera": CAMERA,
        "scene_seed": scene_seed,
        "dt_s": float(dt_s),
        "ticks": ticks,
        "num_envs": count,
        "core_abi": 1,
        "native_binary_sha256": sha256_file(_binding.__file__),
        "recorder_sha256": sha256_file(__file__),
        "inspection_evaluation": "not_implemented",
    }
    return metadata, arrays


def operator_frame(arrays: dict[str, np.ndarray], tick: int, env: int) -> np.ndarray | None:
    """No access to a newly captured operator frame while disconnected."""
    if not 0 <= tick < len(arrays["connected"]) or not 0 <= env < arrays["connected"].shape[1]:
        raise IndexError("operator frame index out of bounds")
    if not arrays["connected"][tick, env]:
        return None
    frame = arrays["frames_local"][tick, env].copy()
    frame.setflags(write=False)
    return frame


def _require_array(value: np.ndarray, name: str, dtype: object, ndim: int) -> None:
    if not isinstance(value, np.ndarray) or value.dtype != dtype or value.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}D {np.dtype(dtype)} array")
    if not value.flags.c_contiguous or not np.isfinite(value).all():
        raise ValueError(f"{name} must be contiguous and finite")
