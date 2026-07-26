from __future__ import annotations

from typing import Sequence

import gymnasium
import numpy as np

from . import _binding
from .binding_kwargs import build_binding_kwargs
from .config import FlightConfig
from .renderer import DroneFrame, FlightRenderer
from .vision import VisionObservationBatchEncoder


class DronePlanarEnv:
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        config: FlightConfig,
        num_envs: int | None = None,
        buf=None,
        seed: int = 0,
        emit_logs: bool = True,
        render_mode: str | None = None,
    ) -> None:
        if buf is not None:
            raise NotImplementedError("custom shared buffers are not supported by the local FlightRL wrapper")
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError(f"unsupported render mode: {render_mode}")

        self.config = config
        self.num_agents = num_envs or config.environment.num_envs
        self.render_mode = render_mode
        self.single_observation_space = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(config.observation_dim,),
            dtype=np.float32,
        )
        self.single_action_space = gymnasium.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(config.action_dim,),
            dtype=np.float32,
        )
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space

        self.observations = np.zeros((self.num_agents, config.observation_dim), dtype=np.float32)
        self.actions = np.zeros((self.num_agents, config.action_dim), dtype=np.float32)
        self.rewards = np.zeros(self.num_agents, dtype=np.float32)
        self.terminals = np.zeros(self.num_agents, dtype=np.uint8)
        self.truncations = np.zeros(self.num_agents, dtype=np.uint8)
        self._vision_encoder = (
            VisionObservationBatchEncoder(config.vision, self.num_agents)
            if config.sensors.include_vision_sensor
            else None
        )
        self._vision_values = (
            np.zeros((self.num_agents, config.vision.flat_dim), dtype=np.float32)
            if config.sensors.include_vision_sensor
            else None
        )
        self._vision_ready = self._vision_encoder is None

        self._emit_logs = emit_logs
        self._handles: list[int] = []
        self._renderer: FlightRenderer | None = None
        self._report_interval = config.logging.report_interval
        self._tick = 0

        kwargs = build_binding_kwargs(config, host_fed_vision=True)
        for env_idx in range(self.num_agents):
            handle = _binding.env_init(
                self.observations[env_idx : env_idx + 1],
                self.actions[env_idx : env_idx + 1],
                self.rewards[env_idx : env_idx + 1],
                self.terminals[env_idx : env_idx + 1],
                self.truncations[env_idx : env_idx + 1],
                seed + env_idx,
                **kwargs,
            )
            self._handles.append(handle)
        self._vec_handle = _binding.vectorize(*self._handles)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, list[dict[str, float]]]:
        self._tick = 0
        _binding.vec_reset(self._vec_handle, 0 if seed is None else seed)
        if self._vision_encoder is not None and self._vision_values is not None:
            self._vision_encoder.reset()
            self._vision_values.fill(0.0)
            self._vision_ready = False
            self._apply_vision_observation()
        return self.observations, []

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]]]:
        if not self._vision_ready:
            raise RuntimeError("set_vision_frames() or set_vision_observations() is required before stepping")
        self._tick += 1
        self.actions[:] = np.asarray(actions, dtype=np.float32)
        _binding.vec_step(self._vec_handle)
        self._apply_vision_observation()
        info: list[dict[str, float]] = []
        if self._emit_logs and self._tick % self._report_interval == 0:
            log = _binding.vec_log(self._vec_handle)
            if log:
                info.append(log)
        return self.observations, self.rewards, self.terminals, self.truncations, info

    @property
    def vision_observation_shape(self) -> tuple[int, int, int] | None:
        return self.config.vision.shape if self._vision_encoder is not None else None

    def set_vision_frames(self, frames: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
        if self._vision_encoder is None:
            raise RuntimeError("vision observations are not enabled in this environment")
        batch = _as_frame_batch(frames, self.num_agents)
        values = self._vision_encoder.encode(batch)
        self.set_vision_observations(values)
        return values.reshape((self.num_agents, *self.config.vision.shape))

    def set_vision_observations(self, values: np.ndarray) -> None:
        if self._vision_values is None:
            raise RuntimeError("vision observations are not enabled in this environment")
        array = np.asarray(values, dtype=np.float32)
        expected = (self.num_agents, self.config.vision.flat_dim)
        if array.shape == (self.num_agents, *self.config.vision.shape):
            array = array.reshape(expected)
        if array.shape != expected:
            raise ValueError(f"expected vision observations with shape {expected}, got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError("vision observations contain non-finite values")
        self._vision_values[:] = array
        self._vision_ready = True
        self._apply_vision_observation()

    def snapshot(self, env_index: int = 0) -> dict[str, float]:
        return _binding.env_get(self._handles[env_index])

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            raise ValueError("render_mode is not enabled for this environment")
        frame = self._snapshot_frame()
        if self._renderer is None:
            fps = float(self.metadata.get("render_fps", 30))
            self._renderer = FlightRenderer(self.config, self.render_mode, fps=fps)
        return self._renderer.render(frame)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if hasattr(self, "_vec_handle"):
            _binding.vec_close(self._vec_handle)

    def _snapshot_frame(self, env_index: int = 0) -> DroneFrame:
        snapshot = self.snapshot(env_index)
        return DroneFrame(
            x=snapshot["x"],
            z=snapshot["z"],
            vx=snapshot["vx"],
            vz=snapshot["vz"],
            ax=snapshot["ax"],
            az=snapshot["az"],
            pitch=snapshot["pitch"],
            pitch_rate=snapshot["pitch_rate"],
            target_x=snapshot["target_x"],
            target_z=snapshot["target_z"],
            wind_x=snapshot["wind_x"],
            wind_z=snapshot["wind_z"],
            distance=snapshot["distance"],
            reward_total=snapshot["reward_total"],
            motor_thrusts=(
                snapshot["motor_front_left"],
                snapshot["motor_front_right"],
                snapshot["motor_rear_left"],
                snapshot["motor_rear_right"],
            ),
            commands=(
                snapshot["command_0"],
                snapshot["command_1"],
                snapshot["command_2"],
                snapshot["command_3"],
            ),
            action_dim=int(snapshot["action_dim"]),
            active_target=int(snapshot["active_target"]),
            target_count=int(snapshot["target_count"]),
        )

    def _apply_vision_observation(self) -> None:
        if self._vision_values is not None:
            self.observations[:, self.config.vision_slice] = self._vision_values


def _as_frame_batch(frames: Sequence[np.ndarray] | np.ndarray, batch_size: int) -> tuple[np.ndarray, ...]:
    if isinstance(frames, np.ndarray) and batch_size == 1 and frames.ndim in (2, 3):
        return (frames,)
    if isinstance(frames, np.ndarray):
        if frames.shape[0] != batch_size:
            raise ValueError(f"expected {batch_size} vision frames, got array shape {frames.shape}")
        return tuple(frames[index] for index in range(batch_size))
    batch = tuple(frames)
    if len(batch) != batch_size:
        raise ValueError(f"expected {batch_size} vision frames, got {len(batch)}")
    return batch
