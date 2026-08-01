from __future__ import annotations

import ctypes

import numpy as np

from flightrl.mujoco.semantic_training import (
    SemanticTrainingEnv,
    SimulatedSemanticDetectorConfig,
)
from flightrl.vision import VisionObservationConfig


class SemanticPufferDriver:
    """Pointer-backed Python vector driver for PufferLib 4's PyTorch trainer."""

    gpu = False
    obs_dtype = "FloatTensor"

    def __init__(
        self,
        *,
        room_seeds: tuple[int, ...],
        agents_per_room: int,
        seed: int,
        detector: SimulatedSemanticDetectorConfig | None = None,
        active_exploration: bool = False,
        vision_width: int = 64,
        vision_height: int = 48,
        room_profile: str = "standard",
    ) -> None:
        if not room_seeds or agents_per_room <= 0:
            raise ValueError(
                "room_seeds and agents_per_room must be non-empty and positive"
            )
        self.envs = tuple(
            SemanticTrainingEnv(
                room_seed=room_seed,
                num_envs=agents_per_room,
                seed=seed + index,
                detector=detector,
                active_exploration=active_exploration,
                vision_config=VisionObservationConfig(
                    width=vision_width,
                    height=vision_height,
                    color_mode="grayscale",
                    frame_stack=1,
                    include_delta=True,
                    include_motion_mask=True,
                    normalization="minus_one_one",
                ),
                room_profile=room_profile,
            )
            for index, room_seed in enumerate(room_seeds)
        )
        self.agents_per_room = int(agents_per_room)
        self.total_agents = len(self.envs) * self.agents_per_room
        self.obs_size = int(self.envs[0].single_observation_space.shape[0])
        self.num_atns = int(self.envs[0].single_action_space.shape[0])
        self.observations = np.empty(
            (self.total_agents, self.obs_size), dtype=np.float32
        )
        self.rewards = np.zeros(self.total_agents, dtype=np.float32)
        self.terminals = np.zeros(self.total_agents, dtype=np.float32)
        self.obs_ptr = self.observations.ctypes.data
        self.rewards_ptr = self.rewards.ctypes.data
        self.terminals_ptr = self.terminals.ctypes.data
        self._episode_logs: list[dict[str, float]] = []
        self.reset()

    @property
    def driver_env(self):
        return self.envs[0]

    def reset(self) -> None:
        for index, env in enumerate(self.envs):
            observations, _ = env.reset()
            self._slice(index, self.observations)[:] = observations
        self.rewards.fill(0.0)
        self.terminals.fill(0.0)
        self._episode_logs.clear()

    def expert_actions(self) -> np.ndarray:
        return np.concatenate([env.expert_actions() for env in self.envs], axis=0)

    def target_observed(self) -> np.ndarray:
        return np.concatenate([env.target_observed for env in self.envs], axis=0)

    def target_visible(self) -> np.ndarray:
        return np.concatenate([env.target_visible for env in self.envs], axis=0)

    def front_clearance(self) -> np.ndarray:
        return np.concatenate([env.front_clearance() for env in self.envs], axis=0)

    def action_corridor_clearance(self) -> np.ndarray:
        return np.concatenate(
            [env.action_corridor_clearance() for env in self.envs],
            axis=0,
        )

    def horizontal_clearance(self) -> np.ndarray:
        return np.concatenate(
            [env.horizontal_clearance() for env in self.envs],
            axis=0,
        )

    def navigation_clearance(self) -> np.ndarray:
        return np.concatenate(
            [env.navigation_clearance() for env in self.envs],
            axis=0,
        )

    def cpu_step(self, actions_ptr: int) -> None:
        buffer_type = ctypes.c_float * (self.total_agents * self.num_atns)
        actions = np.ctypeslib.as_array(buffer_type.from_address(actions_ptr)).reshape(
            self.total_agents,
            self.num_atns,
        )
        self._step(actions, write_observations=True)

    def teacher_step(self, actions: np.ndarray) -> None:
        self._step(actions, write_observations=False)

    def _step(self, actions: np.ndarray, *, write_observations: bool) -> None:
        for index, env in enumerate(self.envs):
            observations, rewards, terminals, truncations, infos = env.step(
                self._slice(index, actions),
                write_observations=write_observations,
            )
            if write_observations:
                self._slice(index, self.observations)[:] = observations
            self._slice(index, self.rewards)[:] = rewards
            self._slice(index, self.terminals)[:] = np.asarray(terminals) | np.asarray(
                truncations
            )
            self._episode_logs.extend(infos)

    def log(self) -> dict[str, float]:
        count = sum(entry.get("n", 0.0) for entry in self._episode_logs)
        if count == 0:
            return {}
        keys = set().union(*(entry.keys() for entry in self._episode_logs)) - {"n"}
        result = {
            key: sum(
                entry.get(key, 0.0) * entry.get("n", 0.0)
                for entry in self._episode_logs
            )
            / count
            for key in keys
        }
        result["n"] = count
        self._episode_logs.clear()
        return result

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def _slice(self, index: int, values: np.ndarray) -> np.ndarray:
        start = index * self.agents_per_room
        return values[start : start + self.agents_per_room]
