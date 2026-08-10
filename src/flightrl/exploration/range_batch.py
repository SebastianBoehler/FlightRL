from __future__ import annotations

import numpy as np

from .range_contract import RANGE_ACTION_DIM
from .range_env import RangeExplorationEnv


class RangeExplorationBatch:
    def __init__(
        self,
        *,
        num_envs: int,
        seed: int,
        maximum_episode_steps: int = 1_200,
        stress: bool = True,
    ) -> None:
        if type(num_envs) is not int or num_envs <= 0:
            raise ValueError("range exploration batch size must be positive")
        self.num_envs = num_envs
        self.seed = seed
        self.envs = [
            RangeExplorationEnv(
                seed=seed + index,
                maximum_episode_steps=maximum_episode_steps,
                stress=stress,
            )
            for index in range(num_envs)
        ]
        self.observations = np.stack(
            [env._last_observation for env in self.envs]
        ).astype(np.float32)

    def reset(self, *, seed: int | None = None) -> np.ndarray:
        base = self.seed if seed is None else seed
        self.observations = np.stack(
            [env.reset(seed=base + index)[0] for index, env in enumerate(self.envs)]
        ).astype(np.float32)
        return self.observations.copy()

    def step(self, actions: np.ndarray):
        values = np.asarray(actions, dtype=np.float32)
        if values.shape != (self.num_envs, RANGE_ACTION_DIM):
            raise ValueError("batched range actions have the wrong shape")
        results = [env.step(values[index]) for index, env in enumerate(self.envs)]
        observations, rewards, terminated, truncated, infos = zip(*results, strict=True)
        self.observations = np.stack(observations).astype(np.float32)
        return (
            self.observations.copy(),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terminated, dtype=bool),
            np.asarray(truncated, dtype=bool),
            list(infos),
        )

    def reset_done(self, mask: np.ndarray, *, seed: int) -> np.ndarray:
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != (self.num_envs,):
            raise ValueError("range reset mask has the wrong shape")
        for index in np.flatnonzero(selected):
            self.observations[index] = self.envs[int(index)].reset(
                seed=seed + int(index)
            )[0]
        return self.observations.copy()
