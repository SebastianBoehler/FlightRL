from __future__ import annotations

import gymnasium
import numpy as np

from flightrl.mujoco.semantic_vision_env import MuJoCoSemanticVisionEnv


class MuJoCoSemanticVisionGymEnv(gymnasium.Env):
    """Scalar Gymnasium adapter for PufferLib emulation/vectorization."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self.seed_value = int(seed)
        self.backend = MuJoCoSemanticVisionEnv(
            num_envs=1,
            seed=self.seed_value,
            auto_reset=False,
        )
        self.observation_space = self.backend.single_observation_space
        self.action_space = self.backend.single_action_space

    def reset(self, *, seed: int | None = None, options=None):
        del options
        super().reset(seed=seed)
        observations, _ = self.backend.reset(
            self.seed_value if seed is None else seed
        )
        return observations[0].copy(), {}

    def step(self, action):
        observations, rewards, terminals, truncations, infos = self.backend.step(
            np.asarray(action, dtype=np.float32).reshape(1, 4)
        )
        info = {} if not infos else dict(infos[0])
        return (
            observations[0].copy(),
            float(rewards[0]),
            bool(terminals[0]),
            bool(truncations[0]),
            info,
        )

    def render(self):
        return self.backend.sim.render_rgb(
            width=320,
            height=240,
            camera="aideck",
        )

    def close(self) -> None:
        self.backend.close()
