from __future__ import annotations

from math import pi

import numpy as np

from .range_batch import RangeExplorationBatch
from .range_env import RangeExplorationEnv
from .range_mapper import RangePose
from .range_world import RangeWorld


_APPROACH_SIDES = ("west", "east", "south", "north")


class RangeObstacleTrainingBatch(RangeExplorationBatch):
    def __init__(self, *, num_envs: int, seed: int) -> None:
        if type(num_envs) is not int or num_envs <= 0 or type(seed) is not int:
            raise ValueError("obstacle training batch size and seed are invalid")
        rng = np.random.default_rng(seed)
        self.num_envs = num_envs
        self.seed = seed
        self.approach_sides = tuple(
            _APPROACH_SIDES[index % len(_APPROACH_SIDES)]
            for index in range(num_envs)
        )
        self.envs = [
            _training_environment(rng, seed + index, side)
            for index, side in enumerate(self.approach_sides)
        ]
        self.observations = np.stack(
            [env._last_observation for env in self.envs]
        ).astype(np.float32)


def _training_environment(
    rng: np.random.Generator,
    seed: int,
    side: str,
) -> RangeExplorationEnv:
    occupied = RangeWorld.open_room().occupied.copy()
    height = int(rng.integers(4, 12))
    width = int(rng.integers(4, 12))
    row = int(rng.integers(16, 38))
    column = int(rng.integers(16, 38))
    occupied[row : row + height, column : column + width] = True
    world = RangeWorld(occupied)
    distance = float(rng.uniform(0.60, 0.70))
    lateral = float(rng.uniform(-0.15, 0.15))
    center_x = (column + width / 2.0) * 0.10
    center_y = (row + height / 2.0) * 0.10
    poses = {
        "west": RangePose(column * 0.10 - distance, center_y + lateral, 0.0),
        "east": RangePose((column + width) * 0.10 + distance, center_y + lateral, pi),
        "south": RangePose(center_x + lateral, row * 0.10 - distance, pi / 2.0),
        "north": RangePose(
            center_x + lateral,
            (row + height) * 0.10 + distance,
            -pi / 2.0,
        ),
    }
    pose = poses[side]
    if world.collides(pose.x_m, pose.y_m):
        raise RuntimeError("generated obstacle curriculum start is not collision-free")
    return RangeExplorationEnv(
        seed=seed,
        maximum_episode_steps=300,
        stress=False,
        world=world,
        initial_pose=pose,
    )
