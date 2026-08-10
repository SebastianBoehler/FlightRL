from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np
import torch

from .range_env import RangeExplorationEnv
from .range_mapper import RangePose
from .range_policy import RangeExplorationActorCritic
from .range_world import RangeWorld


@dataclass(frozen=True, slots=True)
class RangeObstacleChallengeCase:
    name: str
    world: RangeWorld
    initial_pose: RangePose


def range_obstacle_challenge_cases() -> tuple[RangeObstacleChallengeCase, ...]:
    occupied = RangeWorld.open_room().occupied.copy()
    occupied[28:36, 28:36] = True
    world = RangeWorld(occupied)
    return (
        RangeObstacleChallengeCase("west", world, RangePose(2.15, 3.20, 0.0)),
        RangeObstacleChallengeCase("east", world, RangePose(4.25, 3.20, pi)),
        RangeObstacleChallengeCase("south", world, RangePose(3.20, 2.15, pi / 2.0)),
        RangeObstacleChallengeCase("north", world, RangePose(3.20, 4.25, -pi / 2.0)),
    )


def evaluate_range_obstacle_challenge(
    model: RangeExplorationActorCritic,
    *,
    horizon: int,
) -> dict[str, object]:
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("range obstacle challenge horizon must be positive")
    episodes = []
    model.eval()
    for index, case in enumerate(range_obstacle_challenge_cases()):
        env = RangeExplorationEnv(
            seed=47_000 + index,
            maximum_episode_steps=horizon,
            stress=False,
            world=case.world,
            initial_pose=case.initial_pose,
        )
        observation, _info = env.reset(seed=47_000 + index)
        minimum_front_m = float(observation[4096]) * 4.0
        maximum_front_m = minimum_front_m
        path_m = 0.0
        collision = False
        safety_terminal = False
        for _step in range(horizon):
            before = env.truth_pose
            with torch.no_grad():
                action, _value = model.forward_step(
                    torch.from_numpy(observation[None, :])
                )
            observation, _reward, terminated, truncated, info = env.step(
                action[0].cpu().numpy()
            )
            after = env.truth_pose
            path_m += float(np.hypot(after.x_m - before.x_m, after.y_m - before.y_m))
            front_m = float(observation[4096]) * 4.0
            minimum_front_m = min(minimum_front_m, front_m)
            maximum_front_m = max(maximum_front_m, front_m)
            collision = collision or bool(info["collision"])
            safety_terminal = safety_terminal or bool(info["safety_terminal"])
            if terminated or truncated:
                break
        challenged = minimum_front_m <= 0.66
        escaped = challenged and maximum_front_m >= 0.85 and path_m >= 0.50
        episodes.append(
            {
                "case": case.name,
                "minimum_front_range_m": minimum_front_m,
                "maximum_front_range_m": maximum_front_m,
                "path_length_m": path_m,
                "front_challenge": challenged,
                "escaped_challenge": escaped,
                "collision": collision,
                "safety_terminal": safety_terminal,
            }
        )
    return {
        "episodes": episodes,
        "challenge_rate": float(np.mean([row["front_challenge"] for row in episodes])),
        "escape_rate": float(np.mean([row["escaped_challenge"] for row in episodes])),
        "collision_rate": float(np.mean([row["collision"] for row in episodes])),
        "safety_terminal_rate": float(
            np.mean([row["safety_terminal"] for row in episodes])
        ),
        "actor_observation_contains_truth": False,
    }
