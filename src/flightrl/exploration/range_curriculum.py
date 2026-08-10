from __future__ import annotations

from math import atan2, pi

import numpy as np
import torch
import torch.nn.functional as F

from .range_env import RangeExplorationEnv
from .range_observation import build_range_exploration_observation
from .range_policy import RangeExplorationActorCritic
from .range_ppo import frontier_yaw_targets


RANGE_CURRICULUM_SCHEMA = (
    "flightrl.range_exploration.counterfactual_curriculum.v1"
)


def sample_range_counterfactual_batch(
    *,
    seed: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if type(seed) is not int or type(batch_size) is not int or batch_size <= 0:
        raise ValueError("range curriculum seed and batch size must be positive integers")
    if batch_size % 4:
        raise ValueError("range curriculum batch size must be divisible by four")
    rng = np.random.default_rng(seed)
    observations: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for _ in range(batch_size // 4):
        row = int(rng.integers(7, 14))
        column = int(rng.integers(6, 13))
        left_map = _single_frontier_map(row, column)
        right_map = left_map[:, :, ::-1].copy()
        clear_ranges = rng.uniform(0.55, 0.95, size=4).astype(np.float32)
        clear_ranges[0] = float(rng.uniform(0.60, 0.95))
        right_ranges = clear_ranges[[0, 1, 3, 2]]
        previous = np.asarray(
            (rng.uniform(0.0, 0.8), rng.uniform(-0.8, 0.8)),
            dtype=np.float32,
        )
        yaw = float(
            np.clip(
                atan2(16.0 - column, max(0.5, 16.0 - row)) / (pi / 2.0),
                0.15,
                0.90,
            )
        )
        forward = 0.65
        pairs = (
            (left_map, clear_ranges, previous, (forward, yaw)),
            (right_map, right_ranges, previous * (1.0, -1.0), (forward, -yaw)),
            (left_map, _blocked(clear_ranges), previous, (0.0, yaw)),
            (right_map, _blocked(right_ranges), previous * (1.0, -1.0), (0.0, -yaw)),
        )
        for map_crop, ranges, prior, target in pairs:
            observations.append(
                build_range_exploration_observation(
                    map_crop,
                    ranges,
                    np.ones(4, dtype=np.float32),
                    np.asarray(prior, dtype=np.float32),
                )
            )
            targets.append(np.asarray(target, dtype=np.float32))
    return np.stack(observations), np.stack(targets)


def train_range_counterfactual_curriculum(
    model: RangeExplorationActorCritic,
    optimizer: torch.optim.Optimizer,
    *,
    seed: int,
    steps: int,
    batch_size: int,
) -> dict[str, object]:
    if type(steps) is not int or steps <= 0:
        raise ValueError("range curriculum steps must be a positive integer")
    losses = []
    model.train()
    for step in range(steps):
        observations, targets = sample_range_counterfactual_batch(
            seed=seed + step,
            batch_size=batch_size,
        )
        action, _value = model.forward_step(torch.from_numpy(observations))
        loss = F.mse_loss(action, torch.from_numpy(targets))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return {
        "schema": RANGE_CURRICULUM_SCHEMA,
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "selected_frontier_runtime_input": False,
        "labels_are_map_derived_training_only": True,
    }


def collect_range_natural_counterfactual_batch(
    *,
    seed: int,
    base_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if type(seed) is not int or type(base_count) is not int or base_count <= 0:
        raise ValueError("natural range curriculum seed and base count must be positive")
    bases: list[tuple[np.ndarray, float]] = []
    world_seed = seed
    maximum_worlds = base_count * 4
    for _ in range(maximum_worlds):
        env = RangeExplorationEnv(
            seed=world_seed,
            maximum_episode_steps=300,
            stress=False,
        )
        observation, _info = env.reset(seed=world_seed)
        world_seed += 1
        for step in range(300):
            frontier_count = float(
                observation[:4096].reshape(4, 32, 32)[3].sum()
            )
            if step > 20 and step % 4 == 0 and frontier_count >= 3.0:
                target, active = frontier_yaw_targets(
                    torch.from_numpy(observation[None, :])
                )
                yaw = float(target[0])
                if bool(active[0]) and abs(yaw) >= 0.15:
                    bases.append((observation.copy(), yaw))
                    if len(bases) == base_count:
                        return _natural_counterfactual_pairs(bases)
            front_m = float(observation[4096]) * 4.0
            action = np.asarray(
                (0.35, 0.18) if front_m > 0.70 else (0.0, 0.55),
                dtype=np.float32,
            )
            observation, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break
    raise RuntimeError("natural range curriculum could not collect enough mapper states")


def train_range_natural_curriculum(
    model: RangeExplorationActorCritic,
    optimizer: torch.optim.Optimizer,
    observations: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int,
    steps: int,
    batch_size: int,
) -> dict[str, object]:
    values = np.asarray(observations, dtype=np.float32)
    labels = np.asarray(targets, dtype=np.float32)
    if (
        values.ndim != 2
        or values.shape[1] != 4106
        or labels.shape != (len(values), 2)
        or len(values) < batch_size
        or not np.isfinite(values).all()
        or not np.isfinite(labels).all()
    ):
        raise ValueError("natural range curriculum dataset is incompatible")
    if type(steps) is not int or steps <= 0 or type(batch_size) is not int or batch_size <= 0:
        raise ValueError("natural range curriculum steps and batch size must be positive")
    rng = np.random.default_rng(seed)
    losses = []
    model.train()
    for _step in range(steps):
        indices = rng.integers(0, len(values), size=batch_size)
        action, _value = model.forward_step(torch.from_numpy(values[indices]))
        loss = F.mse_loss(action, torch.from_numpy(labels[indices]))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return {
        "schema": RANGE_CURRICULUM_SCHEMA,
        "source": "mapper_rollout",
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "examples": len(values),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "selected_frontier_runtime_input": False,
        "labels_are_map_derived_training_only": True,
    }


def _single_frontier_map(row: int, column: int) -> np.ndarray:
    value = np.zeros((4, 32, 32), dtype=np.float32)
    value[0, 15:18, 15:18] = 1.0
    for alpha in np.linspace(0.0, 1.0, 20):
        path_row = int(round(16.0 + alpha * (row - 16.0)))
        path_column = int(round(16.0 + alpha * (column - 16.0)))
        value[1, path_row, path_column] = 1.0
    value[3, row : row + 2, column : column + 2] = 1.0
    return value


def _blocked(ranges: np.ndarray) -> np.ndarray:
    value = ranges.copy()
    value[0] = 0.05
    return value


def _natural_counterfactual_pairs(
    bases: list[tuple[np.ndarray, float]],
) -> tuple[np.ndarray, np.ndarray]:
    observations: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for base, yaw in bases:
        clear = base.copy()
        clear[4096] = 0.8
        clear[4100] = 1.0
        mirrored = _mirror_observation(clear)
        blocked = clear.copy()
        blocked[4096] = 0.05
        mirrored_blocked = mirrored.copy()
        mirrored_blocked[4096] = 0.05
        observations.extend((clear, mirrored, blocked, mirrored_blocked))
        targets.extend(((0.65, yaw), (0.65, -yaw), (0.0, yaw), (0.0, -yaw)))
    return (
        np.stack(observations).astype(np.float32),
        np.asarray(targets, dtype=np.float32),
    )


def _mirror_observation(observation: np.ndarray) -> np.ndarray:
    value = observation.copy()
    value[:4096] = observation[:4096].reshape(4, 32, 32)[:, :, ::-1].reshape(-1)
    value[4096:4100] = observation[4096:4100][[0, 1, 3, 2]]
    value[4100:4104] = observation[4100:4104][[0, 1, 3, 2]]
    value[4105] = -observation[4105]
    return value
