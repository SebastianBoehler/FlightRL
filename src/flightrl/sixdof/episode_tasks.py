from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def task_probability_vector(
    tasks: tuple[str, ...],
    task_probabilities: dict[str, float] | None = None,
) -> np.ndarray:
    _validate_tasks(tasks)
    weights = np.ones(len(tasks), dtype=np.float64)
    if task_probabilities:
        unknown = sorted(set(task_probabilities) - set(tasks))
        if unknown:
            raise ValueError(f"unknown task probability weight(s): {', '.join(unknown)}")
        for task, weight in task_probabilities.items():
            if not np.isfinite(weight) or weight <= 0:
                raise ValueError("task probability weights must be positive")
            weights[tasks.index(task)] = float(weight)
    return _normalize_probabilities(tasks, weights)


def sample_task_indices(
    rng: np.random.Generator,
    num_envs: int,
    tasks: tuple[str, ...],
    probabilities: np.ndarray | None = None,
) -> np.ndarray:
    _validate_tasks(tasks)
    if type(num_envs) is not int or num_envs <= 0:
        raise ValueError("task sample count must be a positive integer")
    normalized = (
        None
        if probabilities is None
        else _normalize_probabilities(tasks, probabilities)
    )
    if len(tasks) == 1:
        return np.zeros(num_envs, dtype=np.int64)
    if normalized is not None:
        return rng.choice(len(tasks), size=num_envs, p=normalized).astype(np.int64)
    return rng.integers(0, len(tasks), size=num_envs, dtype=np.int64)


@dataclass(slots=True)
class EpisodeTaskAssignments:
    tasks: tuple[str, ...]
    probabilities: np.ndarray
    rng: np.random.Generator
    indices: np.ndarray

    @classmethod
    def sample(
        cls,
        *,
        rng: np.random.Generator,
        num_envs: int,
        tasks: tuple[str, ...],
        probabilities: np.ndarray | None = None,
    ) -> EpisodeTaskAssignments:
        resolved = (
            task_probability_vector(tasks)
            if probabilities is None
            else _normalize_probabilities(tasks, probabilities)
        )
        return cls(
            tasks=tasks,
            probabilities=resolved,
            rng=rng,
            indices=sample_task_indices(rng, num_envs, tasks, resolved),
        )

    def resample(self, done: np.ndarray) -> None:
        mask = np.asarray(done, dtype=bool)
        if mask.shape != self.indices.shape:
            raise ValueError("done mask must match episode task assignments")
        count = int(np.sum(mask))
        if count:
            self.indices[mask] = sample_task_indices(
                self.rng,
                count,
                self.tasks,
                self.probabilities,
            )

    def resample_all(self) -> None:
        self.indices[:] = sample_task_indices(
            self.rng,
            len(self.indices),
            self.tasks,
            self.probabilities,
        )

    def apply(self, env, *, reward_mode: str = "env") -> np.ndarray:
        if hasattr(env, "set_native_context"):
            env.set_native_context(
                task_indices=self.indices,
                tasks=self.tasks,
                reward_mode=reward_mode,
            )
        return env.observations.copy()


def _validate_tasks(tasks: tuple[str, ...]) -> None:
    if not tasks:
        raise ValueError("episode task set cannot be empty")
    if len(set(tasks)) != len(tasks):
        raise ValueError("episode task set cannot contain duplicates")


def _normalize_probabilities(
    tasks: tuple[str, ...],
    probabilities: np.ndarray,
) -> np.ndarray:
    _validate_tasks(tasks)
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (len(tasks),):
        raise ValueError("task probabilities must match the task set")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("task probabilities must be finite and nonnegative")
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("task probabilities must have positive sum")
    return values / total
