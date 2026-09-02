from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .env import SixDofEnv
from .episode_tasks import EpisodeTaskAssignments, task_probability_vector
from .observation import OBSERVATION_MODES, augment_observation
from .policies import teacher_actions
from .tasks import append_task_encoding, parse_task_spec, select_task_actions, task_observation_dim


def collect_teacher_dataset(
    *,
    task_spec: str,
    num_envs: int,
    steps: int,
    seed: int,
    use_native_step: bool,
    reset_profile: str | None = None,
    observation_mode: str = "base",
    execution_noise_std: float = 0.0,
    task_probabilities: dict[str, float] | None = None,
) -> dict[str, np.ndarray | dict]:
    if observation_mode not in OBSERVATION_MODES:
        raise ValueError(f"unknown observation mode {observation_mode!r}")
    if execution_noise_std < 0:
        raise ValueError("execution_noise_std must be non-negative")
    tasks = parse_task_spec(task_spec)
    sampling_probabilities = task_probability_vector(tasks, task_probabilities)
    rng = np.random.default_rng(seed)
    env = SixDofEnv(num_envs=num_envs, seed=seed, task=tasks[0], use_native_step=use_native_step, reset_profile=reset_profile)
    env.reset(seed=seed)
    episode_tasks = EpisodeTaskAssignments.sample(
        rng=rng,
        num_envs=num_envs,
        tasks=tasks,
        probabilities=sampling_probabilities,
    )
    obs = episode_tasks.apply(env)
    observations = []
    actions = []
    task_indices_all = []
    terminals = []
    previous_obs = None
    previous_action = np.zeros((num_envs, 4), dtype=np.float32)
    fresh = np.ones(num_envs, dtype=bool)
    for _ in range(steps):
        task_indices = episode_tasks.indices
        obs = episode_tasks.apply(env)
        labels = teacher_labels(env, tasks, task_indices)
        model_obs = append_task_encoding(obs.copy(), task_indices, len(tasks))
        if previous_obs is None:
            previous_obs = model_obs.copy()
        previous_obs[fresh] = model_obs[fresh]
        observations.append(augment_observation(model_obs, previous_obs, previous_action, observation_mode))
        actions.append(labels.copy())
        task_indices_all.append(task_indices.copy())
        executed = execution_actions(labels, rng, execution_noise_std)
        episode_tasks.apply(env)
        obs, _reward, terminal, truncation, _info = env.step(executed)
        terminals.append(terminal.copy())
        previous_obs = model_obs.copy()
        previous_action = executed.copy()
        fresh[:] = False
        done = terminal | truncation
        if np.any(done):
            obs = env.reset_done(done)
            episode_tasks.resample(done)
            obs = episode_tasks.apply(env)
            previous_action[done.astype(bool)] = 0.0
            fresh = done.astype(bool)
    stacked_obs = np.concatenate(observations).astype(np.float32)
    stacked_actions = np.concatenate(actions).astype(np.float32)
    stacked_tasks = np.concatenate(task_indices_all).astype(np.int64)
    stacked_terminals = np.concatenate(terminals).astype(np.uint8)
    metadata = {
        "tasks": list(tasks),
        "task_spec": task_spec,
        "num_envs": num_envs,
        "steps": steps,
        "seed": seed,
        "native_step": use_native_step,
        "reset_profile": env.reset_profile.name,
        "observation_mode": observation_mode,
        "observation_dim": int(stacked_obs.shape[1]),
        "base_observation_dim": 28,
        "action_dim": int(stacked_actions.shape[1]),
        "terminal_fraction": float(np.mean(stacked_terminals)),
        "execution_policy": "noisy_teacher" if execution_noise_std > 0 else "teacher",
        "execution_noise_std": execution_noise_std,
        "task_probability_weights": task_probabilities or {},
        "task_sampling_probabilities": {task: float(probability) for task, probability in zip(tasks, sampling_probabilities, strict=True)},
    }
    return {
        "observations": stacked_obs,
        "actions": stacked_actions,
        "task_indices": stacked_tasks,
        "terminals": stacked_terminals,
        "metadata": metadata,
    }


def write_dataset(path: str | Path, dataset: dict[str, np.ndarray | dict]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        observations=dataset["observations"],
        actions=dataset["actions"],
        task_indices=dataset["task_indices"],
        terminals=dataset["terminals"],
        metadata=json.dumps(dataset["metadata"], sort_keys=True),
    )
    return output


def load_dataset(path: str | Path) -> dict[str, np.ndarray | dict]:
    data = np.load(Path(path), allow_pickle=False)
    return {
        "observations": data["observations"].astype(np.float32),
        "actions": data["actions"].astype(np.float32),
        "task_indices": data["task_indices"].astype(np.int64),
        "terminals": data["terminals"].astype(np.uint8),
        "metadata": json.loads(str(data["metadata"])),
    }


def merge_datasets(paths: list[str | Path], extra: dict[str, np.ndarray | dict]) -> dict[str, np.ndarray | dict]:
    datasets = [load_dataset(path) for path in paths] + [extra]
    reference = datasets[0]
    observations, actions, task_indices, terminals = [], [], [], []
    for dataset in datasets:
        validate_compatible(reference, dataset)
        observations.append(dataset["observations"])
        actions.append(dataset["actions"])
        task_indices.append(dataset["task_indices"])
        terminals.append(dataset["terminals"])
    metadata = dict(extra["metadata"])
    metadata["source_datasets"] = [str(path) for path in paths]
    metadata["samples"] = int(sum(len(chunk) for chunk in observations))
    metadata["terminal_fraction"] = float(np.mean(np.concatenate(terminals)))
    return {
        "observations": np.concatenate(observations).astype(np.float32),
        "actions": np.concatenate(actions).astype(np.float32),
        "task_indices": np.concatenate(task_indices).astype(np.int64),
        "terminals": np.concatenate(terminals).astype(np.uint8),
        "metadata": metadata,
    }


def parse_task_probabilities(items: list[str] | tuple[str, ...]) -> dict[str, float]:
    probabilities = {}
    for item in items:
        if "=" not in item:
            raise ValueError("task probabilities must be TASK=WEIGHT")
        task, value = item.split("=", 1)
        task = task.strip()
        if not task:
            raise ValueError("task probability task name must not be empty")
        weight = float(value)
        if weight <= 0:
            raise ValueError("task probability weights must be positive")
        probabilities[task] = weight
    return probabilities


def execution_actions(labels: np.ndarray, rng: np.random.Generator, noise_std: float) -> np.ndarray:
    if noise_std <= 0:
        return labels
    noise = rng.normal(0.0, noise_std, size=labels.shape).astype(np.float32)
    return np.clip(labels + noise, -1.0, 1.0).astype(np.float32)


def teacher_labels(env: SixDofEnv, tasks: tuple[str, ...], task_indices: np.ndarray) -> np.ndarray:
    if len(tasks) == 1:
        return teacher_actions(env, task=tasks[0])
    return select_task_actions({task: teacher_actions(env, task=task) for task in tasks}, task_indices, tasks)


def expected_observation_dim(task_spec: str) -> int:
    return 28 + task_observation_dim(parse_task_spec(task_spec))


def validate_compatible(reference: dict[str, np.ndarray | dict], candidate: dict[str, np.ndarray | dict]) -> None:
    reference_meta = reference["metadata"]
    candidate_meta = candidate["metadata"]
    if reference["observations"].shape[1] != candidate["observations"].shape[1]:
        raise ValueError("cannot merge datasets with different observation dimensions")
    if reference["actions"].shape[1] != candidate["actions"].shape[1]:
        raise ValueError("cannot merge datasets with different action dimensions")
    if tuple(reference_meta["tasks"]) != tuple(candidate_meta["tasks"]):
        raise ValueError("cannot merge datasets with different task order")
