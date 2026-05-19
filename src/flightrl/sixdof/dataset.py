from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .env import SixDofCrazyflieEnv
from .policies import teacher_actions
from .tasks import append_task_encoding, parse_task_spec, select_task_actions, task_observation_dim


def collect_teacher_dataset(
    *,
    task_spec: str,
    num_envs: int,
    steps: int,
    seed: int,
    use_native_step: bool,
) -> dict[str, np.ndarray | dict]:
    tasks = parse_task_spec(task_spec)
    rng = np.random.default_rng(seed)
    env = SixDofCrazyflieEnv(num_envs=num_envs, seed=seed, task=tasks[0], use_native_step=use_native_step)
    obs, _ = env.reset(seed=seed)
    observations = []
    actions = []
    task_indices_all = []
    terminals = []
    for _ in range(steps):
        task_indices = sample_task_indices(rng, num_envs, tasks)
        labels = teacher_labels(env, tasks, task_indices)
        observations.append(append_task_encoding(obs.copy(), task_indices, len(tasks)))
        actions.append(labels.copy())
        task_indices_all.append(task_indices.copy())
        obs, _reward, terminal, truncation, _info = env.step(labels)
        terminals.append(terminal.copy())
        done = terminal | truncation
        if np.any(done):
            obs = env.reset_done(done)
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
        "observation_dim": int(stacked_obs.shape[1]),
        "base_observation_dim": 28,
        "action_dim": int(stacked_actions.shape[1]),
        "terminal_fraction": float(np.mean(stacked_terminals)),
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


def sample_task_indices(rng: np.random.Generator, num_envs: int, tasks: tuple[str, ...]) -> np.ndarray:
    if len(tasks) == 1:
        return np.zeros(num_envs, dtype=np.int64)
    return rng.integers(0, len(tasks), size=num_envs, dtype=np.int64)


def teacher_labels(env: SixDofCrazyflieEnv, tasks: tuple[str, ...], task_indices: np.ndarray) -> np.ndarray:
    if len(tasks) == 1:
        return teacher_actions(env, task=tasks[0])
    return select_task_actions({task: teacher_actions(env, task=task) for task in tasks}, task_indices, tasks)


def expected_observation_dim(task_spec: str) -> int:
    return 28 + task_observation_dim(parse_task_spec(task_spec))
