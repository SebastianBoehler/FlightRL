from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .dataset import merge_datasets, sample_task_indices, teacher_labels
from .env import SixDofCrazyflieEnv
from .evaluation import checkpoint_tasks, load_policy_from_checkpoint
from .tasks import append_task_encoding, parse_task_spec


def collect_policy_dataset(
    *,
    checkpoint_path: str | Path,
    task_spec: str | None,
    num_envs: int,
    steps: int,
    seed: int,
    use_native_step: bool,
    beta: float = 0.0,
    reset_profile: str | None = None,
) -> dict[str, np.ndarray | dict]:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    model = load_policy_from_checkpoint(checkpoint)
    policy_tasks = checkpoint_tasks(checkpoint)
    selected_tasks = parse_task_spec(task_spec) if task_spec else policy_tasks
    validate_selected_tasks(selected_tasks, policy_tasks)
    rng = np.random.default_rng(seed)
    env = SixDofCrazyflieEnv(num_envs=num_envs, seed=seed, task=selected_tasks[0], use_native_step=use_native_step, reset_profile=reset_profile)
    obs, _ = env.reset(seed=seed)
    observations, actions, task_indices_all, terminals = [], [], [], []
    beta = float(np.clip(beta, 0.0, 1.0))
    for _ in range(steps):
        local_indices = sample_task_indices(rng, num_envs, selected_tasks)
        policy_indices = policy_task_indices(selected_tasks, policy_tasks, local_indices)
        labels = teacher_labels(env, selected_tasks, local_indices)
        model_obs = append_task_encoding(obs.copy(), policy_indices, len(policy_tasks))
        policy_actions = predict_actions(model, model_obs)
        executed = beta * labels + (1.0 - beta) * policy_actions
        observations.append(model_obs.copy())
        actions.append(labels.copy())
        task_indices_all.append(policy_indices.copy())
        obs, _reward, terminal, truncation, _info = env.step(executed)
        terminals.append(terminal.copy())
        done = terminal | truncation
        if np.any(done):
            obs = env.reset_done(done)
    return build_dataset(
        observations,
        actions,
        task_indices_all,
        terminals,
        {
            "tasks": list(policy_tasks),
            "task_spec": ",".join(policy_tasks),
            "collected_tasks": list(selected_tasks),
            "source_checkpoint": str(checkpoint_path),
            "rollout_policy": "checkpoint",
            "beta": beta,
            "num_envs": num_envs,
            "steps": steps,
            "seed": seed,
            "native_step": use_native_step,
            "reset_profile": env.reset_profile.name,
        },
    )


def predict_actions(model, observations: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.from_numpy(observations).float()).cpu().numpy().astype(np.float32)


def build_dataset(observations, actions, task_indices, terminals, metadata: dict) -> dict[str, np.ndarray | dict]:
    stacked_obs = np.concatenate(observations).astype(np.float32)
    stacked_actions = np.concatenate(actions).astype(np.float32)
    stacked_tasks = np.concatenate(task_indices).astype(np.int64)
    stacked_terminals = np.concatenate(terminals).astype(np.uint8)
    metadata = {
        **metadata,
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


def policy_task_indices(selected_tasks: tuple[str, ...], policy_tasks: tuple[str, ...], local_indices: np.ndarray) -> np.ndarray:
    mapped = np.asarray([policy_tasks.index(task) for task in selected_tasks], dtype=np.int64)
    return mapped[local_indices].astype(np.int64)


def validate_selected_tasks(selected_tasks: tuple[str, ...], policy_tasks: tuple[str, ...]) -> None:
    missing = [task for task in selected_tasks if task not in policy_tasks]
    if missing:
        raise ValueError(f"selected task(s) not present in checkpoint: {', '.join(missing)}")

