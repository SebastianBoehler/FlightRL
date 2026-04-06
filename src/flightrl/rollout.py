from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .env import DronePlanarEnv


def sample_policy_action(policy: torch.nn.Module, observation: np.ndarray, state: Any = None) -> tuple[np.ndarray, Any]:
    device = next(policy.parameters()).device
    obs_tensor = torch.as_tensor(observation[None, :], dtype=torch.float32, device=device)
    with torch.no_grad():
        logits, _, next_state = policy.forward_eval(obs_tensor, state)
    return logits.mean.squeeze(0).cpu().numpy(), next_state


def collect_rollout(env: DronePlanarEnv, steps: int, policy: torch.nn.Module | None = None, seed: int = 0) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    action_dim = int(np.prod(env.single_action_space.shape))
    obs, _ = env.reset(seed=seed)
    trace: list[dict[str, float]] = []
    policy_state = None
    if policy is not None:
        policy_state = policy.initial_state(1, next(policy.parameters()).device)

    for step_idx in range(steps):
        if policy is None:
            action = rng.uniform(-1.0, 1.0, size=(action_dim,)).astype(np.float32)
        else:
            action, policy_state = sample_policy_action(policy, obs[0], policy_state)

        batch_action = np.repeat(action[None, :], env.num_agents, axis=0).astype(np.float32)
        next_obs, rewards, terminals, truncations, _ = env.step(batch_action)
        snapshot = env.snapshot(0)
        snapshot.update(
            step=float(step_idx),
            reward=float(rewards[0]),
            terminal=float(terminals[0]),
            truncation=float(truncations[0]),
        )
        for idx, value in enumerate(action):
            snapshot[f"action_{idx}"] = float(value)
        trace.append(snapshot)
        obs = next_obs
        if env.render_mode is not None:
            env.render()
    return trace


def save_rollout(trace: list[dict[str, float]], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".json":
        output.write_text(json.dumps(trace, indent=2))
        return output

    fieldnames = sorted({key for row in trace for key in row})
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trace)
    return output


def load_rollout(path: str | Path) -> list[dict[str, float]]:
    input_path = Path(path)
    if input_path.suffix == ".json":
        return json.loads(input_path.read_text())
    with input_path.open() as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]
