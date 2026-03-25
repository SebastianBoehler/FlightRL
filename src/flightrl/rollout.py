from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from .env import DronePlanarEnv
from .policy import FlightPolicy


def sample_policy_action(policy: FlightPolicy, observation: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits, _ = policy.forward_eval(torch.as_tensor(observation[None, :], dtype=torch.float32))
        return logits.mean.squeeze(0).cpu().numpy()


def collect_rollout(env: DronePlanarEnv, steps: int, policy: FlightPolicy | None = None, seed: int = 0) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    action_dim = int(np.prod(env.single_action_space.shape))
    obs, _ = env.reset(seed=seed)
    trace: list[dict[str, float]] = []
    for step_idx in range(steps):
        action = sample_policy_action(policy, obs[0]) if policy else rng.uniform(-1.0, 1.0, size=(action_dim,)).astype(np.float32)
        batch_action = np.repeat(action[None, :], env.num_agents, axis=0).astype(np.float32)
        next_obs, rewards, terminals, truncations, _ = env.step(batch_action)
        snapshot = env.snapshot(0)
        snapshot.update(step=float(step_idx), reward=float(rewards[0]), terminal=float(terminals[0]), truncation=float(truncations[0]))
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
