from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .evaluation import evaluate_policy
from .policies import SixDofPolicy


@dataclass(frozen=True, slots=True)
class OfflineTrainConfig:
    dataset: str
    hidden_size: int = 256
    epochs: int = 30
    batch_size: int = 8192
    learning_rate: float = 1e-3
    val_ratio: float = 0.1
    seed: int = 17
    eval_steps: int = 800
    eval_num_envs: int = 128
    select_by_eval: bool = False
    use_native_step: bool = False


def train_offline_policy(data: dict, config: OfflineTrainConfig) -> dict:
    torch.manual_seed(config.seed)
    observations = data["observations"]
    actions = data["actions"]
    metadata = data["metadata"]
    tasks = tuple(metadata["tasks"])
    train_idx, val_idx = split_indices(len(observations), config.val_ratio, config.seed)
    model = SixDofPolicy(hidden_size=config.hidden_size, input_dim=observations.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    best = None
    history = []
    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(model, optimizer, observations[train_idx], actions[train_idx], config.batch_size)
        val_loss = dataset_loss(model, observations[val_idx], actions[val_idx], config.batch_size)
        eval_metrics = evaluation_metrics(model, tasks, config) if config.select_by_eval else None
        history.append(history_entry(epoch, train_loss, val_loss, eval_metrics))
        candidate = payload(model, config, metadata, tasks, val_loss, epoch, eval_metrics)
        if best is None or checkpoint_score(candidate, config) < checkpoint_score(best, config):
            best = candidate
    assert best is not None
    model.load_state_dict(best["state_dict"])
    best["history"] = history
    best["metrics"] = evaluate_policy(
        model,
        tasks,
        seed=config.seed + 1000,
        steps=config.eval_steps,
        num_envs=config.eval_num_envs,
        use_native_step=config.use_native_step,
    )
    return best


def evaluation_metrics(model, tasks: tuple[str, ...], config: OfflineTrainConfig) -> dict:
    return evaluate_policy(
        model,
        tasks,
        seed=config.seed + 1000,
        steps=config.eval_steps,
        num_envs=config.eval_num_envs,
        use_native_step=config.use_native_step,
    )


def history_entry(epoch: int, train_loss: float, val_loss: float, eval_metrics: dict | None) -> dict:
    entry = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
    if eval_metrics is not None:
        entry["eval_position_error_m"] = eval_metrics["mean_position_error_m"]
        entry["eval_completed_fraction"] = eval_metrics["mean_completed_fraction"]
        entry["eval_clearance_p01_m"] = eval_metrics.get("clearance_p01_m", eval_metrics["min_clearance_m"])
    return entry


def checkpoint_score(checkpoint: dict, config: OfflineTrainConfig) -> tuple:
    if not config.select_by_eval:
        return (checkpoint["val_loss"],)
    metrics = checkpoint["selection_metrics"]
    return (
        -metrics["mean_completed_fraction"],
        -metrics.get("clearance_p01_m", metrics["min_clearance_m"]),
        metrics["mean_position_error_m"],
        checkpoint["val_loss"],
    )


def split_indices(count: int, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(count)
    val_count = max(1, int(count * val_ratio))
    return indices[val_count:], indices[:val_count]


def train_epoch(model, optimizer, observations: np.ndarray, actions: np.ndarray, batch_size: int) -> float:
    order = torch.randperm(len(observations))
    obs = torch.from_numpy(observations).float()
    target = torch.from_numpy(actions).float()
    losses = []
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        loss = F.mse_loss(model(obs[idx]), target[idx])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


def dataset_loss(model, observations: np.ndarray, actions: np.ndarray, batch_size: int) -> float:
    losses = []
    with torch.no_grad():
        for start in range(0, len(observations), batch_size):
            obs = torch.from_numpy(observations[start : start + batch_size]).float()
            target = torch.from_numpy(actions[start : start + batch_size]).float()
            losses.append(float(F.mse_loss(model(obs), target)))
    return float(np.mean(losses))


def payload(model, config: OfflineTrainConfig, metadata: dict, tasks: tuple[str, ...], val_loss: float, epoch: int, selection_metrics: dict | None) -> dict:
    return {
        "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "task": ",".join(tasks),
        "tasks": list(tasks),
        "task_conditioned": len(tasks) > 1,
        "hidden_size": config.hidden_size,
        "observation_dim": int(metadata["observation_dim"]),
        "base_observation_dim": 28,
        "action_dim": 4,
        "dataset": config.dataset,
        "selection_epoch": epoch,
        "selection_mode": "eval" if config.select_by_eval else "val_loss",
        "selection_metrics": selection_metrics,
        "val_loss": val_loss,
        "note": "Offline teacher-imitation checkpoint; simulation-only and not approved for live hardware.",
    }
