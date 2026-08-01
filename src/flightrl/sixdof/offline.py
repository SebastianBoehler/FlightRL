from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .checkpoint_contract import build_checkpoint_payload
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
    eval_reset_profile: str | None = None
    action_weighting: str = "none"
    task_weights: dict[str, float] | None = None


def train_offline_policy(data: dict, config: OfflineTrainConfig) -> dict:
    torch.manual_seed(config.seed)
    observations = data["observations"]
    actions = data["actions"]
    metadata = data["metadata"]
    tasks = tuple(metadata["tasks"])
    train_idx, val_idx = split_indices(len(observations), config.val_ratio, config.seed)
    model = SixDofPolicy(hidden_size=config.hidden_size, input_dim=observations.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    action_weights = compute_action_weights(actions, config.action_weighting)
    sample_weights = compute_sample_weights(data["task_indices"], tasks, config.task_weights)
    best = None
    history = []
    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(model, optimizer, observations[train_idx], actions[train_idx], config.batch_size, action_weights, sample_weights[train_idx])
        val_loss = dataset_loss(model, observations[val_idx], actions[val_idx], config.batch_size, action_weights, sample_weights[val_idx])
        observation_mode = str(metadata.get("observation_mode", "base"))
        eval_metrics = evaluation_metrics(model, tasks, config, observation_mode) if config.select_by_eval else None
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
        reset_profile=config.eval_reset_profile,
        observation_mode=str(metadata.get("observation_mode", "base")),
    )
    return best


def evaluation_metrics(model, tasks: tuple[str, ...], config: OfflineTrainConfig, observation_mode: str) -> dict:
    return evaluate_policy(
        model,
        tasks,
        seed=config.seed + 1000,
        steps=config.eval_steps,
        num_envs=config.eval_num_envs,
        use_native_step=config.use_native_step,
        reset_profile=config.eval_reset_profile,
        observation_mode=observation_mode,
    )


def history_entry(epoch: int, train_loss: float, val_loss: float, eval_metrics: dict | None) -> dict:
    entry = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
    if eval_metrics is not None:
        entry["eval_position_error_m"] = eval_metrics["mean_position_error_m"]
        entry["eval_yaw_error_rad"] = eval_metrics.get("mean_yaw_error_rad")
        entry["eval_yaw_error_p95_rad"] = eval_metrics.get("yaw_error_p95_rad")
        entry["eval_completed_fraction"] = eval_metrics["mean_completed_fraction"]
        entry["eval_clearance_p01_m"] = eval_metrics.get("clearance_p01_m", eval_metrics["min_clearance_m"])
    return entry


def checkpoint_score(checkpoint: dict, config: OfflineTrainConfig) -> tuple:
    if not config.select_by_eval:
        return (checkpoint["val_loss"],)
    metrics = checkpoint["selection_metrics"]
    return (
        -metrics["mean_completed_fraction"],
        -metrics.get("mean_survival_fraction", metrics["mean_completed_fraction"]),
        -metrics.get("clearance_p01_m", metrics["min_clearance_m"]),
        metrics["mean_position_error_m"],
        metrics.get("mean_yaw_error_rad", 0.0),
        metrics.get("yaw_error_p95_rad", metrics.get("mean_yaw_error_rad", 0.0)),
        metrics.get("action_saturation_fraction", 0.0),
        checkpoint["val_loss"],
    )


def compute_action_weights(actions: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return np.ones(actions.shape[1], dtype=np.float32)
    if mode == "inverse_std":
        std = np.std(actions, axis=0).astype(np.float32)
        weights = 1.0 / np.maximum(std, 1e-3)
        return (weights / np.mean(weights)).astype(np.float32)
    raise ValueError(f"unknown action weighting mode {mode!r}")


def compute_sample_weights(task_indices: np.ndarray, tasks: tuple[str, ...], task_weights: dict[str, float] | None) -> np.ndarray:
    weights = np.ones(len(task_indices), dtype=np.float32)
    if not task_weights:
        return weights
    unknown = sorted(set(task_weights) - set(tasks))
    if unknown:
        raise ValueError(f"unknown task weight(s): {', '.join(unknown)}")
    for task, weight in task_weights.items():
        if weight <= 0:
            raise ValueError("task weights must be positive")
        weights[task_indices == tasks.index(task)] = float(weight)
    return (weights / np.mean(weights)).astype(np.float32)


def split_indices(count: int, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(count)
    val_count = max(1, int(count * val_ratio))
    return indices[val_count:], indices[:val_count]


def train_epoch(model, optimizer, observations: np.ndarray, actions: np.ndarray, batch_size: int, action_weights: np.ndarray, sample_weights: np.ndarray) -> float:
    order = torch.randperm(len(observations))
    obs = torch.from_numpy(observations).float()
    target = torch.from_numpy(actions).float()
    weights = torch.from_numpy(action_weights).float()
    samples = torch.from_numpy(sample_weights).float()
    losses = []
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        loss = weighted_mse(model(obs[idx]), target[idx], weights, samples[idx])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


def dataset_loss(model, observations: np.ndarray, actions: np.ndarray, batch_size: int, action_weights: np.ndarray, sample_weights: np.ndarray | None = None) -> float:
    weights = torch.from_numpy(action_weights).float()
    samples = torch.from_numpy(sample_weights if sample_weights is not None else np.ones(len(observations), dtype=np.float32)).float()
    losses = []
    with torch.no_grad():
        for start in range(0, len(observations), batch_size):
            obs = torch.from_numpy(observations[start : start + batch_size]).float()
            target = torch.from_numpy(actions[start : start + batch_size]).float()
            losses.append(float(weighted_mse(model(obs), target, weights, samples[start : start + batch_size])))
    return float(np.mean(losses))


def weighted_mse(prediction: torch.Tensor, target: torch.Tensor, action_weights: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
    per_sample = torch.mean(((prediction - target) * action_weights) ** 2, dim=1)
    return torch.mean(per_sample * sample_weights)


def payload(model, config: OfflineTrainConfig, metadata: dict, tasks: tuple[str, ...], val_loss: float, epoch: int, selection_metrics: dict | None) -> dict:
    checkpoint = build_checkpoint_payload(
        state_dict={key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        tasks=tasks,
        hidden_size=config.hidden_size,
        observation_mode=str(metadata.get("observation_mode", "base")),
    )
    if checkpoint["observation_dim"] != int(metadata["observation_dim"]):
        raise ValueError("dataset observation dimension does not match the current six-DoF checkpoint contract")
    checkpoint.update({
        "dataset": config.dataset,
        "selection_epoch": epoch,
        "selection_mode": "eval" if config.select_by_eval else "val_loss",
        "eval_reset_profile": config.eval_reset_profile or "broad",
        "action_weighting": config.action_weighting,
        "task_weights": config.task_weights or {},
        "selection_metrics": selection_metrics,
        "val_loss": val_loss,
        "note": "Offline teacher-imitation checkpoint; simulation-only and not approved for live hardware.",
    })
    return checkpoint
