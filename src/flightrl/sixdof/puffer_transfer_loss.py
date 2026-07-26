from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .puffer_observation import scale_previous_action_observation
from .replay_loss import replay_sample_weights, weighted_envelope_loss, weighted_mse_loss


def puffer_crash_replay_mse(policy: torch.nn.Module, crash_replay: dict[str, torch.Tensor] | None, config: Any) -> torch.Tensor:
    if crash_replay is None or config.crash_replay_coef <= 0.0 or len(crash_replay["observations"]) == 0:
        return next(policy.parameters()).new_tensor(0.0)
    observations = crash_replay["observations"]
    targets = crash_replay["target_actions"]
    batch_size = min(config.crash_replay_batch_size, len(observations))
    indices = torch.randperm(len(observations))[:batch_size]
    observation_batch = scale_previous_action_observation(observations[indices], config.previous_action_observation_scale)
    prediction = puffer_unclamped_mean(policy, observation_batch)
    weights = replay_sample_weights(crash_replay, indices)
    loss = weighted_mse_loss(prediction, targets[indices], weights)
    if config.crash_replay_envelope_coef > 0.0:
        loss = loss + config.crash_replay_envelope_coef * weighted_envelope_loss(prediction, config.crash_replay_action_abs_limit, weights)
    return loss


def puffer_transfer_replay_mse(policy: torch.nn.Module, replay: dict[str, torch.Tensor] | None, config: Any) -> torch.Tensor:
    if replay is None or config.transfer_replay_coef <= 0.0 or len(replay["observations"]) == 0:
        return next(policy.parameters()).new_tensor(0.0)
    observations = replay["observations"]
    targets = replay["target_actions"]
    batch_size = min(config.transfer_replay_batch_size, len(observations))
    indices = torch.randperm(len(observations))[:batch_size]
    prediction = puffer_unclamped_mean(policy, observations[indices])
    target = targets[indices]
    weights = replay_sample_weights(replay, indices)
    loss = weighted_mse_loss(prediction, target, weights) + 0.2 * transfer_sign_loss(prediction, target)
    vertical = replay.get("vertical_mask")
    if vertical is not None:
        vertical_idx = vertical[indices].bool()
        if bool(torch.any(vertical_idx)):
            loss = loss + transfer_sign_loss(prediction[vertical_idx, 1:3], target[vertical_idx, 1:3])
    if config.transfer_replay_envelope_coef > 0.0:
        loss = loss + config.transfer_replay_envelope_coef * weighted_envelope_loss(
            prediction,
            config.transfer_replay_action_abs_limit,
            weights,
        )
    return loss + 0.5 * autoregressive_transfer_replay_mse(policy, replay, config)


def autoregressive_transfer_replay_mse(policy: torch.nn.Module, replay: dict[str, torch.Tensor], config: Any) -> torch.Tensor:
    observations = replay["observations"]
    sequence_start = replay.get("sequence_start")
    if sequence_start is None or len(observations) == 0:
        return next(policy.parameters()).new_tensor(0.0)
    starts = torch.nonzero(sequence_start.bool(), as_tuple=False).flatten()
    if len(starts) == 0:
        starts = observations.new_tensor([0], dtype=torch.long)
    max_len = max(1, min(64, int(config.transfer_replay_batch_size)))
    losses = [autoregressive_window_loss(policy, replay, int(start), max_len) for start in starts[:4]]
    return torch.stack(losses).mean()


def autoregressive_window_loss(policy: torch.nn.Module, replay: dict[str, torch.Tensor], start: int, max_len: int) -> torch.Tensor:
    observations = replay["observations"]
    targets = replay["target_actions"]
    sequence_start = replay["sequence_start"].bool()
    end = min(len(observations), start + max_len)
    for idx in range(start + 1, end):
        if bool(sequence_start[idx]):
            end = idx
            break
    previous_action = observations[start, -4:].clone()
    predictions = []
    for idx in range(start, end):
        observation = observations[idx : idx + 1].clone()
        observation[:, -4:] = previous_action
        prediction = puffer_unclamped_mean(policy, observation)
        predictions.append(prediction[0])
        previous_action = prediction.detach().clamp(-1.0, 1.0)[0]
    prediction_tensor = torch.stack(predictions)
    target = targets[start:end]
    return F.mse_loss(prediction_tensor, target) + 0.2 * transfer_sign_loss(prediction_tensor, target)


def transfer_sign_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = torch.abs(target) > 0.02
    if not bool(torch.any(mask)):
        return prediction.new_tensor(0.0)
    signed_margin = prediction[mask] * torch.sign(target[mask])
    return torch.mean(torch.relu(0.02 - signed_margin).pow(2))


def puffer_unclamped_mean(policy: torch.nn.Module, observations: torch.Tensor) -> torch.Tensor:
    hidden = policy.network(policy.encoder(observations))
    return policy.decoder.mean_action(hidden)
