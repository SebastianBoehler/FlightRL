from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from flightrl.mujoco.semantic_puffer_driver import SemanticPufferDriver
from flightrl.mujoco.semantic_safety_replay import safety_supervision_losses
from flightrl.mujoco.semantic_safety_training import RecurrentSafetyBootstrap
from flightrl.mujoco.semantic_vision_policy import SemanticVisionPolicy


@dataclass(frozen=True, slots=True)
class ImitationHistory:
    action_losses: tuple[float, ...]
    visibility_losses: tuple[float, ...]
    clearance_losses: tuple[float, ...]
    collision_risk_losses: tuple[float, ...]
    safety_replay_updates: int
    replay_clearance_losses: tuple[float, ...]
    replay_collision_risk_losses: tuple[float, ...]
    replay_danger_sequences: int
    replay_safe_sequences: int


@dataclass(frozen=True, slots=True)
class ExpertSequence:
    observations: np.ndarray
    actions: np.ndarray
    target_observed: np.ndarray
    target_visible: np.ndarray
    corridor_clearance: np.ndarray
    state_resets: np.ndarray
    start_state: tuple[torch.Tensor] | None
    next_state: tuple[torch.Tensor] | None


def collect_expert_sequence(
    driver: SemanticPufferDriver,
    horizon: int,
    *,
    rollout_policy: SemanticVisionPolicy | None = None,
    policy_state: tuple[torch.Tensor] | None = None,
    policy_probability: float = 0.0,
    rng: np.random.Generator | None = None,
) -> ExpertSequence:
    observations = []
    actions = []
    target_observed = []
    target_visible = []
    corridor_clearance = []
    state_resets = []
    current_resets = np.zeros(driver.total_agents, dtype=np.float32)
    if rollout_policy is not None and policy_state is None:
        policy_state = rollout_policy.initial_state(driver.total_agents, "cpu")
    start_state = _detach_state(policy_state)
    generator = rng or np.random.default_rng(0)
    for _ in range(horizon):
        state_resets.append(current_resets.copy())
        observations.append(driver.observations.copy())
        expert = np.ascontiguousarray(driver.expert_actions(), dtype=np.float32)
        actions.append(expert.copy())
        target_observed.append(driver.target_observed().copy())
        target_visible.append(driver.target_visible().copy())
        corridor_clearance.append(driver.action_corridor_clearance().copy())
        rollout = expert
        if rollout_policy is not None:
            with torch.no_grad():
                distribution, _, policy_state = rollout_policy.forward_eval(
                    torch.from_numpy(driver.observations.copy()),
                    policy_state,
                )
            if policy_probability > 0.0:
                predicted = distribution.mean.clamp(-1.0, 1.0).numpy()
                use_policy = generator.random(driver.total_agents) < policy_probability
                rollout = np.where(use_policy[:, None], predicted, expert).astype(
                    np.float32
                )
        driver.cpu_step(np.ascontiguousarray(rollout).ctypes.data)
        current_resets = driver.terminals.copy()
        if policy_state is not None:
            alive = torch.from_numpy(1.0 - current_resets).reshape(1, -1, 1)
            policy_state = tuple(value * alive for value in policy_state)
    return ExpertSequence(
        observations=np.stack(observations, axis=1),
        actions=np.stack(actions, axis=1),
        target_observed=np.stack(target_observed, axis=1),
        target_visible=np.stack(target_visible, axis=1),
        corridor_clearance=np.stack(corridor_clearance, axis=1),
        state_resets=np.stack(state_resets, axis=1),
        start_state=start_state,
        next_state=_detach_state(policy_state),
    )


def bootstrap_imitation(
    policy: SemanticVisionPolicy,
    driver: SemanticPufferDriver,
    *,
    updates: int,
    horizon: int,
    learning_rate: float,
    visibility_loss_scale: float = 0.15,
    clearance_loss_scale: float = 1.0,
    collision_risk_loss_scale: float = 2.0,
) -> ImitationHistory:
    visibility_head = nn.Linear(policy.encoder.vision_feature_dim, 1)
    has_policy_safety = any(
        item is not None
        for item in (
            policy.recurrent_visual_safety,
            policy.recurrent_safety,
            policy.visual_safety,
            policy.clearance_head,
        )
    )
    auxiliary_clearance_head = (
        None
        if has_policy_safety
        else nn.Linear(policy.encoder.vision_feature_dim, 1)
    )
    auxiliary_parameters = (
        tuple(auxiliary_clearance_head.parameters())
        if auxiliary_clearance_head is not None
        else ()
    )
    safety_bootstrap = (
        RecurrentSafetyBootstrap(
            policy.recurrent_visual_safety,
            learning_rate=learning_rate,
            clearance_loss_scale=clearance_loss_scale,
            collision_risk_loss_scale=collision_risk_loss_scale,
            seed=17,
        )
        if policy.recurrent_visual_safety is not None
        else None
    )
    safety_parameter_ids = (
        {id(parameter) for parameter in safety_bootstrap.parameters}
        if safety_bootstrap is not None
        else set()
    )
    navigation_policy_parameters = tuple(
        parameter
        for parameter in policy.parameters()
        if id(parameter) not in safety_parameter_ids
    )
    navigation_parameters = (
        *navigation_policy_parameters,
        *visibility_head.parameters(),
        *auxiliary_parameters,
    )
    optimizer = torch.optim.AdamW(navigation_parameters, lr=learning_rate)
    action_losses = []
    visibility_losses = []
    clearance_losses = []
    collision_risk_losses = []
    policy.train()
    visibility_head.train()
    rng = np.random.default_rng(17)
    rollout_state = policy.initial_state(driver.total_agents, "cpu")
    for update in range(1, updates + 1):
        policy_probability = 0.75 * min(1.0, update / max(1, updates // 2))
        sequence = collect_expert_sequence(
            driver,
            horizon,
            rollout_policy=policy,
            policy_state=rollout_state,
            policy_probability=policy_probability,
            rng=rng,
        )
        rollout_state = sequence.next_state
        observations = sequence.observations
        actions = sequence.actions
        observed = sequence.target_observed
        visible = sequence.target_visible
        clearance = sequence.corridor_clearance
        state_resets = torch.from_numpy(sequence.state_resets)
        observation_tensor = torch.from_numpy(observations)
        distribution, _, _next_state, predicted_clearance, _predicted_risk = (
            policy.forward_train_with_aux(
                observation_tensor,
                state=sequence.start_state,
                terminals=state_resets,
            )
        )
        targets = torch.from_numpy(actions).reshape(-1, actions.shape[-1])
        dimension_weights = targets.new_tensor((2.0, 2.0, 1.0, 2.0))
        per_sample = torch.mean(
            (distribution.mean - targets) ** 2 * dimension_weights,
            dim=1,
        )
        observed_flat = torch.from_numpy(observed.reshape(-1)).bool()
        visible_flat = torch.from_numpy(visible.reshape(-1)).bool()
        sample_weights = (
            1.0 + 3.0 * (~observed_flat).float() + 2.0 * visible_flat.float()
        )
        sample_weights += 4.0 * (observed_flat & ~visible_flat).float()
        clearance_flat = torch.from_numpy(clearance.reshape(-1)).float()
        sample_weights += 4.0 * (clearance_flat < 0.9).float()
        sample_weights += 2.0 * (torch.abs(targets[:, 0]) < 0.05).float()
        action_loss = torch.mean(per_sample * sample_weights) / torch.mean(
            sample_weights
        )
        flat_observations = observation_tensor.reshape(
            -1,
            observations.shape[-1],
        )
        clearance_features = policy.encoder.vision_features(flat_observations)
        visibility_logits = visibility_head(clearance_features).squeeze(1)
        visibility_loss = functional.binary_cross_entropy_with_logits(
            visibility_logits,
            visible_flat.float(),
            weight=1.0 + 3.0 * visible_flat.float(),
        )
        if predicted_clearance is None:
            clearance_prediction_m = 4.0 * torch.sigmoid(
                auxiliary_clearance_head(clearance_features)
            ).squeeze(1)
        else:
            clearance_prediction_m = predicted_clearance.squeeze(1)
        clearance_loss, collision_risk_loss = safety_supervision_losses(
            clearance_prediction_m,
            clearance_flat,
        )
        loss = action_loss + visibility_loss_scale * visibility_loss
        if safety_bootstrap is None:
            loss = (
                loss
                + clearance_loss_scale * clearance_loss
                + collision_risk_loss_scale * collision_risk_loss
            )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(navigation_parameters, 1.0)
        optimizer.step()
        if safety_bootstrap is not None:
            safety_metrics = safety_bootstrap.step(
                update=update,
                predicted_clearance_m=predicted_clearance,
                target_clearance_m=torch.from_numpy(clearance),
                observations=observations,
                state_resets=sequence.state_resets,
            )
            clearance_value = safety_metrics.clearance_loss
            risk_value = safety_metrics.collision_risk_loss
        else:
            clearance_value = float(clearance_loss.detach())
            risk_value = float(collision_risk_loss.detach())
        action_losses.append(float(action_loss.detach()))
        visibility_losses.append(float(visibility_loss.detach()))
        clearance_losses.append(clearance_value)
        collision_risk_losses.append(risk_value)
        if update == 1 or update == updates or update % 16 == 0:
            print(
                f"bootstrap={update}/{updates} "
                f"action_mse={action_losses[-1]:.5f} "
                f"visibility_bce={visibility_losses[-1]:.5f} "
                f"clearance_loss={clearance_losses[-1]:.5f} "
                f"risk_bce={collision_risk_losses[-1]:.5f}",
                flush=True,
            )
    replay_counts = (
        safety_bootstrap.replay.counts
        if safety_bootstrap is not None
        else {"danger": 0, "safe": 0}
    )
    return ImitationHistory(
        action_losses=tuple(action_losses),
        visibility_losses=tuple(visibility_losses),
        clearance_losses=tuple(clearance_losses),
        collision_risk_losses=tuple(collision_risk_losses),
        safety_replay_updates=(
            len(safety_bootstrap.replay_clearance_losses)
            if safety_bootstrap is not None
            else 0
        ),
        replay_clearance_losses=(
            tuple(safety_bootstrap.replay_clearance_losses)
            if safety_bootstrap is not None
            else ()
        ),
        replay_collision_risk_losses=(
            tuple(safety_bootstrap.replay_collision_risk_losses)
            if safety_bootstrap is not None
            else ()
        ),
        replay_danger_sequences=replay_counts["danger"],
        replay_safe_sequences=replay_counts["safe"],
    )


def _detach_state(state: tuple[torch.Tensor] | None) -> tuple[torch.Tensor] | None:
    if state is None:
        return None
    return tuple(value.detach().clone() for value in state)
