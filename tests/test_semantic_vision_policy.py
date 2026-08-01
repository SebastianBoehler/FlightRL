from __future__ import annotations

import gymnasium
import numpy as np
import torch

from flightrl.mujoco.semantic_observation import SemanticStudentObservationLayout
from flightrl.mujoco.semantic_safety_encoder import collision_risk_from_clearance
from flightrl.mujoco.semantic_vision_policy import SemanticVisionPolicy
from flightrl.navigation.spatial_memory import SpatialMemoryConfig
from flightrl.vision import VisionObservationConfig


class SemanticDummyEnv:
    vision_config = VisionObservationConfig(
        width=64,
        height=48,
        color_mode="grayscale",
        include_delta=True,
        include_motion_mask=True,
    )
    memory_config = SpatialMemoryConfig()
    layout = SemanticStudentObservationLayout(vision_config, memory_config)
    single_observation_space = gymnasium.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(layout.flat_dim,),
        dtype=np.float32,
    )
    single_action_space = gymnasium.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(4,),
        dtype=np.float32,
    )
    semantic_action_mode = "target_gated"


class ActiveSemanticDummyEnv(SemanticDummyEnv):
    semantic_action_mode = "active_exploration"


def test_collision_risk_is_calibrated_from_clearance() -> None:
    clearance = torch.tensor([[0.4], [0.65], [1.2]])
    risk = collision_risk_from_clearance(clearance)

    assert risk[0] > risk[1] > risk[2]
    assert torch.allclose(risk[1], torch.tensor([0.5]))


def test_semantic_recurrent_training_matches_rollout_state_contract() -> None:
    torch.manual_seed(13)
    policy = SemanticVisionPolicy(SemanticDummyEnv(), hidden_size=16)
    observations = torch.randn(
        2,
        5,
        SemanticDummyEnv.layout.flat_dim,
    ).clamp(-1.0, 1.0)
    initial_state = (torch.rand(1, 2, 16),)
    terminals = torch.zeros(2, 5)
    terminals[0, 2] = 1.0

    distribution, values, final_state = policy.forward_train_recurrent(
        observations,
        initial_state,
        terminals,
    )
    rollout_means = []
    rollout_values = []
    rollout_state = initial_state
    for step in range(observations.shape[1]):
        alive = (1.0 - terminals[:, step]).reshape(1, -1, 1)
        rollout_state = tuple(value * alive for value in rollout_state)
        rollout_distribution, value, rollout_state = policy.forward_eval(
            observations[:, step],
            rollout_state,
        )
        rollout_means.append(rollout_distribution.mean)
        rollout_values.append(value.reshape(-1))

    assert torch.allclose(
        distribution.mean.reshape(2, 5, 4),
        torch.stack(rollout_means, dim=1),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        values,
        torch.stack(rollout_values, dim=1),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        final_state[0],
        rollout_state[0],
        atol=1e-5,
        rtol=1e-5,
    )


def test_dedicated_recurrent_safety_stays_calibrated_during_navigation_updates() -> None:
    torch.manual_seed(17)
    policy = SemanticVisionPolicy(
        ActiveSemanticDummyEnv(),
        hidden_size=16,
        recurrent_safety=False,
        recurrent_visual_safety=True,
    )
    observations = torch.randn(
        2,
        ActiveSemanticDummyEnv.layout.flat_dim,
    ).clamp(-1.0, 1.0)
    state = policy.initial_state(2, "cpu")

    with torch.no_grad():
        _, _, _, clearance_before, risk_before = policy.forward_eval_with_aux(
            observations,
            state,
        )
    policy.freeze_visual_safety_encoder()
    with torch.no_grad():
        for parameter in (
            *policy.encoder.parameters(),
            *policy.network.parameters(),
        ):
            parameter.add_(torch.randn_like(parameter))
        _, _, _, clearance_after, risk_after = policy.forward_eval_with_aux(
            observations,
            state,
        )

    assert policy.recurrent_visual_safety is not None
    assert all(
        not parameter.requires_grad
        for parameter in policy.recurrent_visual_safety.parameters()
    )
    assert any(parameter.requires_grad for parameter in policy.encoder.parameters())
    assert torch.equal(clearance_before, clearance_after)
    assert torch.equal(risk_before, risk_after)


def test_recurrent_visual_safety_training_matches_episode_resets() -> None:
    torch.manual_seed(19)
    policy = SemanticVisionPolicy(
        ActiveSemanticDummyEnv(),
        hidden_size=16,
        recurrent_safety=False,
        recurrent_visual_safety=True,
    )
    observations = torch.randn(
        2,
        5,
        ActiveSemanticDummyEnv.layout.flat_dim,
    ).clamp(-1.0, 1.0)
    initial_state = policy.initial_state(2, "cpu")
    terminals = torch.zeros(2, 5)
    terminals[1, 3] = 1.0

    distribution, values, final_state, clearance, risk = (
        policy.forward_train_with_aux(
            observations,
            state=initial_state,
            terminals=terminals,
        )
    )
    rollout_state = initial_state
    rollout_means = []
    rollout_values = []
    rollout_clearance = []
    rollout_risk = []
    for step in range(observations.shape[1]):
        alive = (1.0 - terminals[:, step]).reshape(1, -1, 1)
        rollout_state = tuple(value * alive for value in rollout_state)
        outputs = policy.forward_eval_with_aux(
            observations[:, step],
            rollout_state,
        )
        rollout_distribution, value, rollout_state, step_clearance, step_risk = (
            outputs
        )
        rollout_means.append(rollout_distribution.mean)
        rollout_values.append(value.reshape(-1))
        rollout_clearance.append(step_clearance)
        rollout_risk.append(step_risk)

    assert torch.allclose(
        distribution.mean.reshape(2, 5, 4),
        torch.stack(rollout_means, dim=1),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        values,
        torch.stack(rollout_values, dim=1),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        clearance.reshape(2, 5, 1),
        torch.stack(rollout_clearance, dim=1),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        risk.reshape(2, 5, 1),
        torch.stack(rollout_risk, dim=1),
        atol=1e-5,
        rtol=1e-5,
    )
    assert policy.recurrent_visual_safety is not None
    vision = observations[
        ...,
        policy.encoder.layout.vision_slice,
    ].reshape(2, 5, *policy.encoder.layout.vision.shape)
    replay_clearance, replay_risk, replay_state = (
        policy.recurrent_visual_safety.forward_train_vision(
            vision,
            state=initial_state[1:],
            terminals=terminals,
        )
    )
    assert torch.allclose(clearance, replay_clearance, atol=1e-5, rtol=1e-5)
    assert torch.allclose(risk, replay_risk, atol=1e-5, rtol=1e-5)
    assert torch.allclose(final_state[0], rollout_state[0], atol=1e-5, rtol=1e-5)
    assert torch.allclose(final_state[1], rollout_state[1], atol=1e-5, rtol=1e-5)
    assert replay_state is not None
    assert torch.allclose(replay_state[0], final_state[1], atol=1e-5, rtol=1e-5)


def test_action_loss_does_not_recalibrate_dedicated_safety_model() -> None:
    torch.manual_seed(23)
    policy = SemanticVisionPolicy(
        ActiveSemanticDummyEnv(),
        hidden_size=16,
        recurrent_safety=False,
        recurrent_visual_safety=True,
    )
    observations = torch.randn(
        2,
        4,
        ActiveSemanticDummyEnv.layout.flat_dim,
    ).clamp(-1.0, 1.0)
    state = policy.initial_state(2, "cpu")

    distribution, _values, _state, _clearance, _risk = (
        policy.forward_train_with_aux(
            observations,
            state=state,
            terminals=torch.zeros(2, 4),
        )
    )
    distribution.mean.square().mean().backward()

    assert policy.recurrent_visual_safety is not None
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in policy.recurrent_visual_safety.parameters()
    )
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in policy.decoder.parameters()
    )
