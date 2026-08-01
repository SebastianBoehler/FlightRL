from __future__ import annotations

import torch
from torch import nn
import pytest

from flightrl.puffer4_edge_contract import (
    EDGE_ACTION_DIM,
    EDGE_FRAME_PIXELS,
    EDGE_MISSION_TOKEN_COUNT,
    EDGE_OBSERVATION_DIM,
    edge_target_id,
)
from flightrl.puffer4_edge_policy import EdgeNavigationActor
from flightrl.puffer4_edge_budget import edge_actor_budget


def _valid_observation(batch_size: int) -> torch.Tensor:
    observation = torch.zeros(batch_size, EDGE_OBSERVATION_DIM)
    observation[:, :EDGE_FRAME_PIXELS] = torch.randint(
        0,
        16,
        (batch_size, EDGE_FRAME_PIXELS),
    ).float() / 15.0
    telemetry_start = EDGE_FRAME_PIXELS
    observation[:, telemetry_start + 8] = 1.0
    observation[:, telemetry_start + 14] = 1.0
    observation[:, -EDGE_MISSION_TOKEN_COUNT] = 1.0
    return observation


def test_edge_actor_meets_initial_aideck_weight_budget() -> None:
    actor = EdgeNavigationActor(hidden_size=48)

    parameters = sum(parameter.numel() for parameter in actor.parameters())
    convolution_count = sum(
        isinstance(module, nn.Conv2d) for module in actor.modules()
    )

    assert parameters <= 50_000
    assert parameters <= 64 * 1024
    assert convolution_count == 2
    assert not any(isinstance(module, nn.GELU) for module in actor.modules())


def test_edge_actor_outputs_bounded_setpoints_and_shared_grounding() -> None:
    torch.manual_seed(7)
    actor = EdgeNavigationActor(hidden_size=48)
    observation = _valid_observation(3)
    state = actor.initial_state(batch_size=3)

    action, grounding, next_state = actor.forward_step(observation, state)

    assert action.shape == (3, EDGE_ACTION_DIM)
    assert grounding.shape == (3, 4)
    assert next_state.shape == (3, 48)
    assert torch.all(action >= -1.0)
    assert torch.all(action <= 1.0)
    assert torch.equal(action[:, 1:3], torch.zeros_like(action[:, 1:3]))
    assert torch.all(grounding[:, 0] >= 0.0)
    assert torch.all(grounding[:, 0] <= 1.0)
    assert torch.all(grounding[:, 1:3] >= -1.0)
    assert torch.all(grounding[:, 1:3] <= 1.0)
    assert torch.all(grounding[:, 3] >= 0.0)
    assert torch.all(grounding[:, 3] <= 1.0)


def test_edge_actor_starts_as_exact_door_axis_persistence_policy() -> None:
    actor = EdgeNavigationActor(hidden_size=48)
    observation = _valid_observation(2)
    telemetry_start = EDGE_FRAME_PIXELS
    observation[:, telemetry_start + 15 : telemetry_start + 19] = torch.tensor(
        ((0.75, 0.50, -0.25, -0.625), (-0.5, -0.75, 1.0, 0.25))
    )

    action, _grounding, _state = actor.forward_step(
        observation,
        actor.initial_state(2),
    )

    assert actor.action_head[0].out_features == 2
    assert torch.count_nonzero(actor.action_head[0].weight) == 0
    assert torch.count_nonzero(actor.action_head[0].bias) == 0
    assert torch.equal(
        action,
        torch.tensor(((0.75, 0.0, 0.0, -0.625), (-0.5, 0.0, 0.0, 0.25))),
    )


def test_edge_actor_clamps_bounded_residuals_over_applied_feedback() -> None:
    actor = EdgeNavigationActor(hidden_size=48)
    observation = _valid_observation(1)
    telemetry_start = EDGE_FRAME_PIXELS
    observation[0, telemetry_start + 15] = 0.8
    observation[0, telemetry_start + 18] = -0.8
    with torch.no_grad():
        actor.action_head[0].bias.copy_(torch.tensor((0.4, -0.4)))

    action, _grounding, _state = actor.forward_step(
        observation,
        actor.initial_state(1),
    )

    assert torch.equal(action, torch.tensor(((1.0, 0.0, 0.0, -1.0),)))


def test_edge_actor_can_reverse_a_saturated_yaw_command_in_one_step() -> None:
    actor = EdgeNavigationActor(hidden_size=48)
    observation = _valid_observation(1)
    telemetry_start = EDGE_FRAME_PIXELS
    observation[0, telemetry_start + 18] = 1.0
    with torch.no_grad():
        actor.action_head[0].bias[1] = -2.0

    action, _grounding, _state = actor.forward_step(
        observation,
        actor.initial_state(1),
    )

    assert action[0, 3] == -1.0


def test_edge_grounding_is_conditioned_on_active_target() -> None:
    actor = EdgeNavigationActor(hidden_size=48)
    with torch.no_grad():
        actor.grounding_target_gate[0].weight.zero_()
        actor.grounding_target_gate[0].bias.fill_(-3.0)
        actor.grounding_target_gate[0].weight[0, edge_target_id("door")] = 6.0
        actor.grounding_target_gate[0].weight[1, edge_target_id("monitor")] = 6.0
        actor.grounding_head.weight.zero_()
        actor.grounding_head.bias.zero_()
        actor.grounding_head.weight[0, 0] = 1.0
        actor.grounding_head.weight[0, 1] = -1.0
    visual = torch.ones(2, actor.grounding_head.in_features)
    mission = torch.zeros(2, EDGE_MISSION_TOKEN_COUNT)
    mission[0, edge_target_id("door")] = 1.0
    mission[1, edge_target_id("monitor")] = 1.0

    grounding = actor._grounding(visual, mission)

    assert grounding[0, 0] > grounding[1, 0]


def test_training_visibility_logit_keeps_gradient_when_runtime_output_saturates() -> None:
    actor = EdgeNavigationActor(hidden_size=48)
    with torch.no_grad():
        actor.grounding_head.weight.zero_()
        actor.grounding_head.bias.zero_()
        actor.grounding_head.bias[0] = 100.0
    observation = _valid_observation(1)

    action, grounding, visibility_logit, state = actor.forward_training_step(
        observation,
        actor.initial_state(1),
    )
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        visibility_logit,
        torch.zeros_like(visibility_logit),
    )
    loss.backward()

    runtime = actor.forward_step(observation, actor.initial_state(1))
    assert grounding[0, 0] == 1.0
    assert actor.grounding_head.bias.grad[0] > 0.0
    assert torch.equal(action, runtime[0])
    assert torch.equal(grounding, runtime[1])
    assert torch.equal(state, runtime[2])


def test_training_step_rejects_nonfinite_visibility_logit() -> None:
    actor = EdgeNavigationActor(hidden_size=48)
    with torch.no_grad():
        actor.grounding_head.bias[0] = float("inf")

    with pytest.raises(RuntimeError, match="nonfinite"):
        actor.forward_training_step(_valid_observation(1), actor.initial_state(1))


def test_edge_actor_reset_state_is_deterministic() -> None:
    torch.manual_seed(11)
    actor = EdgeNavigationActor(hidden_size=48)
    observation = _valid_observation(2)

    first = actor.forward_step(observation, actor.initial_state(2))
    repeated = actor.forward_step(observation, actor.initial_state(2))

    assert torch.equal(first[0], repeated[0])
    assert torch.equal(first[1], repeated[1])
    assert torch.equal(first[2], repeated[2])


@pytest.mark.parametrize("invalid", (float("nan"), float("inf"), -0.01, 1.01))
def test_edge_actor_rejects_invalid_frame_values(invalid: float) -> None:
    actor = EdgeNavigationActor(hidden_size=48)
    observation = _valid_observation(1)
    observation[0, 0] = invalid

    with pytest.raises(ValueError, match="finite|frame"):
        actor.forward_step(observation, actor.initial_state(1))


def test_edge_actor_rejects_noncanonical_mission_or_state() -> None:
    actor = EdgeNavigationActor(hidden_size=48)
    observation = _valid_observation(1)
    observation[0, -EDGE_MISSION_TOKEN_COUNT:] = torch.tensor([1.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="one-hot"):
        actor.forward_step(observation, actor.initial_state(1))

    observation = _valid_observation(1)
    state = actor.initial_state(1)
    state[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        actor.forward_step(observation, state)

    state = actor.initial_state(1)
    state[0, 0] = 6.01
    with pytest.raises(ValueError, match="invariant"):
        actor.forward_step(observation, state)


def test_edge_actor_rejects_non_gray4_or_impossible_unit_vectors() -> None:
    actor = EdgeNavigationActor(hidden_size=48)
    observation = _valid_observation(1)
    observation[0, 0] = 0.5
    with pytest.raises(ValueError, match="gray4"):
        actor.forward_step(observation, actor.initial_state(1))

    observation = _valid_observation(1)
    observation[0, EDGE_FRAME_PIXELS + 8] = 0.5
    with pytest.raises(ValueError, match="body-up"):
        actor.forward_step(observation, actor.initial_state(1))


def test_edge_actor_budget_reports_wire_and_compute_costs() -> None:
    actor = EdgeNavigationActor(hidden_size=48)

    budget = edge_actor_budget(actor)

    assert budget["parameter_count"] == 17_602
    assert budget["int8_weight_bytes"] == 17_240
    assert budget["quantized_parameter_bytes"] == 18_688
    assert budget["packed_input_bytes"] == 1_635
    assert budget["macs_per_step"] == 96_048
    assert budget["model_input_elements"] == EDGE_OBSERVATION_DIM
    assert budget["largest_internal_activation_elements"] == 1_536
    assert budget["largest_single_tensor_elements"] == EDGE_OBSERVATION_DIM
    assert budget["within_contract"] is True
