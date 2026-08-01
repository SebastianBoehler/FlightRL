from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from flightrl.puffer4_door_imitation import (
    _forward_min_gru_sequence,
    door_imitation_sample_weights,
)
from flightrl.puffer4_door_observation import DOOR_SENSOR_DIM
from flightrl.puffer4_door_policy import DOOR_PIXELS
from flightrl.puffer4_door_policy import DOOR_OBS_DIM, DOOR_POLICY_OBS_DIM
from flightrl.puffer4_door_training import (
    door_teacher_actions,
    fixed_door_gate,
    fixed_door_teacher_gate,
)
from flightrl.puffer4_door_training_gates import (
    fixed_door_gate as direct_fixed_door_gate,
    fixed_door_teacher_gate as direct_fixed_door_teacher_gate,
)
from flightrl.puffer4_door_grounding import (
    GROUNDING_EVALUATION_APPEARANCE_SEED,
    GROUNDING_SELECTION_APPEARANCE_SEED,
    GROUNDING_TRAIN_APPEARANCE_SEEDS,
    balanced_visibility_loss,
    door_grounding_labels,
)
from flightrl.puffer4_door_grounding_metrics import (
    calibrate_visibility_threshold,
    fixed_door_grounder_gate,
    grounding_metrics,
    grounding_selection_score,
)
from flightrl.puffer4_door_mujoco_replay import (
    mixed_grounding_selection_score,
)


class DummyMinGRU(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            (nn.Linear(hidden_size, 3 * hidden_size, bias=False),)
        )

    @staticmethod
    def _g(value: torch.Tensor) -> torch.Tensor:
        return torch.where(value >= 0, value + 0.5, value.sigmoid())

    @staticmethod
    def _log_g(value: torch.Tensor) -> torch.Tensor:
        return torch.where(
            value >= 0,
            (F.relu(value) + 0.5).log(),
            -F.softplus(-value),
        )

    @staticmethod
    def _highway(
        value: torch.Tensor,
        output: torch.Tensor,
        projection: torch.Tensor,
    ) -> torch.Tensor:
        gate = projection.sigmoid()
        return gate * output + (1.0 - gate) * value

    @staticmethod
    def _heinsen_scan(
        log_coefficients: torch.Tensor,
        log_values: torch.Tensor,
    ) -> torch.Tensor:
        prefix = log_coefficients.cumsum(dim=1)
        return (
            prefix + (log_values - prefix).logcumsumexp(dim=1)
        ).exp()

    def forward_train(self, hidden: torch.Tensor) -> torch.Tensor:
        candidate, gate, projection = self.layers[0](hidden).chunk(3, dim=-1)
        output = self._heinsen_scan(
            -F.softplus(gate),
            -F.softplus(-gate) + self._log_g(candidate),
        )
        return self._highway(hidden, output, projection)

    def initial_state(
        self,
        batch_size: int,
        device: str,
    ) -> tuple[torch.Tensor, ...]:
        hidden_size = self.layers[0].in_features
        return (torch.zeros((1, batch_size, hidden_size), device=device),)

    def forward_eval(
        self,
        hidden: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        candidate, gate, projection = self.layers[0](hidden).chunk(3, dim=-1)
        output = torch.lerp(
            state[0][0],
            self._g(candidate),
            gate.sigmoid(),
        )
        return self._highway(hidden, output, projection), (output[None],)


def test_door_teacher_actions_use_only_excluded_tail() -> None:
    observations = torch.zeros((2, DOOR_OBS_DIM))
    observations[:, DOOR_POLICY_OBS_DIM:] = torch.tensor(
        (
            (0.75, -0.5, 1.0, 0.25, 0.5, 0.1),
            (0.0, 1.0, 0.0, 0.75, 0.5, 0.0),
        )
    )

    assert torch.equal(
        door_teacher_actions(observations),
        torch.tensor(((0.75, -0.5), (0.0, 1.0))),
    )


def test_imitation_weights_use_phase_instead_of_yaw_threshold() -> None:
    observations = torch.zeros((3, DOOR_OBS_DIM))
    phase_offset = 3 * DOOR_PIXELS + DOOR_SENSOR_DIM
    observations[0, phase_offset] = 1.0
    observations[1, phase_offset + 1] = 1.0
    observations[2, phase_offset + 2] = 1.0
    targets = torch.tensor(((0.0, 0.85), (0.0, 0.4), (0.5, 0.0)))

    weights = door_imitation_sample_weights(observations, targets)

    torch.testing.assert_close(weights, torch.tensor((5.0, 3.0, 1.0)))


def test_grounding_labels_preserve_sequence_batch_contract() -> None:
    observations = torch.zeros((2, 3, DOOR_OBS_DIM))
    observations[..., DOOR_POLICY_OBS_DIM + 2 :] = torch.tensor(
        (1.0, 0.25, 0.50, 0.10)
    )

    labels = door_grounding_labels(observations)

    assert labels.shape == (2, 3, 4)
    assert labels.reshape(-1, 4).shape == (6, 4)


def test_parallel_min_gru_matches_uninterrupted_training_scan() -> None:
    torch.manual_seed(19)
    network = DummyMinGRU(hidden_size=8)
    hidden = torch.randn(3, 7, 8)
    terminals = torch.zeros((3, 7))
    state = network.initial_state(3, "cpu")

    expected = network.forward_train(hidden.clone())
    actual, _ = _forward_min_gru_sequence(network, hidden, terminals, state)

    assert torch.allclose(actual, expected, atol=1.0e-6)


def test_parallel_min_gru_resets_after_terminal() -> None:
    torch.manual_seed(23)
    network = DummyMinGRU(hidden_size=8)
    first = torch.randn(2, 7, 8)
    second = first.clone()
    second[:, :3] = torch.randn(2, 3, 8)
    terminals = torch.zeros((2, 7))
    terminals[:, 2] = 1.0
    state = network.initial_state(2, "cpu")

    first_output, _ = _forward_min_gru_sequence(
        network,
        first,
        terminals,
        state,
    )
    second_output, _ = _forward_min_gru_sequence(
        network,
        second,
        terminals,
        state,
    )

    assert torch.allclose(
        first_output[:, 3:],
        second_output[:, 3:],
        atol=1.0e-5,
    )


def test_parallel_min_gru_includes_carried_initial_state() -> None:
    torch.manual_seed(29)
    network = DummyMinGRU(hidden_size=8)
    hidden = torch.randn(2, 7, 8)
    state = (torch.rand((1, 2, 8)),)
    sequential_state = tuple(item.clone() for item in state)
    outputs = []
    for step in range(hidden.shape[1]):
        output, sequential_state = network.forward_eval(
            hidden[:, step],
            sequential_state,
        )
        outputs.append(output)

    expected = torch.stack(outputs, dim=1)
    actual, actual_state = _forward_min_gru_sequence(
        network,
        hidden,
        torch.zeros((2, 7)),
        state,
    )

    assert torch.allclose(actual, expected, atol=1.0e-6)
    assert torch.allclose(
        actual_state[0],
        sequential_state[0],
        atol=1.0e-6,
    )


def test_parallel_min_gru_clears_terminal_final_state() -> None:
    torch.manual_seed(31)
    network = DummyMinGRU(hidden_size=8)
    hidden = torch.randn(2, 7, 8)
    terminals = torch.zeros((2, 7))
    terminals[0, -1] = 1.0

    _, state = _forward_min_gru_sequence(
        network,
        hidden,
        terminals,
        network.initial_state(2, "cpu"),
    )

    assert torch.equal(state[0][:, 0], torch.zeros((1, 8)))
    assert torch.count_nonzero(state[0][:, 1]) > 0


def test_fixed_door_gate_enforces_camera_causal_outside_fov_success() -> None:
    assert fixed_door_gate is direct_fixed_door_gate
    full = {
        "success_rate": 0.84,
        "collision_rate": 0.02,
        "outside_fov_success_rate": 0.72,
    }
    masked = {"success_rate": 0.03}

    assert fixed_door_gate(full, masked)["passed"] is True

    masked["success_rate"] = 0.08
    gate = fixed_door_gate(full, masked)
    assert gate["passed"] is False
    assert gate["failures"] == ["camera_mask"]


def test_fixed_door_teacher_gate_requires_valid_control_upper_bound() -> None:
    assert fixed_door_teacher_gate is direct_fixed_door_teacher_gate
    metrics = {
        "success_rate": 0.97,
        "collision_rate": 0.01,
        "outside_fov_success_rate": 0.95,
    }

    assert fixed_door_teacher_gate(metrics)["passed"] is True

    metrics["collision_rate"] = 0.03
    assert fixed_door_teacher_gate(metrics)["failures"] == ["collision"]


def test_balanced_visibility_loss_weights_rare_negative_examples() -> None:
    logits = torch.full((10,), 4.0)
    labels = torch.tensor([1.0] * 9 + [0.0])

    balanced = balanced_visibility_loss(logits, labels)
    ordinary = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)

    assert balanced > ordinary


def test_grounding_selection_score_penalizes_false_positives() -> None:
    clean = {
        "visibility_recall": 0.95,
        "visibility_false_positive_rate": 0.02,
        "centroid_median_error_widths": 0.08,
    }
    shortcut = {
        "visibility_recall": 0.99,
        "visibility_false_positive_rate": 0.40,
        "centroid_median_error_widths": 0.04,
    }

    assert grounding_selection_score(clean) > grounding_selection_score(shortcut)


def test_mixed_selection_is_limited_by_the_weaker_visual_domain() -> None:
    strong = {
        "visibility_auroc": 0.98,
        "visibility_recall": 0.98,
        "visibility_false_positive_rate": 0.01,
        "centroid_median_error_widths": 0.04,
    }
    weak = {
        "visibility_auroc": 0.80,
        "visibility_recall": 0.80,
        "visibility_false_positive_rate": 0.20,
        "centroid_median_error_widths": 0.15,
    }

    assert abs(mixed_grounding_selection_score(strong, weak) + 0.10) < 1.0e-6


def test_visibility_threshold_is_calibrated_without_final_labels() -> None:
    probabilities = torch.tensor([0.95] * 95 + [0.20] * 5 + [0.10] * 100)
    logits = torch.logit(probabilities).unsqueeze(1).repeat(1, 2)
    labels = torch.zeros((200, 2))
    labels[:100, 0] = 1.0

    threshold = calibrate_visibility_threshold(logits, labels)
    metrics = grounding_metrics(
        logits,
        labels,
        visibility_threshold=threshold,
    )

    assert abs(metrics["visibility_recall"] - 0.95) < 1.0e-6
    assert metrics["visibility_false_positive_rate"] == 0.0
    assert metrics["visibility_auroc"] == 1.0


def test_native_grounder_gate_matches_preregistered_synthetic_gate() -> None:
    metrics = {
        "visibility_auroc": 0.91,
        "centroid_median_error_widths": 0.11,
        "visibility_recall": 0.70,
        "visibility_false_positive_rate": 0.20,
    }

    assert fixed_door_grounder_gate(metrics)["passed"] is True


def test_native_grounding_appearance_partitions_are_disjoint() -> None:
    training = set(GROUNDING_TRAIN_APPEARANCE_SEEDS)

    assert GROUNDING_SELECTION_APPEARANCE_SEED not in training
    assert GROUNDING_EVALUATION_APPEARANCE_SEED not in training
    assert (
        GROUNDING_SELECTION_APPEARANCE_SEED
        != GROUNDING_EVALUATION_APPEARANCE_SEED
    )
