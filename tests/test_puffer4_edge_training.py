from __future__ import annotations

import json
import numpy as np
import pytest
import torch

import flightrl.puffer4_edge_sequence as edge_sequence
import flightrl.puffer4_edge_training as edge_training
from flightrl.puffer4_edge_collection_evidence import (
    canonical_edge_environment_config,
    edge_environment_config_sha256,
)
from flightrl.puffer4_edge_policy import EdgeNavigationActor
from flightrl.puffer4_edge_sequence import EdgeSequenceDataset
from flightrl.puffer4_edge_training import (
    EdgeTrainConfig,
    EdgeTrainingRejected,
    apply_recurrent_resets,
    evaluate_edge_sequence_loss,
    evaluate_edge_visual_ablation_loss,
    train_edge_student,
)
from flightrl.puffer4_edge_training_state import edge_state_dict_sha256
from puffer4_edge_training_support import training_dataset, training_metadata


def _critical_dataset(split: str, seed: int) -> EdgeSequenceDataset:
    steps, agents = 6, 1
    telemetry = np.zeros((steps, agents, 19), dtype=np.float32)
    telemetry[..., 8] = 1.0
    telemetry[..., 14] = 1.0
    actions = np.zeros((steps, agents, 4), dtype=np.float32)
    actions[3, 0, 0] = 0.8
    grounding = np.zeros((steps, agents, 4), dtype=np.float32)
    grounding[2:4, 0] = (1.0, 0.25, -0.25, 0.4)
    resets = np.zeros((steps, agents), dtype=np.uint8)
    resets[[0, 4], 0] = 1
    dones = np.zeros((steps, agents), dtype=np.uint8)
    dones[3, 0] = 1
    return EdgeSequenceDataset(
        packed_frames=np.zeros((steps, agents, 1536), dtype=np.uint8),
        telemetry=telemetry,
        target_ids=np.zeros((steps, agents), dtype=np.uint8),
        teacher_actions=actions,
        behavior_actions=actions.copy(),
        execution_student_mask=np.zeros(agents, dtype=np.uint8),
        grounding=grounding,
        resets=resets,
        dones=dones,
        metadata=training_metadata(split, seed, steps, agents),
    )

def test_training_trace_validation_does_not_repeat_per_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    original = edge_sequence.require_edge_execution_trace

    def counted(dataset) -> None:
        calls.append(dataset.metadata["split"])
        original(dataset)

    monkeypatch.setattr(edge_sequence, "require_edge_execution_trace", counted)
    try:
        train_edge_student(
            training_dataset("train", 11),
            training_dataset("selection", 21),
            EdgeTrainConfig(epochs=3, tbptt_steps=2),
        )
    except EdgeTrainingRejected:
        pass

    assert calls == ["train", "selection"]


def test_training_rejects_environment_mismatch_before_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = training_dataset("train", 11)
    selection = training_dataset("selection", 21)
    profile = dict(selection.metadata["collection_profile"])
    profile["obstacle_probability"] = 0.75
    selection.metadata["collection_profile"] = profile
    selection.metadata["environment_config"] = canonical_edge_environment_config(
        environment=selection.metadata["environment"],
        agents=selection.shape[1],
        base_seed=selection.metadata["base_seed"],
        appearance_seed=selection.metadata["appearance_seed"],
        collection_profile=profile,
    )
    selection.metadata["environment_config_sha256"] = edge_environment_config_sha256(
        selection.metadata["environment_config"]
    )

    def optimizer_must_not_start(*_args, **_kwargs):
        raise AssertionError("optimizer must not start for mismatched datasets")

    monkeypatch.setattr(edge_training.torch.optim, "AdamW", optimizer_must_not_start)
    with pytest.raises(ValueError, match="environments do not match"):
        train_edge_student(train, selection, EdgeTrainConfig(epochs=1))


def test_training_rejects_uneven_tbptt_chunks_before_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def optimizer_must_not_start(*_args, **_kwargs):
        raise AssertionError("optimizer must not start for an uneven TBPTT tail")

    monkeypatch.setattr(edge_training.torch.optim, "AdamW", optimizer_must_not_start)
    with pytest.raises(ValueError, match="divide evenly"):
        train_edge_student(
            training_dataset("train", 11),
            training_dataset("selection", 21),
            EdgeTrainConfig(epochs=1, tbptt_steps=3),
        )


def test_loss_weights_balance_episodes_visibility_classes_and_decisions() -> None:
    dataset = _critical_dataset("train", 11)

    weights = edge_training.edge_sequence_loss_weights(dataset)

    assert weights.critical[:, 0].tolist() == [True, False, True, True, True, False]
    assert weights.episode[:4, 0].sum() == pytest.approx(
        float(weights.episode[4:, 0].sum())
    )
    visible = torch.from_numpy(dataset.grounding[..., 0] > 0.5)
    assert weights.visibility[visible].sum() == pytest.approx(
        float(weights.visibility[~visible].sum())
    )
    assert weights.decision[0, 0] > weights.decision[1, 0]
    assert weights.decision[2, 0] > weights.decision[1, 0]
    assert weights.decision[3, 0] > weights.decision[1, 0]
    assert weights.decision[4, 0] > weights.decision[5, 0]


def test_baselines_expose_persistence_and_constant_grounding_shortcuts() -> None:
    dataset = _critical_dataset("selection", 21)
    dataset.telemetry[..., 15:19] = dataset.teacher_actions
    config = EdgeTrainConfig(epochs=1)

    baselines = edge_training.edge_training_baselines(dataset, config)

    assert baselines["previous_action"] == {
        "frame_action_loss": 0.0,
        "decision_action_loss": 0.0,
    }
    constant = baselines["constant_grounding"]
    assert constant["visible_probability"] == pytest.approx(0.5)
    assert constant["box"] == pytest.approx([0.25, -0.25, 0.4])
    assert constant["visibility_loss"] == pytest.approx(np.log(2.0))
    assert constant["box_loss"] == pytest.approx(0.0)
    assert constant["grounding_loss"] == pytest.approx(
        config.visibility_loss_weight * np.log(2.0)
    )


def test_visibility_loss_uses_logits_so_saturated_errors_keep_gradient() -> None:
    logit = torch.tensor([100.0], requires_grad=True)

    loss = edge_training.balanced_visibility_loss(
        logit,
        torch.tensor([0.0]),
        torch.tensor([1.0]),
    )
    loss.backward()

    assert float(loss.detach()) == pytest.approx(100.0)
    assert logit.grad.item() == pytest.approx(1.0)


def test_mixed_batch_recurrent_reset_only_zeros_flagged_rows() -> None:
    state = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    reset = apply_recurrent_resets(state, np.array([0, 1, 0], dtype=np.uint8))

    assert torch.equal(reset[0], state[0])
    assert torch.equal(reset[1], torch.zeros(4))
    assert torch.equal(reset[2], state[2])


def test_sequence_trainer_rejects_different_native_build_fingerprints() -> None:
    train = training_dataset("train", 11)
    selection = training_dataset("selection", 21)
    selection.metadata["native_build_fingerprint"]["extension"]["sha256"] = "d" * 64

    with pytest.raises(ValueError, match="do not match"):
        train_edge_student(train, selection, EdgeTrainConfig(epochs=1))


def test_sequence_trainer_rejects_checkpoint_that_only_copies_previous_action() -> None:
    train = training_dataset("train", 11)
    selection = training_dataset("selection", 21)
    train.telemetry[..., 15:19] = train.teacher_actions
    selection.telemetry[..., 15:19] = selection.teacher_actions

    with pytest.raises(EdgeTrainingRejected, match="baseline") as caught:
        train_edge_student(
            train,
            selection,
            EdgeTrainConfig(epochs=1, tbptt_steps=2, learning_rate=1.0e-3),
        )

    report = caught.value.report
    assert report["status"] == "rejected"
    assert len(report["history"]) == 1
    assert set(report["baselines"]) == {"previous_action", "constant_grounding"}
    assert report["baseline_gate"]["passed"] is False
    assert "previous_action" in report["baseline_gate"]["failed_checks"]
    assert report["baseline_gate"]["checks"] == report["history"][0]["baseline_checks"]
    assert report["best_selection_metrics"] == report["history"][0]["selection"]
    json.dumps(report, allow_nan=False)


def test_sequence_trainer_requires_multi_agent_visual_ablation() -> None:
    with pytest.raises(ValueError, match="at least two selection agents"):
        train_edge_student(
            training_dataset("train", 11),
            _critical_dataset("selection", 21),
            EdgeTrainConfig(epochs=1),
        )


def test_sequence_trainer_selects_strict_door_only_edge_checkpoint_parent() -> None:
    train = training_dataset("train", 11)
    selection = training_dataset("selection", 21)
    config = EdgeTrainConfig(epochs=8, tbptt_steps=2, learning_rate=5.0e-3)
    torch.manual_seed(config.seed)
    initial = EdgeNavigationActor(hidden_size=48)
    initial_loss = evaluate_edge_sequence_loss(initial, selection, config)[
        "selection_score"
    ]

    actor, report = train_edge_student(train, selection, config)

    final_loss = evaluate_edge_sequence_loss(actor, selection, config)[
        "selection_score"
    ]
    ablated_loss = evaluate_edge_visual_ablation_loss(actor, selection, config)
    assert final_loss < initial_loss
    assert ablated_loss == report["best_selection_visual_ablation_metrics"]
    assert report["status"] == "complete"
    assert report["trained_target_ids"] == [0]
    assert report["best_selection_loss"] == final_loss
    assert (
        report["best_selection_metrics"]
        == report["history"][report["best_epoch"] - 1]["selection"]
    )
    assert report["baseline_gate"] == {
        "passed": True,
        "checks": {
            "previous_action": True,
            "constant_grounding": True,
            "visual_dependence": True,
        },
    }
    assert set(report["baselines"]) == {"previous_action", "constant_grounding"}
    assert report["selection_rule"] == edge_training.EDGE_SELECTION_RULE
    assert report["selected_actor_state_sha256"] == edge_state_dict_sha256(
        actor.state_dict()
    )
    assert "previous_action_dropout_probability" not in report["config"]
    assert report["loss_contract"]["previous_action"] == (
        "exact_stm32_applied_feedback_without_value_masking"
    )
    assert set(report["history"][0]["selection"]) == {
        "total_loss",
        "decision_action_loss",
        "frame_action_loss",
        "visibility_loss",
        "box_loss",
        "grounding_loss",
        "selection_score",
    }
    assert set(report["history"][0]["selection_visual_ablation"]) == set(
        report["history"][0]["selection"]
    )
    assert report["deployment_authority"] is False
    assert len(report["history"]) == config.epochs
