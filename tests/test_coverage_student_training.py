from __future__ import annotations

import numpy as np
import pytest
import torch

import flightrl.exploration.student_training as training
from flightrl.exploration.student_provenance import coverage_sequence_sha256
from flightrl.exploration.student_sequence import (
    CoverageSequenceDataset,
    EVENT_ADVANCE,
    EVENT_ENTER_SCAN,
    coverage_sequence_metadata,
)


def _paired_dataset(split: str, scene_start: int) -> CoverageSequenceDataset:
    steps, agents = 4, 2
    packed = np.empty((steps, agents, 1536), dtype=np.uint8)
    packed[:, 0] = 0x22
    packed[:, 1] = 0xEE
    telemetry = np.zeros((steps, agents, 19), dtype=np.float32)
    telemetry[..., 8] = 1.0
    telemetry[..., 14] = 1.0
    actions = np.empty((steps, agents, 2), dtype=np.float32)
    actions[:, 0] = (0.5, 0.0)
    actions[:, 1] = (0.0, 1.0)
    resets = np.ones((steps, agents), dtype=np.uint8)
    dones = np.ones((steps, agents), dtype=np.uint8)
    scene_ids = (scene_start, scene_start)
    return CoverageSequenceDataset(
        packed_frames=packed,
        telemetry=telemetry,
        teacher_actions=actions,
        resets=resets,
        dones=dones,
        front_clearance_m=np.tile(
            np.asarray((1.5, 0.4), dtype=np.float32), (steps, 1)
        ),
        event_labels=np.tile(
            np.asarray((EVENT_ADVANCE, EVENT_ENTER_SCAN), dtype=np.uint8),
            (steps, 1),
        ),
        scene_ids=np.asarray(scene_ids, dtype=np.uint32),
        pair_ids=np.tile(np.arange(steps, dtype=np.int64)[:, None], (1, agents)),
        metadata=coverage_sequence_metadata(
            split=split, steps=steps, scene_ids=scene_ids
        ),
    )


def test_persistence_baseline_rescales_yaw_feedback_from_45_to_8() -> None:
    dataset = _paired_dataset("selection", 21)
    dataset.pair_ids[:] = -1
    dataset.teacher_actions[:] = (0.0, 1.0)
    dataset.event_labels[:] = EVENT_ENTER_SCAN
    dataset.telemetry[..., 18] = 8.0 / 45.0

    metrics = training.persistence_baseline_metrics(dataset)

    assert metrics["action_loss"] == pytest.approx(0.0, abs=1.0e-12)
    assert metrics["decision_action_loss"] == pytest.approx(0.0, abs=1.0e-12)


def test_history_permutation_changes_only_complete_frame_streams() -> None:
    dataset = _paired_dataset("selection", 21)

    clean = torch.stack(
        [dataset.model_observation(step) for step in range(dataset.shape[0])]
    )
    permuted = torch.stack(
        [
            training.history_permuted_observation(dataset, step)
            for step in range(dataset.shape[0])
        ]
    )

    assert torch.equal(permuted[:, :, :3072], clean[:, :, :3072].roll(1, dims=1))
    assert torch.equal(permuted[:, :, 3072:], clean[:, :, 3072:])


def test_recurrent_bc_beats_persistence_and_telemetry_only_on_camera_pairs() -> None:
    train = _paired_dataset("train", 11)
    selection = _paired_dataset("selection", 21)
    config = training.CoverageTrainConfig(
        epochs=80,
        learning_rate=1.0e-2,
        tbptt_steps=1,
        seed=7,
    )

    actor, report = training.train_coverage_student(train, selection, config)

    assert report["status"] == "complete"
    assert report["causal_gate"]["passed"] is True
    assert report["selection"]["decision_mode_accuracy"] == pytest.approx(1.0)
    assert report["selection"]["matched_pair_mode_accuracy"] == pytest.approx(1.0)
    assert report["selection_history_permuted"][
        "decision_mode_accuracy"
    ] == pytest.approx(0.0)
    assert report["causal_gate"]["checks"]["matched_counterfactual"] is True
    assert report["selection"]["decision_action_loss"] < report["persistence_baseline"][
        "decision_action_loss"
    ]
    assert report["selection"]["decision_action_loss"] < report[
        "telemetry_only_baseline"
    ]["decision_action_loss"]
    assert report["selection_history_permuted"]["decision_action_loss"] > report[
        "selection"
    ]["decision_action_loss"] * 1.05
    assert report["training_authority"] is False
    assert report["deployment_authority"] is False
    assert report["flight_authority"] is False
    assert report["datasets"]["train"]["sha256"] == coverage_sequence_sha256(train)
    assert report["datasets"]["selection"]["sha256"] == (
        coverage_sequence_sha256(selection)
    )
    assert actor.parameter_count <= 50_000
