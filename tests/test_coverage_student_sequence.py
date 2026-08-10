from __future__ import annotations

import numpy as np
import pytest
import torch

from flightrl.exploration.student_provenance import coverage_sequence_sha256
import flightrl.exploration.student_sequence as sequence


def _dataset(*, pair: bool = False):
    steps, agents = (1, 2) if pair else (2, 2)
    packed = np.zeros((steps, agents, 1536), dtype=np.uint8)
    packed[..., 0] = np.asarray((0x10, 0x20), dtype=np.uint8)
    telemetry = np.zeros((steps, agents, 19), dtype=np.float32)
    telemetry[..., 8] = 1.0
    telemetry[..., 14] = 1.0
    actions = np.zeros((steps, agents, 2), dtype=np.float32)
    actions[..., 0] = 0.5
    resets = np.zeros((steps, agents), dtype=np.uint8)
    resets[0] = 1
    dones = np.zeros((steps, agents), dtype=np.uint8)
    dones[-1] = 1
    clearance = np.full((steps, agents), 2.0, dtype=np.float32)
    events = np.full(
        (steps, agents), sequence.EVENT_ADVANCE, dtype=np.uint8
    )
    pair_ids = np.full((steps, agents), -1, dtype=np.int64)
    if pair:
        packed[0, 1, 0] = 0xF0
        actions[0, 1] = (0.0, 1.0)
        clearance[0] = (1.5, 0.4)
        events[0] = (sequence.EVENT_ADVANCE, sequence.EVENT_ENTER_SCAN)
        pair_ids[0] = 7
    scene_ids = (11, 11) if pair else (11, 12)
    return sequence.CoverageSequenceDataset(
        packed_frames=packed,
        telemetry=telemetry,
        teacher_actions=actions,
        resets=resets,
        dones=dones,
        front_clearance_m=clearance,
        event_labels=events,
        scene_ids=np.asarray(scene_ids, dtype=np.uint32),
        pair_ids=pair_ids,
        metadata=sequence.coverage_sequence_metadata(
            split="selection",
            steps=steps,
            scene_ids=scene_ids,
        ),
    )


def test_model_observation_is_exact_gray4_plus_telemetry_only() -> None:
    dataset = _dataset()

    observation = dataset.model_observation(0)

    assert observation.shape == (2, 3091)
    assert observation.dtype == torch.float32
    assert observation[0, :4].tolist() == pytest.approx(
        [1.0 / 15.0, 0.0, 0.0, 0.0]
    )
    assert observation[1, 0].item() == pytest.approx(2.0 / 15.0)
    np.testing.assert_array_equal(
        observation[:, 3072:].numpy(), dataset.telemetry[0]
    )

    changed = _dataset()
    changed.front_clearance_m[:] = 0.01
    changed.event_labels[:] = sequence.EVENT_CONTINUE_SCAN
    changed.scene_ids[:] = (998, 999)
    changed.pair_ids[:] = 99
    changed.metadata.update(
        sequence.coverage_sequence_metadata(
            split="selection", steps=2, scene_ids=(998, 999)
        )
    )
    assert torch.equal(observation, changed.model_observation(0))


def test_sequence_round_trip_preserves_arrays_and_provenance(tmp_path) -> None:
    dataset = _dataset()

    output = sequence.write_coverage_sequence(tmp_path / "sequence.npz", dataset)
    loaded = sequence.load_coverage_sequence(output)

    assert loaded.metadata == dataset.metadata
    for name in sequence.COVERAGE_SEQUENCE_ARRAYS:
        np.testing.assert_array_equal(getattr(loaded, name), getattr(dataset, name))
    assert coverage_sequence_sha256(loaded) == coverage_sequence_sha256(dataset)

    before = coverage_sequence_sha256(loaded)
    loaded.packed_frames[0, 0, 0] ^= 0x01
    assert coverage_sequence_sha256(loaded) != before


def test_matched_pair_contract_holds_nonvisual_history_equal() -> None:
    dataset = _dataset(pair=True)

    report = sequence.require_matched_counterfactual_pairs(dataset)

    assert report == {
        "pairs": 1,
        "clear_samples": 1,
        "blocked_samples": 1,
        "history_steps": 1,
    }

    dataset.telemetry[0, 1, 0] = 0.25
    with pytest.raises(ValueError, match="nonvisual history"):
        sequence.require_matched_counterfactual_pairs(dataset)


def test_matched_pair_contract_binds_clearance_to_teacher_mode() -> None:
    dataset = _dataset(pair=True)
    dataset.teacher_actions[0] = dataset.teacher_actions[0, ::-1]
    dataset.event_labels[0] = dataset.event_labels[0, ::-1]

    with pytest.raises(ValueError, match="teacher mode does not match clearance"):
        sequence.require_matched_counterfactual_pairs(dataset)


def test_sequence_contract_binds_event_to_exact_teacher_mode() -> None:
    dataset = _dataset()
    dataset.teacher_actions[0, 0] = (0.0, 1.0)

    with pytest.raises(ValueError, match="teacher action does not match event"):
        sequence.require_coverage_sequence_dataset(dataset)


def test_matched_pair_contract_requires_same_source_scene() -> None:
    dataset = _dataset(pair=True)
    dataset.scene_ids[1] = 12
    dataset.metadata.update(
        sequence.coverage_sequence_metadata(
            split="selection", steps=1, scene_ids=(11, 12)
        )
    )

    with pytest.raises(ValueError, match="source scene does not match"):
        sequence.require_matched_counterfactual_pairs(dataset)
