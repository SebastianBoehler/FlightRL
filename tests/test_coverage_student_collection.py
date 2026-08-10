from __future__ import annotations

import numpy as np
import pytest

import flightrl.exploration.student_collection as collection
from flightrl.exploration.student_sequence import (
    EVENT_ADVANCE,
    EVENT_ENTER_SCAN,
    require_matched_counterfactual_pairs,
)
from flightrl.mujoco import is_mujoco_available, is_mujoco_rendering_available


pytestmark = pytest.mark.skipif(
    not is_mujoco_available() or not is_mujoco_rendering_available(),
    reason="MuJoCo rendering is unavailable",
)


def test_teacher_collector_stores_camera_abi_and_privileged_labels_separately() -> None:
    dataset = collection.collect_teacher_dataset(
        (610, 611), split="train", maximum_steps=4
    )

    assert dataset.shape == (4, 2)
    assert dataset.metadata["scene_ids"] == [610, 611]
    assert dataset.model_observation(0).shape == (2, 3091)
    assert np.all(dataset.front_clearance_m > 0.0)
    assert np.all(dataset.pair_ids == -1)
    assert np.all(dataset.resets[0] == 1)
    assert np.all(dataset.dones[-1] == 1)
    assert set(map(tuple, dataset.teacher_actions.reshape(-1, 2))) <= {
        (0.5, 0.0),
        (0.0, 1.0),
    }


def test_real_counterfactual_pair_matches_telemetry_but_changes_label() -> None:
    dataset = collection.collect_matched_counterfactual_pair(
        seed=612, split="selection"
    )

    report = require_matched_counterfactual_pairs(dataset)

    assert report["pairs"] == 1
    np.testing.assert_array_equal(dataset.scene_ids, (612, 612))
    assert dataset.metadata["scene_ids"] == [612, 612]
    np.testing.assert_array_equal(dataset.telemetry[0, 0], dataset.telemetry[0, 1])
    assert not np.array_equal(dataset.packed_frames[0, 0], dataset.packed_frames[0, 1])
    np.testing.assert_array_equal(dataset.teacher_actions[0, 0], (0.5, 0.0))
    np.testing.assert_array_equal(dataset.teacher_actions[0, 1], (0.0, 1.0))
    np.testing.assert_array_equal(
        dataset.event_labels[0], (EVENT_ADVANCE, EVENT_ENTER_SCAN)
    )
    assert dataset.front_clearance_m[0, 0] >= 0.85
    assert dataset.front_clearance_m[0, 1] <= 0.65
