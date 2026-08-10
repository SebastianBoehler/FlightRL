from __future__ import annotations

import numpy as np
import pytest

from flightrl.exploration.range_contract import (
    RANGE_ACTION_DIM,
    RANGE_EXPLORATION_OBSERVATION_DIM,
    RANGE_MAP_SHAPE,
    range_exploration_contract_payload,
)
from flightrl.exploration.range_observation import (
    build_range_exploration_observation,
)


def test_range_contract_gives_policy_direction_control_without_privileged_inputs() -> None:
    contract = range_exploration_contract_payload()

    assert contract["contract_id"] == "range-frontier-exploration-v2"
    assert contract["observation"]["flat_values"] == 4106
    assert contract["observation"]["segments"] == {
        "exploration_map": [0, 4096],
        "horizontal_ranges": [4096, 4100],
        "range_validity": [4100, 4104],
        "previous_applied_action": [4104, 4106],
    }
    assert contract["observation"]["map"]["channels"] == [
        "visited",
        "observed_free",
        "occupied",
        "frontier",
    ]
    assert contract["observation"]["prohibited_actor_inputs"] == [
        "selected_frontier",
        "target_bearing",
        "target_pose",
        "scene_geometry",
        "privileged_pose",
        "simulator_truth",
    ]
    assert contract["observation"]["temporal_context"] == (
        "explicit_occupancy_map_and_previous_action"
    )
    assert contract["action"]["normalized_order"] == ["forward", "yaw"]
    assert contract["action"]["policy_owns_exploration_direction"] is True
    assert contract["authority"] == {
        "training": False,
        "shadow": False,
        "deployment": False,
        "flight": False,
    }
    assert RANGE_EXPLORATION_OBSERVATION_DIM == 4106
    assert RANGE_MAP_SHAPE == (4, 32, 32)
    assert RANGE_ACTION_DIM == 2


def test_range_observation_flattens_exact_segments_as_float32() -> None:
    exploration_map = np.zeros((4, 32, 32), dtype=np.float32)
    exploration_map[0, 0, 0] = 1.0
    exploration_map[3, 31, 31] = 1.0

    observation = build_range_exploration_observation(
        exploration_map,
        np.asarray((0.25, 0.5, 0.75, 1.0), dtype=np.float32),
        np.asarray((1.0, 0.0, 1.0, 0.0), dtype=np.float32),
        np.asarray((0.4, -0.5), dtype=np.float32),
    )

    assert observation.shape == (4106,)
    assert observation.dtype == np.float32
    assert observation[0] == np.float32(1.0)
    assert observation[4095] == np.float32(1.0)
    assert observation[4096:4106].tolist() == pytest.approx(
        [0.25, 0.5, 0.75, 1.0, 1.0, 0.0, 1.0, 0.0, 0.4, -0.5]
    )


@pytest.mark.parametrize(
    ("map_value", "ranges", "validity", "previous_action", "message"),
    [
        (np.zeros((4, 31, 32)), np.zeros(4), np.ones(4), np.zeros(2), "map shape"),
        (np.full((4, 32, 32), np.nan), np.zeros(4), np.ones(4), np.zeros(2), "finite"),
        (np.zeros((4, 32, 32)), np.asarray((0.0, 0.0, 0.0, 1.1)), np.ones(4), np.zeros(2), "ranges"),
        (np.zeros((4, 32, 32)), np.zeros(4), np.asarray((1.0, 0.0, 0.5, 1.0)), np.zeros(2), "validity"),
        (np.zeros((4, 32, 32)), np.zeros(4), np.ones(4), np.asarray((-0.1, 0.0)), "action"),
    ],
)
def test_range_observation_rejects_values_outside_live_contract(
    map_value: np.ndarray,
    ranges: np.ndarray,
    validity: np.ndarray,
    previous_action: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_range_exploration_observation(
            map_value,
            ranges,
            validity,
            previous_action,
        )
