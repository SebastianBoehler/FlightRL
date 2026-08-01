from __future__ import annotations

import importlib
import importlib.util
from math import pi

import pytest


def _mission_module():
    spec = importlib.util.find_spec("flightrl.puffer4_door_mission")
    assert spec is not None, "the corrected fixed-door mission metric is missing"
    return importlib.import_module("flightrl.puffer4_door_mission")


def _settled_sample(mission):
    return mission.DoorMissionSample(
        position_m=(0.80, 0.0, 0.80),
        velocity_m_s=(0.0, 0.0, 0.0),
        yaw_rad=pi,
        yaw_rate_rad_s=0.0,
        room_bounds_m=(0.0, 4.0, -2.0, 2.0, 0.0, 2.5, 4.0),
        door_face=0,
        target_position_m=(0.80, 0.0, 0.80),
        target_yaw_rad=pi,
        visible=True,
    )


def test_mission_metric_v1_encodes_approach_and_settle_at_0p80m() -> None:
    mission = _mission_module()
    metric = mission.FIXED_DOOR_MISSION_METRIC_V1

    assert metric.metric_id == "fixed-door-approach-settle-0p80m-v1"
    assert metric.target_standoff_m == pytest.approx(0.80)
    assert metric.planar_position_tolerance_m == pytest.approx(0.10)
    assert metric.vertical_position_tolerance_m == pytest.approx(0.10)
    assert metric.standoff_tolerance_m == pytest.approx(0.08)
    assert metric.yaw_alignment_tolerance_rad == pytest.approx(pi / 18.0)
    assert metric.max_horizontal_speed_m_s == pytest.approx(0.08)
    assert metric.max_vertical_speed_m_s == pytest.approx(0.05)
    assert metric.max_yaw_rate_rad_s == pytest.approx(pi / 36.0)
    assert metric.dwell_steps == 33


@pytest.mark.parametrize(
    "changes",
    (
        {"position_m": (0.68, 0.0, 0.80)},
        {"position_m": (0.80, 0.11, 0.80)},
        {"position_m": (0.80, 0.0, 0.91)},
        {"velocity_m_s": (0.081, 0.0, 0.0)},
        {"velocity_m_s": (0.0, 0.0, 0.051)},
        {"yaw_rad": pi + pi / 18.0 + 1.0e-4},
        {"yaw_rate_rad_s": pi / 36.0 + 1.0e-4},
        {"visible": False},
    ),
)
def test_mission_metric_rejects_each_unsettled_condition(changes: dict) -> None:
    mission = _mission_module()
    sample = _settled_sample(mission)
    sample = mission.DoorMissionSample(
        **({
            "position_m": sample.position_m,
            "velocity_m_s": sample.velocity_m_s,
            "yaw_rad": sample.yaw_rad,
            "yaw_rate_rad_s": sample.yaw_rate_rad_s,
            "room_bounds_m": sample.room_bounds_m,
            "door_face": sample.door_face,
            "target_position_m": sample.target_position_m,
            "target_yaw_rad": sample.target_yaw_rad,
            "visible": sample.visible,
        } | changes)
    )

    result = mission.FIXED_DOOR_MISSION_METRIC_V1.evaluate(
        sample,
        prior_dwell_steps=12,
    )

    assert result.in_tolerance is False
    assert result.dwell_steps == 0
    assert result.success is False


def test_mission_metric_requires_consecutive_dwell_and_resets_on_breach() -> None:
    mission = _mission_module()
    metric = mission.FIXED_DOOR_MISSION_METRIC_V1
    sample = _settled_sample(mission)
    dwell = 0

    for _ in range(metric.dwell_steps - 1):
        result = metric.evaluate(sample, prior_dwell_steps=dwell)
        dwell = result.dwell_steps
        assert result.success is False

    breach = mission.DoorMissionSample(
        position_m=sample.position_m,
        velocity_m_s=(0.09, 0.0, 0.0),
        yaw_rad=sample.yaw_rad,
        yaw_rate_rad_s=sample.yaw_rate_rad_s,
        room_bounds_m=sample.room_bounds_m,
        door_face=sample.door_face,
        target_position_m=sample.target_position_m,
        target_yaw_rad=sample.target_yaw_rad,
        visible=sample.visible,
    )
    assert metric.evaluate(breach, prior_dwell_steps=dwell).dwell_steps == 0

    dwell = 0
    for _ in range(metric.dwell_steps):
        result = metric.evaluate(sample, prior_dwell_steps=dwell)
        dwell = result.dwell_steps

    assert result.success is True
    assert result.dwell_steps == metric.dwell_steps


def test_only_current_metric_is_promotion_compatible() -> None:
    mission = _mission_module()

    assert mission.classify_mission_metric(None) == "incompatible"
    assert (
        mission.classify_mission_metric(
            {"metric_id": "retired-mission-metric"}
        )
        == "incompatible"
    )
    with pytest.raises(ValueError, match="incompatible"):
        mission.require_current_mission_metric(None)


@pytest.mark.parametrize("prior", (True, -1, 0.5, 34))
def test_mission_dwell_state_requires_a_nonnegative_integer(prior) -> None:
    mission = _mission_module()

    with pytest.raises(ValueError, match="state range"):
        mission.FIXED_DOOR_MISSION_METRIC_V1.evaluate(
            _settled_sample(mission),
            prior_dwell_steps=prior,
        )


@pytest.mark.parametrize(
    "changes",
    (
        {"visible": "false"},
        {"door_face": 0.0},
        {"yaw_rad": "3.14"},
        {"position_m": (0.80, 0.0)},
    ),
)
def test_mission_metric_rejects_noncanonical_sample_types(changes: dict) -> None:
    mission = _mission_module()
    sample = _settled_sample(mission)
    values = {
        field: getattr(sample, field)
        for field in sample.__dataclass_fields__
    }

    result = mission.FIXED_DOOR_MISSION_METRIC_V1.evaluate(
        mission.DoorMissionSample(**(values | changes)),
        prior_dwell_steps=0,
    )

    assert result == mission.DoorMissionEvaluation(False, 0, False)
