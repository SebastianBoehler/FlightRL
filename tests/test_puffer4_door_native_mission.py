from __future__ import annotations

import ctypes
from math import cos, pi, sin
from pathlib import Path
import shutil
import subprocess

import pytest

from flightrl.puffer4_door_mission import (
    DoorMissionSample,
    FIXED_DOOR_MISSION_METRIC_V1,
)


ROOT = Path(__file__).resolve().parents[1]
MISSION_SOURCE = ROOT / "src/flightrl/native/native_door_mission.c"


class NativeMissionConfig(ctypes.Structure):
    _fields_ = (
        ("target_standoff_m", ctypes.c_float),
        ("planar_position_tolerance_m", ctypes.c_float),
        ("vertical_position_tolerance_m", ctypes.c_float),
        ("standoff_tolerance_m", ctypes.c_float),
        ("yaw_alignment_tolerance_rad", ctypes.c_float),
        ("max_horizontal_speed_m_s", ctypes.c_float),
        ("max_vertical_speed_m_s", ctypes.c_float),
        ("max_yaw_rate_rad_s", ctypes.c_float),
        ("dwell_steps", ctypes.c_int),
    )


class NativeMissionState(ctypes.Structure):
    _fields_ = (("dwell_steps", ctypes.c_int),)


@pytest.fixture(scope="module")
def native_mission_step(tmp_path_factory):
    if not MISSION_SOURCE.is_file():
        pytest.skip("native corrected mission predicate is missing")
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("a C compiler is required for the native mission test")
    library_path = tmp_path_factory.mktemp("door-mission") / "mission.so"
    subprocess.run(
        (
            compiler,
            "-shared",
            "-fPIC",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(MISSION_SOURCE),
            "-lm",
            "-o",
            str(library_path),
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    library = ctypes.CDLL(str(library_path))
    step = library.flightrl_door_mission_step
    step.argtypes = (
        ctypes.POINTER(NativeMissionConfig),
        ctypes.POINTER(NativeMissionState),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
        ctypes.c_int,
    )
    step.restype = ctypes.c_int
    return step


def test_native_corrected_mission_predicate_exists() -> None:
    assert MISSION_SOURCE.is_file(), "native corrected mission predicate is missing"


def _config() -> NativeMissionConfig:
    metric = FIXED_DOOR_MISSION_METRIC_V1
    return NativeMissionConfig(
        metric.target_standoff_m,
        metric.planar_position_tolerance_m,
        metric.vertical_position_tolerance_m,
        metric.standoff_tolerance_m,
        metric.yaw_alignment_tolerance_rad,
        metric.max_horizontal_speed_m_s,
        metric.max_vertical_speed_m_s,
        metric.max_yaw_rate_rad_s,
        metric.dwell_steps,
    )


def _sample(**changes) -> DoorMissionSample:
    values = {
        "position_m": (0.80, 0.0, 0.80),
        "velocity_m_s": (0.0, 0.0, 0.0),
        "yaw_rad": pi,
        "yaw_rate_rad_s": 0.0,
        "room_bounds_m": (0.0, 4.0, -2.0, 2.0, 0.0, 2.5, 4.0),
        "door_face": 0,
        "target_position_m": (0.80, 0.0, 0.80),
        "target_yaw_rad": pi,
        "visible": True,
    }
    return DoorMissionSample(**(values | changes))


def _native_step(step, sample: DoorMissionSample, state: NativeMissionState) -> int:
    quaternion = (
        ctypes.c_float * 4
    )(cos(sample.yaw_rad / 2.0), 0.0, 0.0, sin(sample.yaw_rad / 2.0))
    position = (ctypes.c_float * 3)(*sample.position_m)
    velocity = (ctypes.c_float * 3)(*sample.velocity_m_s)
    body_rates = (ctypes.c_float * 3)(0.0, 0.0, sample.yaw_rate_rad_s)
    room = (ctypes.c_float * 7)(*sample.room_bounds_m)
    target = (ctypes.c_float * 3)(*sample.target_position_m)
    config = _config()
    return step(
        ctypes.byref(config),
        ctypes.byref(state),
        position,
        velocity,
        quaternion,
        body_rates,
        room,
        sample.door_face,
        target,
        sample.target_yaw_rad,
        int(sample.visible),
    )


def test_native_mission_requires_the_same_consecutive_dwell(
    native_mission_step,
) -> None:
    metric = FIXED_DOOR_MISSION_METRIC_V1
    state = NativeMissionState()

    for expected_dwell in range(1, metric.dwell_steps):
        assert _native_step(native_mission_step, _sample(), state) == 0
        assert state.dwell_steps == expected_dwell

    assert _native_step(native_mission_step, _sample(), state) == 1
    assert state.dwell_steps == metric.dwell_steps

    assert _native_step(
        native_mission_step,
        _sample(velocity_m_s=(0.081, 0.0, 0.0)),
        state,
    ) == 0
    assert state.dwell_steps == 0


@pytest.mark.parametrize(
    "sample",
    (
        _sample(position_m=(0.68, 0.0, 0.80)),
        _sample(position_m=(0.80, 0.11, 0.80)),
        _sample(position_m=(0.80, 0.0, 0.91)),
        _sample(velocity_m_s=(0.081, 0.0, 0.0)),
        _sample(velocity_m_s=(0.0, 0.0, 0.051)),
        _sample(yaw_rad=pi + pi / 18.0 + 1.0e-4),
        _sample(yaw_rate_rad_s=pi / 36.0 + 1.0e-4),
        _sample(visible=False),
    ),
)
def test_native_mission_matches_python_fail_closed_conditions(
    native_mission_step,
    sample: DoorMissionSample,
) -> None:
    state = NativeMissionState(12)
    native_success = _native_step(native_mission_step, sample, state)
    reference = FIXED_DOOR_MISSION_METRIC_V1.evaluate(
        sample,
        prior_dwell_steps=12,
    )

    assert native_success == int(reference.success)
    assert state.dwell_steps == reference.dwell_steps


@pytest.mark.parametrize(("visible", "prior"), ((2, 0), (1, -1), (1, 34)))
def test_native_mission_rejects_noncanonical_state(
    native_mission_step,
    visible: int,
    prior: int,
) -> None:
    sample = _sample()
    state = NativeMissionState(prior)
    quaternion = (ctypes.c_float * 4)(cos(pi / 2), 0.0, 0.0, sin(pi / 2))
    config = _config()

    success = native_mission_step(
        ctypes.byref(config),
        ctypes.byref(state),
        (ctypes.c_float * 3)(*sample.position_m),
        (ctypes.c_float * 3)(*sample.velocity_m_s),
        quaternion,
        (ctypes.c_float * 3)(0.0, 0.0, sample.yaw_rate_rad_s),
        (ctypes.c_float * 7)(*sample.room_bounds_m),
        sample.door_face,
        (ctypes.c_float * 3)(*sample.target_position_m),
        sample.target_yaw_rad,
        visible,
    )

    assert success == 0
    assert state.dwell_steps == 0
