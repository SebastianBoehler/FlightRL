from __future__ import annotations

import ctypes
from math import cos, pi, sin
from pathlib import Path
import shutil
import subprocess

import pytest
import torch

from flightrl.puffer4_door_contract import (
    PRIVILEGED_DOOR_TEACHER_ACTION_CONTRACT,
)
from flightrl.puffer4_door_mission import FIXED_DOOR_MISSION_METRIC_V1
from flightrl.puffer4_door_teacher import (
    privileged_teacher_actions,
    privileged_teacher_gate,
)


ROOT = Path(__file__).resolve().parents[1]
TEST_SUPPORT = ROOT / "tests/native_door_teacher_test_support.c"


class DoorScene(ctypes.Structure):
    _fields_ = (
        ("door", ctypes.c_float * 6),
        ("obstacle", ctypes.c_float * 6),
        ("target", ctypes.c_float * 3),
        ("target_yaw", ctypes.c_float),
        ("teacher_side", ctypes.c_float),
        ("detour", ctypes.c_float * 2),
        ("coverage", ctypes.c_float * 2),
        ("search_yaw", ctypes.c_float),
        ("search_yaw_progress", ctypes.c_float),
        ("detour_active", ctypes.c_ubyte),
        ("initial_outside_fov", ctypes.c_ubyte),
        ("target_observed", ctypes.c_ubyte),
        ("search_phase", ctypes.c_ubyte),
        ("settle_radius_m", ctypes.c_float),
    )


@pytest.fixture(scope="module")
def teacher_action(tmp_path_factory):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("a C compiler is required for the native teacher test")
    library_path = tmp_path_factory.mktemp("door-teacher") / "teacher.so"
    subprocess.run(
        (
            compiler,
            "-shared",
            "-fPIC",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(TEST_SUPPORT),
            "-lm",
            "-o",
            str(library_path),
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    library = ctypes.CDLL(str(library_path))
    action = library.flightrl_door_teacher_action
    action.argtypes = (
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(DoorScene),
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
    )
    action.restype = None
    return action


def _action(teacher_action, *, position_xy: tuple[float, float]) -> list[float]:
    scene = DoorScene()
    scene.target[:] = (0.80, 0.0, 0.80)
    scene.target_yaw = pi
    scene.target_observed = 1
    scene.settle_radius_m = min(
        FIXED_DOOR_MISSION_METRIC_V1.planar_position_tolerance_m,
        FIXED_DOOR_MISSION_METRIC_V1.standoff_tolerance_m,
    )
    position = (ctypes.c_float * 3)(*position_xy, 0.80)
    quaternion = (ctypes.c_float * 4)(cos(pi / 2.0), 0.0, 0.0, sin(pi / 2.0))
    action = (ctypes.c_float * 2)()
    teacher_action(
        position,
        quaternion,
        ctypes.byref(scene),
        PRIVILEGED_DOOR_TEACHER_ACTION_CONTRACT.max_yawrate_deg_s,
        action,
    )
    return list(action)


def _detour_action(
    teacher_action,
    *,
    distance_to_detour_m: float,
) -> tuple[list[float], int]:
    scene = DoorScene()
    scene.target[:] = (1.0, 0.0, 0.80)
    scene.detour[:] = (0.0, 1.0)
    scene.detour_active = 1
    scene.target_observed = 1
    scene.settle_radius_m = 0.08
    position = (ctypes.c_float * 3)(0.0, 1.0 - distance_to_detour_m, 0.80)
    quaternion = (ctypes.c_float * 4)(
        cos(pi / 4.0),
        0.0,
        0.0,
        sin(pi / 4.0),
    )
    action = (ctypes.c_float * 2)()
    teacher_action(
        position,
        quaternion,
        ctypes.byref(scene),
        PRIVILEGED_DOOR_TEACHER_ACTION_CONTRACT.max_yawrate_deg_s,
        action,
    )
    return list(action), scene.detour_active


def test_teacher_stops_inside_mission_position_tolerance(teacher_action) -> None:
    assert _action(teacher_action, position_xy=(0.80, 0.0)) == pytest.approx(
        (0.0, 0.0),
        abs=1.0e-6,
    )


def test_teacher_retains_forward_approach_outside_settle_region(
    teacher_action,
) -> None:
    assert _action(teacher_action, position_xy=(1.40, 0.0))[0] == pytest.approx(
        0.80
    )


def test_teacher_does_not_settle_inside_planar_but_outside_standoff_tolerance(
    teacher_action,
) -> None:
    assert _action(teacher_action, position_xy=(0.89, 0.0))[0] == pytest.approx(
        0.80
    )


def test_teacher_preserves_detour_until_planned_turn_entry(teacher_action) -> None:
    before_action, before_active = _detour_action(
        teacher_action,
        distance_to_detour_m=0.221,
    )
    after_action, after_active = _detour_action(
        teacher_action,
        distance_to_detour_m=0.219,
    )

    assert before_active == 1
    assert before_action[0] == pytest.approx(0.80)
    assert after_active == 0
    assert after_action[0] == pytest.approx(0.0)


def test_privileged_teacher_reads_only_explicit_tail() -> None:
    observations = torch.zeros(3, 20)
    observations[:, -6:-4] = torch.tensor((0.4, -0.2))
    observations[:, -2:] = torch.tensor((-0.8, 0.9))

    actions = privileged_teacher_actions(observations)

    torch.testing.assert_close(actions, torch.tensor([[0.4, -0.2]] * 3))


@pytest.mark.parametrize("value", (float("nan"), float("inf"), 1.1))
def test_privileged_teacher_rejects_invalid_actions(value: float) -> None:
    observations = torch.zeros(1, 6)
    observations[0, -6] = value

    with pytest.raises(ValueError, match="teacher"):
        privileged_teacher_actions(observations)


def test_privileged_teacher_gate_rejects_nonfinite_metrics() -> None:
    gate = privileged_teacher_gate(
        {
            "success_rate": float("inf"),
            "collision_rate": 0.0,
            "outside_fov_success_rate": 1.0,
        }
    )

    assert gate["passed"] is False
    assert gate["failures"] == ["success_rate"]
