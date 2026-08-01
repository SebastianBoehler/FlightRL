from __future__ import annotations

import ctypes
from pathlib import Path
import shutil
import subprocess

import pytest

from flightrl.puffer4_config import Puffer4ExportSettings
from flightrl.puffer4_door_sections import build_fixed_door_teacher_sections


ROOT = Path(__file__).resolve().parents[1]
TEST_SUPPORT = ROOT / "tests/native_door_scene_test_support.c"


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
def scene_library(tmp_path_factory):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("a C compiler is required for the native scene test")
    library_path = tmp_path_factory.mktemp("door-scene") / "scene.so"
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
    library.flightrl_test_door_collision_margin_m.restype = ctypes.c_float
    library.flightrl_test_door_route_clearance_m.restype = ctypes.c_float
    library.flightrl_test_door_scene_sample.argtypes = (
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(DoorScene),
    )
    library.flightrl_test_door_scene_route_is_clear.argtypes = (
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(DoorScene),
    )
    library.flightrl_test_door_scene_route_is_clear.restype = ctypes.c_int
    library.flightrl_test_door_scene_turn_is_clear.argtypes = (
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(DoorScene),
    )
    library.flightrl_test_door_scene_turn_is_clear.restype = ctypes.c_int
    library.flightrl_test_door_scene_coverage_is_visible.argtypes = (
        ctypes.POINTER(DoorScene),
    )
    library.flightrl_test_door_scene_coverage_is_visible.restype = ctypes.c_int
    return library


def test_route_clearance_covers_edge_envelope_velocity_decay(
    scene_library,
) -> None:
    env = build_fixed_door_teacher_sections(Puffer4ExportSettings())["env"]
    collision_margin = scene_library.flightrl_test_door_collision_margin_m()
    route_clearance = scene_library.flightrl_test_door_route_clearance_m()
    velocity_decay_distance = (
        env["max_horizontal_speed_m_s"] / env["velocity_gain"]
    )

    assert route_clearance - collision_margin >= velocity_decay_distance


def test_obstacle_scenes_never_spawn_inside_or_cross_teacher_route(
    scene_library,
) -> None:
    disabled = 0
    for seed in range(100_000):
        position = (ctypes.c_float * 3)()
        quaternion = (ctypes.c_float * 4)()
        scene = DoorScene()

        scene_library.flightrl_test_door_scene_sample(
            seed,
            position,
            quaternion,
            ctypes.byref(scene),
        )

        if scene.obstacle[0] > 5.0:
            disabled += 1
            continue
        assert scene.detour_active == 1, seed
        assert scene_library.flightrl_test_door_scene_route_is_clear(
            position,
            ctypes.byref(scene),
        ), seed
        assert scene_library.flightrl_test_door_scene_turn_is_clear(
            position,
            ctypes.byref(scene),
        ), seed
        assert scene_library.flightrl_test_door_scene_coverage_is_visible(
            ctypes.byref(scene),
        ), seed

    assert disabled == 0
