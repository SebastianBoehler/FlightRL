from __future__ import annotations

import ctypes
from math import pi, sqrt
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "src/flightrl/native"
PIXELS = 64 * 48
TELEMETRY = 19
ACTOR_OBS = PIXELS + TELEMETRY + 3
STUDENT_OBS = ACTOR_OBS + 9


@pytest.fixture(scope="module")
def edge_native(tmp_path_factory):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("a C compiler is required for the edge student contract test")
    library_path = tmp_path_factory.mktemp("edge-student") / "edge_student.so"
    subprocess.run(
        (
            compiler,
            "-shared",
            "-fPIC",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(NATIVE / "native_door_self_mask.c"),
            str(NATIVE / "native_sixdof_vision.c"),
            str(NATIVE / "native_edge_student_observation.c"),
            str(NATIVE / "native_edge_student_action.c"),
            "-lm",
            "-o",
            str(library_path),
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    return ctypes.CDLL(str(library_path))


def _floats(*values: float):
    return (ctypes.c_float * len(values))(*values)


def test_edge_action_maps_all_axes_and_preserves_applied_feedback(edge_native) -> None:
    mapper = edge_native.flightrl_edge_student_control_action
    mapper.argtypes = (
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    )
    command = (ctypes.c_float * 4)()
    applied = (ctypes.c_float * 4)()

    mapper(_floats(1.2, -0.5, 0.25, -0.5), 45.0, 4.0, command, applied)

    assert list(applied) == pytest.approx((1.0, -0.5, 0.25, -0.5))
    assert list(command) == pytest.approx(
        (1.0, -0.5, 0.25, -0.5 * (pi / 4.0) / 4.0)
    )


def test_edge_telemetry_uses_fixed_units_origins_and_four_action_feedback(
    edge_native,
) -> None:
    telemetry_fn = edge_native.flightrl_edge_student_telemetry
    telemetry_fn.argtypes = (
        *(ctypes.POINTER(ctypes.c_float),) * 4,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    )
    output = (ctypes.c_float * TELEMETRY)()

    telemetry_fn(
        _floats(0.4, -0.2, 0.75),
        _floats(0.2, -0.1, 0.05),
        _floats(1.0, 0.0, 0.0, 0.0),
        _floats(1.0, -2.0, 2.0),
        0.1,
        _floats(0.1, 0.2, 0.5),
        0.0,
        _floats(0.25, -0.5, 0.75, -1.0),
        output,
    )

    assert list(output) == pytest.approx(
        (
            0.2, -0.1, 0.1,
            1.0 / 6.0, -2.0 / 6.0, 0.5,
            0.0, 0.0, 1.0,
            0.65 / 2.5,
            0.3 / 4.0, -0.4 / 4.0, 0.25 / 2.0,
            0.0, 1.0,
            0.25, -0.5, 0.75, -1.0,
        ),
        abs=1.0e-6,
    )


def test_edge_grounding_uses_final_mask_bbox_contract(edge_native) -> None:
    grounding_fn = edge_native.flightrl_edge_grounding_from_mask
    grounding_fn.argtypes = (
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_float),
    )
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[20:30, 20:40] = 1
    grounding = (ctypes.c_float * 4)()

    grounding_fn(
        mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        grounding,
    )

    assert list(grounding) == pytest.approx(
        (
            1.0,
            (20 + 39) / 63.0 - 1.0,
            (20 + 29) / 47.0 - 1.0,
            sqrt((20 * 10) / PIXELS),
        ),
        abs=1.0e-6,
    )

    masked = np.zeros((48, 64), dtype=np.uint8)
    masked[0:3, 0:3] = 1
    grounding_fn(
        masked.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        grounding,
    )
    assert list(grounding) == pytest.approx((0.0, 0.0, 0.0, 0.0))


def test_edge_teacher_discovery_uses_final_rendered_visibility(edge_native) -> None:
    update = edge_native.flightrl_edge_student_update_target_observed
    update.argtypes = (
        ctypes.c_uint8,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
    )
    update.restype = ctypes.c_int
    observed = ctypes.c_uint8(1)
    initial_outside_fov = ctypes.c_uint8(0)

    visible = update(1, 0.0, ctypes.byref(observed), ctypes.byref(initial_outside_fov))

    assert visible == 0
    assert observed.value == 0
    assert initial_outside_fov.value == 1

    visible = update(0, 1.0, ctypes.byref(observed), ctypes.byref(initial_outside_fov))

    assert visible == 1
    assert observed.value == 1
    assert initial_outside_fov.value == 1

    visible = update(0, 0.0, ctypes.byref(observed), ctypes.byref(initial_outside_fov))

    assert visible == 0
    assert observed.value == 1


def test_training_tail_is_teacher_four_then_grounding_four(edge_native) -> None:
    tail_fn = edge_native.flightrl_edge_student_training_tail
    tail_fn.argtypes = (
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint8,
        ctypes.POINTER(ctypes.c_float),
    )
    output = np.full(STUDENT_OBS, np.nan, dtype=np.float32)

    tail_fn(
        _floats(0.8, -0.25),
        _floats(1.0, -0.5, 0.25, 0.4),
        109,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )

    np.testing.assert_array_equal(
        output[ACTOR_OBS:],
        np.asarray(
            (0.8, 0.0, 0.0, -0.25, 1.0, -0.5, 0.25, 0.4, 109.0),
            dtype=np.float32,
        ),
    )


def test_full_edge_observation_is_gray4_exact_and_door_conditioned(edge_native) -> None:
    observation_fn = edge_native.flightrl_edge_student_observation
    observation_fn.argtypes = (
        *(ctypes.POINTER(ctypes.c_float),) * 7,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    )
    output = np.empty(ACTOR_OBS, dtype=np.float32)
    grounding = (ctypes.c_float * 4)()
    pointer = ctypes.POINTER(ctypes.c_float)
    observation_fn(
        _floats(0.0, 0.0, 0.8),
        _floats(0.0, 0.0, 0.0),
        _floats(1.0, 0.0, 0.0, 0.0),
        _floats(0.0, 0.0, 0.0),
        _floats(-2.0, 2.0, -2.0, 2.0, 0.0, 2.5, 4.0),
        _floats(1.0, 0.0, 1.0, 0.0, 2.0, 7.0),
        None,
        60.0,
        13,
        0.0,
        0,
        0.0,
        0.0,
        _floats(0.0, 0.0, 0.8),
        0.0,
        _floats(0.0, 0.0, 0.0, 0.0),
        grounding,
        output.ctypes.data_as(pointer),
    )

    np.testing.assert_allclose(output[:PIXELS] * 15.0, np.rint(output[:PIXELS] * 15.0))
    np.testing.assert_array_equal(output[PIXELS + TELEMETRY :], (1.0, 0.0, 0.0))
    assert grounding[0] == 1.0
    assert -1.0 <= grounding[1] <= 1.0
    assert -1.0 <= grounding[2] <= 1.0
    assert 0.0 < grounding[3] <= 1.0


def test_episode_group_captures_authoritative_reset_visibility() -> None:
    source = (NATIVE / "native_door_env_binding.c").read_text()
    reset_body = source.split("static void c_reset(Env *env) {", 1)[1].split(
        "static void c_step(Env *env) {", 1
    )[0]

    assert reset_body.index("write_door_observation(env, 1)") < reset_body.index(
        "capture_door_episode_group(env, (uint8_t)low_light)"
    )
    assert reset_body.index(
        "capture_door_episode_group(env, (uint8_t)low_light)"
    ) < reset_body.index("flightrl_edge_student_scene_group_tail")
