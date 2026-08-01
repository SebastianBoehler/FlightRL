from __future__ import annotations

import ctypes

import numpy as np

from test_native_edge_student_contract import (
    ACTOR_OBS,
    PIXELS,
    STUDENT_OBS,
    _floats,
)

pytest_plugins = ("test_native_edge_student_contract",)


def _render(edge_native, control_step: int) -> tuple[np.ndarray, np.ndarray]:
    function = edge_native.flightrl_edge_student_observation
    function.argtypes = (
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
    grounding = np.empty(4, dtype=np.float32)
    function(
        _floats(0.0, 0.0, 0.8),
        _floats(0.0, 0.0, 0.0),
        _floats(1.0, 0.0, 0.0, 0.0),
        _floats(0.0, 0.0, 0.0),
        _floats(-2.0, 2.0, -2.0, 2.0, 0.0, 2.5),
        _floats(1.0, 0.0, 1.0, 0.0, 2.0, 7.0),
        None,
        60.0,
        13,
        1.0,
        control_step,
        0.0,
        0.0,
        _floats(0.0, 0.0, 0.8),
        0.0,
        _floats(0.0, 0.0, 0.0, 0.0),
        grounding.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return output, grounding


def test_edge_sensor_noise_is_step_deterministic_without_moving_grounding(
    edge_native,
) -> None:
    first, first_grounding = _render(edge_native, 7)
    repeated, repeated_grounding = _render(edge_native, 7)
    next_frame, next_grounding = _render(edge_native, 8)

    np.testing.assert_array_equal(first, repeated)
    np.testing.assert_array_equal(first_grounding, repeated_grounding)
    np.testing.assert_array_equal(first_grounding, next_grounding)
    assert np.any(first[:PIXELS] != next_frame[:PIXELS])
    np.testing.assert_array_equal(first[:PIXELS] * 15.0, np.rint(first[:PIXELS] * 15.0))


def test_training_tail_appends_exact_scene_group(edge_native) -> None:
    function = edge_native.flightrl_edge_student_training_tail
    function.argtypes = (
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint8,
        ctypes.POINTER(ctypes.c_float),
    )
    output = np.full(STUDENT_OBS, np.nan, dtype=np.float32)

    function(
        _floats(0.8, -0.25),
        _floats(1.0, -0.5, 0.25, 0.4),
        109,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )

    assert output[-1] == 109.0
