from __future__ import annotations

import ctypes
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from flightrl.puffer4_door_contract import (
    PRIVILEGED_DOOR_TEACHER_ACTION_CONTRACT,
)
from flightrl.sixdof.native import native_step


ROOT = Path(__file__).resolve().parents[1]
ACTION_SOURCE = ROOT / "src/flightrl/native/native_door_action.c"


def test_native_door_action_uses_declared_yaw_scale_and_feedback(tmp_path) -> None:
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("a C compiler is required for the native action contract test")
    if not ACTION_SOURCE.is_file():
        pytest.fail("native fixed-door action mapping is not implemented")
    library_path = tmp_path / "libnative_door_action.so"
    subprocess.run(
        (
            compiler,
            "-shared",
            "-fPIC",
            str(ACTION_SOURCE),
            "-o",
            str(library_path),
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    library = ctypes.CDLL(str(library_path))
    action_mapper = library.flightrl_door_control_action
    action_mapper.argtypes = (
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    )
    action_mapper.restype = None
    policy_action = (ctypes.c_float * 2)(0.6, -0.5)
    setpoint = (ctypes.c_float * 4)()
    previous_action = (ctypes.c_float * 2)()

    contract = PRIVILEGED_DOOR_TEACHER_ACTION_CONTRACT
    action_mapper(
        policy_action,
        contract.max_yawrate_deg_s,
        contract.physics_max_yawrate_rad_s,
        setpoint,
        previous_action,
    )

    assert list(setpoint) == pytest.approx(
        (0.6, 0.0, 0.0, -0.5 * contract.native_yaw_action_scale)
    )
    assert list(previous_action) == pytest.approx((0.6, -0.5))

    position = np.array([[0.0, 0.0, 0.65]], dtype=np.float32)
    velocity = np.zeros((1, 3), dtype=np.float32)
    quaternion = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    body_rates = np.zeros((1, 3), dtype=np.float32)
    ranges = np.zeros((1, 6), dtype=np.float32)
    low_level = np.array(
        [[0.0, 0.0, 0.0, setpoint[3]]],
        dtype=np.float32,
    )
    for _ in range(200):
        native_step(
            position,
            velocity,
            quaternion,
            body_rates,
            ranges,
            low_level,
            0.01,
        )

    assert body_rates[0, 2] == pytest.approx(
        np.deg2rad(-0.5 * contract.max_yawrate_deg_s),
        abs=1.0e-5,
    )
