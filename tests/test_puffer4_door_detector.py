from __future__ import annotations

import ctypes
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
DETECTOR_SOURCE = ROOT / "src/flightrl/native/native_door_detector.c"


class DoorDetector(ctypes.Structure):
    _fields_ = (
        ("evidence", ctypes.c_float * 5),
        ("last_update_step", ctypes.c_int),
        ("next_update_step", ctypes.c_int),
        ("recovery_yaw", ctypes.c_float),
        ("target_seen", ctypes.c_ubyte),
    )


@pytest.fixture(scope="module")
def detector_update(tmp_path_factory):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("a C compiler is required for the native detector test")
    library_path = tmp_path_factory.mktemp("door-detector") / "detector.so"
    subprocess.run(
        (
            compiler,
            "-shared",
            "-fPIC",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(DETECTOR_SOURCE),
            "-o",
            str(library_path),
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    library = ctypes.CDLL(str(library_path))
    update = library.flightrl_door_detector_update
    update.argtypes = (
        ctypes.POINTER(DoorDetector),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_float,
        ctypes.c_float,
    )
    update.restype = None
    return update


def _detector() -> DoorDetector:
    detector = DoorDetector()
    detector.evidence[:] = (0.9, -0.3, 0.0, 0.2, 0.0)
    detector.last_update_step = 0
    detector.next_update_step = 100
    detector.target_seen = 1
    return detector


def test_native_detector_uses_explicit_seconds_for_normalized_age(
    detector_update,
) -> None:
    detector = _detector()
    grounding = (ctypes.c_float * 4)()
    rng = ctypes.c_uint32(0)

    detector_update(
        ctypes.byref(detector),
        grounding,
        4,
        ctypes.byref(rng),
        0.1,
        0.5,
    )

    assert list(detector.evidence) == pytest.approx(
        (0.9, -0.3, 0.0, 0.2, 0.8)
    )

    detector_update(
        ctypes.byref(detector),
        grounding,
        5,
        ctypes.byref(rng),
        0.1,
        0.5,
    )

    assert list(detector.evidence) == pytest.approx((0.0, 0.0, 0.0, 0.0, 1.0))


def test_native_detector_invalid_time_scale_is_stale_fail_closed(
    detector_update,
) -> None:
    detector = _detector()
    grounding = (ctypes.c_float * 4)()
    rng = ctypes.c_uint32(0)

    detector_update(
        ctypes.byref(detector),
        grounding,
        1,
        ctypes.byref(rng),
        0.1,
        0.0,
    )

    assert list(detector.evidence) == pytest.approx((0.0, 0.0, 0.0, 0.0, 1.0))


def test_native_age_origin_remains_latest_detector_update_attempt(
    detector_update,
) -> None:
    detector = _detector()
    detector.next_update_step = 10
    grounding = (ctypes.c_float * 4)()
    rng = ctypes.c_uint32(0)

    detector_update(
        ctypes.byref(detector),
        grounding,
        10,
        ctypes.byref(rng),
        0.1,
        1.0,
    )

    assert detector.last_update_step == 10
    assert detector.evidence[4] == pytest.approx(0.0)
