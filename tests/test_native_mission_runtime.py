from __future__ import annotations

import ctypes
from pathlib import Path
import shutil
import subprocess

import pytest

from flightrl.navigation.mission import (
    MissionEvent,
    MissionPhase,
    MissionState,
    next_state,
    phase_limits,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/flightrl/native/mission_runtime.c"

PHASES = {
    MissionPhase.PRE_FLIGHT: 0,
    MissionPhase.TAKEOFF: 1,
    MissionPhase.SEARCH: 2,
    MissionPhase.VERIFY: 3,
    MissionPhase.NAVIGATE: 4,
    MissionPhase.RECOVER: 5,
    MissionPhase.HOLD: 6,
    MissionPhase.LAND: 7,
    MissionPhase.ABORT: 8,
}
EVENTS = {
    event: index
    for index, event in enumerate(
        (
            MissionEvent.PREFLIGHT_PASSED,
            MissionEvent.TAKEOFF_READY,
            MissionEvent.TARGET_ACQUIRED,
            MissionEvent.TARGET_CONFIRMED,
            MissionEvent.TARGET_REJECTED,
            MissionEvent.TARGET_LOST,
            MissionEvent.BLOCKED,
            MissionEvent.RECOVERED,
            MissionEvent.GOAL_REACHED,
            MissionEvent.LANDING_REQUESTED,
            MissionEvent.LANDED,
            MissionEvent.TIMEOUT,
            MissionEvent.ABORT_REQUESTED,
        )
    )
}
COMMAND_SOURCES = {
    "preflight": 0,
    "controller": 1,
    "policy": 2,
    "perception": 3,
    "abort": 4,
}


class NativeMissionState(ctypes.Structure):
    _fields_ = (
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("phase", ctypes.c_uint32),
        ("resume_phase", ctypes.c_uint32),
        ("step", ctypes.c_uint64),
    )


class NativePhaseLimits(ctypes.Structure):
    _fields_ = (
        ("max_speed_m_s", ctypes.c_float),
        ("max_yawrate_deg_s", ctypes.c_float),
        ("learned_policy_phase_eligible", ctypes.c_uint8),
        ("command_source", ctypes.c_uint8),
        ("reserved", ctypes.c_uint16),
    )


@pytest.fixture(scope="module")
def native_runtime(tmp_path_factory):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("a C compiler is required for the native mission runtime test")
    library_path = tmp_path_factory.mktemp("mission-runtime") / "mission.so"
    subprocess.run(
        (
            compiler,
            "-shared",
            "-fPIC",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(SOURCE),
            "-o",
            str(library_path),
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    library = ctypes.CDLL(str(library_path))
    library.flightrl_mission_next.argtypes = (
        ctypes.POINTER(NativeMissionState),
        ctypes.c_uint32,
    )
    library.flightrl_mission_next.restype = ctypes.c_int
    library.flightrl_mission_phase_limits.argtypes = (
        ctypes.c_uint32,
        ctypes.POINTER(NativePhaseLimits),
    )
    library.flightrl_mission_phase_limits.restype = ctypes.c_int
    return library


def _native_state(state: MissionState) -> NativeMissionState:
    resume = 0xFFFFFFFF if state.resume_phase is None else PHASES[state.resume_phase]
    return NativeMissionState(
        1,
        ctypes.sizeof(NativeMissionState),
        PHASES[state.phase],
        resume,
        state.step,
    )


@pytest.mark.parametrize(
    "events",
    (
        (
            MissionEvent.PREFLIGHT_PASSED,
            MissionEvent.TAKEOFF_READY,
            MissionEvent.TARGET_ACQUIRED,
            MissionEvent.TARGET_CONFIRMED,
            MissionEvent.GOAL_REACHED,
        ),
        (
            MissionEvent.PREFLIGHT_PASSED,
            MissionEvent.TAKEOFF_READY,
            MissionEvent.TARGET_ACQUIRED,
            MissionEvent.TARGET_REJECTED,
            MissionEvent.BLOCKED,
            MissionEvent.RECOVERED,
        ),
        (MissionEvent.TIMEOUT, MissionEvent.ABORT_REQUESTED),
    ),
)
def test_native_runtime_matches_reference_transitions(native_runtime, events) -> None:
    reference = MissionState()
    native = _native_state(reference)

    for event in events:
        reference = next_state(reference, event)
        assert native_runtime.flightrl_mission_next(ctypes.byref(native), EVENTS[event]) == 0
        assert native.phase == PHASES[reference.phase]
        assert native.step == reference.step
        expected_resume = (
            0xFFFFFFFF if reference.resume_phase is None else PHASES[reference.resume_phase]
        )
        assert native.resume_phase == expected_resume


def test_native_runtime_matches_reference_limits(native_runtime) -> None:
    for phase, native_phase in PHASES.items():
        native_limits = NativePhaseLimits()
        assert native_runtime.flightrl_mission_phase_limits(
            native_phase, ctypes.byref(native_limits)
        ) == 0
        reference = phase_limits(phase)
        assert native_limits.max_speed_m_s == pytest.approx(reference.max_speed_m_s)
        assert native_limits.max_yawrate_deg_s == pytest.approx(
            reference.max_yawrate_deg_s
        )
        assert bool(native_limits.learned_policy_phase_eligible) is (
            reference.learned_policy_phase_eligible
        )
        assert native_limits.command_source == COMMAND_SOURCES[reference.command_source]


def test_native_runtime_rejects_invalid_transition_without_mutation(native_runtime) -> None:
    native = _native_state(MissionState(phase=MissionPhase.SEARCH, step=7))
    before = bytes(native)

    assert native_runtime.flightrl_mission_next(
        ctypes.byref(native), EVENTS[MissionEvent.LANDED]
    ) == 2
    assert bytes(native) == before
