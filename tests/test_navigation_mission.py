from __future__ import annotations

import pytest

from flightrl.navigation.mission import (
    MissionEvent,
    MissionPhase,
    MissionState,
    next_state,
    phase_limits,
)


def test_mission_state_machine_runs_single_drone_navigation_flow() -> None:
    state = MissionState()

    state = next_state(state, MissionEvent.PREFLIGHT_PASSED)
    assert state.phase is MissionPhase.TAKEOFF
    assert state.agent_id == "drone_0"

    state = next_state(state, MissionEvent.TAKEOFF_READY)
    assert state.phase is MissionPhase.SEARCH

    state = next_state(state, MissionEvent.TARGET_ACQUIRED)
    assert state.phase is MissionPhase.VERIFY

    state = next_state(state, MissionEvent.TARGET_CONFIRMED)
    assert state.phase is MissionPhase.NAVIGATE

    state = next_state(state, MissionEvent.BLOCKED)
    assert state.phase is MissionPhase.RECOVER

    state = next_state(state, MissionEvent.RECOVERED)
    assert state.phase is MissionPhase.NAVIGATE

    state = next_state(state, MissionEvent.GOAL_REACHED)
    assert state.phase is MissionPhase.HOLD


def test_rejected_candidate_returns_to_search_without_navigation_authority() -> None:
    state = MissionState(phase=MissionPhase.SEARCH)

    verifying = next_state(state, MissionEvent.TARGET_ACQUIRED)
    searching = next_state(verifying, MissionEvent.TARGET_REJECTED)

    assert verifying.phase is MissionPhase.VERIFY
    assert phase_limits(verifying.phase).learned_policy_phase_eligible is False
    assert searching.phase is MissionPhase.SEARCH
    assert searching.reason == "target_rejected"


def test_mission_state_machine_abort_wins_from_any_phase() -> None:
    state = MissionState(phase=MissionPhase.NAVIGATE)

    aborted = next_state(state, MissionEvent.ABORT_REQUESTED)

    assert aborted.phase is MissionPhase.ABORT
    assert aborted.reason == "abort_requested"


def test_recovery_resumes_search_without_inventing_target_acquisition() -> None:
    searching = MissionState(
        phase=MissionPhase.SEARCH,
        step=2,
        reason="takeoff_ready",
    )

    recovering = next_state(searching, MissionEvent.BLOCKED)
    resumed = next_state(recovering, MissionEvent.RECOVERED)

    assert recovering.resume_phase is MissionPhase.SEARCH
    assert resumed.phase is MissionPhase.SEARCH
    assert resumed.resume_phase is None


def test_recovery_resumes_verification_without_confirming_the_target() -> None:
    verifying = MissionState(
        phase=MissionPhase.VERIFY,
        step=3,
        reason="target_acquired",
    )

    recovering = next_state(verifying, MissionEvent.BLOCKED)
    resumed = next_state(recovering, MissionEvent.RECOVERED)

    assert recovering.resume_phase is MissionPhase.VERIFY
    assert resumed.phase is MissionPhase.VERIFY


def test_recovery_without_origin_is_rejected() -> None:
    invalid = MissionState(phase=MissionPhase.RECOVER)

    with pytest.raises(ValueError, match="no valid phase"):
        next_state(invalid, MissionEvent.RECOVERED)


def test_phase_limits_tighten_for_hold_and_abort() -> None:
    navigate = phase_limits(MissionPhase.NAVIGATE)
    hold = phase_limits(MissionPhase.HOLD)
    abort = phase_limits(MissionPhase.ABORT)

    assert navigate.max_speed_m_s > hold.max_speed_m_s
    assert abort.learned_policy_phase_eligible is False
    assert hold.command_source == "controller"


def test_invalid_mission_event_is_rejected_without_consuming_state() -> None:
    state = MissionState(phase=MissionPhase.SEARCH, step=7, reason="searching")

    with pytest.raises(ValueError, match="invalid mission transition"):
        next_state(state, MissionEvent.LANDED)

    assert state == MissionState(
        phase=MissionPhase.SEARCH,
        step=7,
        reason="searching",
    )


def test_abort_is_idempotent_after_abort_authority_takes_control() -> None:
    state = MissionState(phase=MissionPhase.ABORT, step=3, reason="timeout")

    assert next_state(state, MissionEvent.ABORT_REQUESTED) is state
    assert next_state(state, MissionEvent.TIMEOUT) is state
