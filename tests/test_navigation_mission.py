from __future__ import annotations

from flightrl.navigation.mission import MissionEvent, MissionPhase, MissionState, next_state, phase_limits


def test_mission_state_machine_runs_single_drone_navigation_flow() -> None:
    state = MissionState()

    state = next_state(state, MissionEvent.PREFLIGHT_PASSED)
    assert state.phase is MissionPhase.TAKEOFF
    assert state.agent_id == "drone_0"

    state = next_state(state, MissionEvent.TAKEOFF_READY)
    assert state.phase is MissionPhase.SEARCH

    state = next_state(state, MissionEvent.TARGET_ACQUIRED)
    assert state.phase is MissionPhase.NAVIGATE

    state = next_state(state, MissionEvent.BLOCKED)
    assert state.phase is MissionPhase.RECOVER

    state = next_state(state, MissionEvent.RECOVERED)
    assert state.phase is MissionPhase.NAVIGATE

    state = next_state(state, MissionEvent.GOAL_REACHED)
    assert state.phase is MissionPhase.HOLD


def test_mission_state_machine_abort_wins_from_any_phase() -> None:
    state = MissionState(phase=MissionPhase.NAVIGATE)

    aborted = next_state(state, MissionEvent.ABORT_REQUESTED)

    assert aborted.phase is MissionPhase.ABORT
    assert aborted.reason == "abort_requested"


def test_phase_limits_tighten_for_hold_and_abort() -> None:
    navigate = phase_limits(MissionPhase.NAVIGATE)
    hold = phase_limits(MissionPhase.HOLD)
    abort = phase_limits(MissionPhase.ABORT)

    assert navigate.max_speed_m_s > hold.max_speed_m_s
    assert abort.allow_learned_policy is False
    assert hold.command_source == "controller"
