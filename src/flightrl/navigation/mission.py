from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MissionPhase(str, Enum):
    PRE_FLIGHT = "preflight"
    TAKEOFF = "takeoff"
    SEARCH = "search"
    NAVIGATE = "navigate"
    RECOVER = "recover"
    HOLD = "hold"
    LAND = "land"
    ABORT = "abort"


class MissionEvent(str, Enum):
    PREFLIGHT_PASSED = "preflight_passed"
    TAKEOFF_READY = "takeoff_ready"
    TARGET_ACQUIRED = "target_acquired"
    TARGET_LOST = "target_lost"
    BLOCKED = "blocked"
    RECOVERED = "recovered"
    GOAL_REACHED = "goal_reached"
    LANDING_REQUESTED = "landing_requested"
    LANDED = "landed"
    TIMEOUT = "timeout"
    ABORT_REQUESTED = "abort_requested"


@dataclass(frozen=True)
class MissionState:
    phase: MissionPhase = MissionPhase.PRE_FLIGHT
    agent_id: str = "drone_0"
    step: int = 0
    reason: str = "initial"


@dataclass(frozen=True)
class PhaseLimits:
    max_speed_m_s: float
    max_yawrate_deg_s: float
    allow_learned_policy: bool
    command_source: str


TRANSITIONS: dict[tuple[MissionPhase, MissionEvent], MissionPhase] = {
    (MissionPhase.PRE_FLIGHT, MissionEvent.PREFLIGHT_PASSED): MissionPhase.TAKEOFF,
    (MissionPhase.TAKEOFF, MissionEvent.TAKEOFF_READY): MissionPhase.SEARCH,
    (MissionPhase.SEARCH, MissionEvent.TARGET_ACQUIRED): MissionPhase.NAVIGATE,
    (MissionPhase.NAVIGATE, MissionEvent.TARGET_LOST): MissionPhase.SEARCH,
    (MissionPhase.NAVIGATE, MissionEvent.BLOCKED): MissionPhase.RECOVER,
    (MissionPhase.SEARCH, MissionEvent.BLOCKED): MissionPhase.RECOVER,
    (MissionPhase.RECOVER, MissionEvent.RECOVERED): MissionPhase.NAVIGATE,
    (MissionPhase.NAVIGATE, MissionEvent.GOAL_REACHED): MissionPhase.HOLD,
    (MissionPhase.HOLD, MissionEvent.LANDING_REQUESTED): MissionPhase.LAND,
    (MissionPhase.LAND, MissionEvent.LANDED): MissionPhase.PRE_FLIGHT,
}


LIMITS: dict[MissionPhase, PhaseLimits] = {
    MissionPhase.PRE_FLIGHT: PhaseLimits(0.0, 0.0, False, "preflight"),
    MissionPhase.TAKEOFF: PhaseLimits(0.12, 20.0, False, "controller"),
    MissionPhase.SEARCH: PhaseLimits(0.20, 35.0, True, "policy"),
    MissionPhase.NAVIGATE: PhaseLimits(0.25, 45.0, True, "policy"),
    MissionPhase.RECOVER: PhaseLimits(0.16, 35.0, False, "controller"),
    MissionPhase.HOLD: PhaseLimits(0.08, 20.0, False, "controller"),
    MissionPhase.LAND: PhaseLimits(0.06, 15.0, False, "controller"),
    MissionPhase.ABORT: PhaseLimits(0.0, 0.0, False, "abort"),
}


def next_state(state: MissionState, event: MissionEvent) -> MissionState:
    if event is MissionEvent.ABORT_REQUESTED:
        return replace_phase(state, MissionPhase.ABORT, event.value)
    if event is MissionEvent.TIMEOUT:
        return replace_phase(state, MissionPhase.ABORT, event.value)
    next_phase = TRANSITIONS.get((state.phase, event), state.phase)
    return replace_phase(state, next_phase, event.value)


def replace_phase(state: MissionState, phase: MissionPhase, reason: str) -> MissionState:
    return MissionState(phase=phase, agent_id=state.agent_id, step=state.step + 1, reason=reason)


def phase_limits(phase: MissionPhase) -> PhaseLimits:
    return LIMITS[phase]
