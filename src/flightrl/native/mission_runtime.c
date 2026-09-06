#include "mission_runtime.h"


static const FlightRLMissionPhaseLimits PHASE_LIMITS[] = {
    {0.00f, 0.0f, 0u, FLIGHTRL_COMMAND_PREFLIGHT, 0u},
    {0.12f, 20.0f, 0u, FLIGHTRL_COMMAND_CONTROLLER, 0u},
    {0.20f, 35.0f, 1u, FLIGHTRL_COMMAND_POLICY, 0u},
    {0.08f, 20.0f, 0u, FLIGHTRL_COMMAND_PERCEPTION, 0u},
    {0.25f, 45.0f, 1u, FLIGHTRL_COMMAND_POLICY, 0u},
    {0.16f, 35.0f, 0u, FLIGHTRL_COMMAND_CONTROLLER, 0u},
    {0.08f, 20.0f, 0u, FLIGHTRL_COMMAND_CONTROLLER, 0u},
    {0.06f, 15.0f, 0u, FLIGHTRL_COMMAND_CONTROLLER, 0u},
    {0.00f, 0.0f, 0u, FLIGHTRL_COMMAND_ABORT, 0u},
};


static int valid_state(const FlightRLMissionState *state) {
    if (state == NULL) {
        return FLIGHTRL_MISSION_INVALID_ARGUMENT;
    }
    if (
        state->abi_version != FLIGHTRL_MISSION_ABI_VERSION ||
        state->struct_size != sizeof(FlightRLMissionState)
    ) {
        return FLIGHTRL_MISSION_INCOMPATIBLE_ABI;
    }
    if (
        state->phase >= FLIGHTRL_MISSION_PHASE_COUNT ||
        (
            state->resume_phase != FLIGHTRL_MISSION_NO_PHASE &&
            state->resume_phase >= FLIGHTRL_MISSION_PHASE_COUNT
        )
    ) {
        return FLIGHTRL_MISSION_INVALID_ARGUMENT;
    }
    return FLIGHTRL_MISSION_OK;
}


static int recoverable_phase(uint32_t phase) {
    return phase == FLIGHTRL_MISSION_SEARCH ||
        phase == FLIGHTRL_MISSION_VERIFY ||
        phase == FLIGHTRL_MISSION_NAVIGATE;
}


static int direct_transition(uint32_t phase, uint32_t event, uint32_t *next) {
    if (phase == FLIGHTRL_MISSION_PREFLIGHT && event == FLIGHTRL_EVENT_PREFLIGHT_PASSED) {
        *next = FLIGHTRL_MISSION_TAKEOFF;
    } else if (phase == FLIGHTRL_MISSION_TAKEOFF && event == FLIGHTRL_EVENT_TAKEOFF_READY) {
        *next = FLIGHTRL_MISSION_SEARCH;
    } else if (phase == FLIGHTRL_MISSION_SEARCH && event == FLIGHTRL_EVENT_TARGET_ACQUIRED) {
        *next = FLIGHTRL_MISSION_VERIFY;
    } else if (phase == FLIGHTRL_MISSION_VERIFY && event == FLIGHTRL_EVENT_TARGET_CONFIRMED) {
        *next = FLIGHTRL_MISSION_NAVIGATE;
    } else if (
        phase == FLIGHTRL_MISSION_VERIFY &&
        (event == FLIGHTRL_EVENT_TARGET_REJECTED || event == FLIGHTRL_EVENT_TARGET_LOST)
    ) {
        *next = FLIGHTRL_MISSION_SEARCH;
    } else if (phase == FLIGHTRL_MISSION_NAVIGATE && event == FLIGHTRL_EVENT_TARGET_LOST) {
        *next = FLIGHTRL_MISSION_SEARCH;
    } else if (phase == FLIGHTRL_MISSION_NAVIGATE && event == FLIGHTRL_EVENT_GOAL_REACHED) {
        *next = FLIGHTRL_MISSION_HOLD;
    } else if (phase == FLIGHTRL_MISSION_HOLD && event == FLIGHTRL_EVENT_LANDING_REQUESTED) {
        *next = FLIGHTRL_MISSION_LAND;
    } else if (phase == FLIGHTRL_MISSION_LAND && event == FLIGHTRL_EVENT_LANDED) {
        *next = FLIGHTRL_MISSION_PREFLIGHT;
    } else {
        return FLIGHTRL_MISSION_INVALID_TRANSITION;
    }
    return FLIGHTRL_MISSION_OK;
}


int flightrl_mission_next(FlightRLMissionState *state, uint32_t event) {
    uint32_t next;
    int status = valid_state(state);
    if (status != FLIGHTRL_MISSION_OK) {
        return status;
    }
    if (event >= FLIGHTRL_EVENT_COUNT) {
        return FLIGHTRL_MISSION_INVALID_ARGUMENT;
    }
    if (state->phase == FLIGHTRL_MISSION_ABORT && (
        event == FLIGHTRL_EVENT_ABORT_REQUESTED || event == FLIGHTRL_EVENT_TIMEOUT
    )) {
        return FLIGHTRL_MISSION_OK;
    }
    if (event == FLIGHTRL_EVENT_ABORT_REQUESTED || event == FLIGHTRL_EVENT_TIMEOUT) {
        next = FLIGHTRL_MISSION_ABORT;
    } else if (event == FLIGHTRL_EVENT_BLOCKED && recoverable_phase(state->phase)) {
        state->resume_phase = state->phase;
        state->phase = FLIGHTRL_MISSION_RECOVER;
        state->step += 1u;
        return FLIGHTRL_MISSION_OK;
    } else if (event == FLIGHTRL_EVENT_RECOVERED && state->phase == FLIGHTRL_MISSION_RECOVER) {
        if (!recoverable_phase(state->resume_phase)) {
            return FLIGHTRL_MISSION_INVALID_TRANSITION;
        }
        next = state->resume_phase;
    } else {
        status = direct_transition(state->phase, event, &next);
        if (status != FLIGHTRL_MISSION_OK) {
            return status;
        }
    }
    state->phase = next;
    state->resume_phase = FLIGHTRL_MISSION_NO_PHASE;
    state->step += 1u;
    return FLIGHTRL_MISSION_OK;
}


int flightrl_mission_phase_limits(
    uint32_t phase,
    FlightRLMissionPhaseLimits *limits
) {
    if (limits == NULL || phase >= FLIGHTRL_MISSION_PHASE_COUNT) {
        return FLIGHTRL_MISSION_INVALID_ARGUMENT;
    }
    *limits = PHASE_LIMITS[phase];
    return FLIGHTRL_MISSION_OK;
}
